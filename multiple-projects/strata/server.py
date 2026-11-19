"""จัดการ llama.cpp server: ติดตั้งไบนารี, เลือก quant, สตาร์ท/หยุด, MoE offload, วัด VRAM จริง และ speculative decoding."""
from __future__ import annotations

import concurrent.futures
import contextlib
import fnmatch
import functools
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from rich import print as rprint
from rich.console import Console

from .config import (
    ALL_PRESET_KEYS,
    DEFAULT_PORT,
    DRAFT_CATALOG,
    DRAFT_DIR,
    LLAMA_LOG_FILE,
    MODEL_CATALOG,
    MODEL_DIR,
    NVIDIA_LIB_DIRS,
    SPEC_STATE,
    S,
)
from .utils import _download_and_extract

if TYPE_CHECKING:  # openai ถูก import ตอนเรียกใช้จริง (ให้ import โมดูลนี้ได้เร็ว)
    from openai import AsyncOpenAI, OpenAI

LLAMA_SERVER_PROC: subprocess.Popen | None = None

NIX_STORE = "/nix/store/"
LLAMA_CPP_DIR = "/content/.llama-cpp"
WRAPPER_REV = "# strata-wrapper-v3"  # v3 = กรอง /nix/store + ตั้ง CUDA_SCALE_LAUNCH_QUEUES=4 สำหรับ multi-GPU


def sanitize_ld_library_path(value: str | None) -> str:
    """ตัด path ของ Nix ออกจาก LD_LIBRARY_PATH

    ไบนารี prebuilt (llama-server) ต้องใช้ glibc ของระบบ — ถ้าไปหยิบ libc/libdl ของ Nix
    จะได้ error `version 'GLIBC_ABI_DT_X86_64_PLT' not found`
    """
    keep: list = []
    for part in (value or "").split(os.pathsep):
        p = part.strip()
        if not p or p.startswith(NIX_STORE) or p in keep:
            continue
        keep.append(p)
    return os.pathsep.join(keep)


def server_env(base: dict | None = None) -> dict:
    """env สำหรับรันไบนารี prebuilt: ไม่มี lib ของ Nix + เปิด CUDA_SCALE_LAUNCH_QUEUES=4 ลดความหน่วงบน 2×T4"""
    env = dict(os.environ if base is None else base)
    env["LD_LIBRARY_PATH"] = sanitize_ld_library_path(env.get("LD_LIBRARY_PATH"))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4")
    if NIX_STORE in (env.get("LD_PRELOAD") or ""):
        env.pop("LD_PRELOAD", None)
    return env


def start_error_hint(tail: str) -> str:
    """แนบคำใบ้ให้แก้ได้เองเมื่อ server ตายเพราะ lib ไม่เข้ากัน"""
    text = tail or ""
    if "GLIBC" in text:
        return (
            "\n💡 ไบนารี prebuilt หยิบ glibc ของ Nix มาผสมกับของระบบ — โค้ดนี้ตัด path `/nix/store` "
            "ออกจาก LD_LIBRARY_PATH ให้แล้ว ถ้ายังเจอให้ลบ /usr/local/bin/llama-server แล้วกดติดตั้งใหม่ "
            "(wrapper จะถูกสร้างพร้อมตัวกรอง)"
        )
    if "argument has been removed" in text:
        return (
            "\n💡 llama.cpp รุ่นนี้เปลี่ยนชื่อ option ของ speculative decoding — โค้ดจะลองสตาร์ทแบบปิด spec "
            "ให้อัตโนมัติ (หรือเลือกโหมด 'off' ใน UI) แล้วจึงค่อยเปิดใหม่เมื่อชื่อ option ถูกต้อง"
        )
    if "CUDA" in text or "cuda" in text:
        return "\n💡 ตรวจว่า Colab เปิด GPU อยู่ (Runtime → Change runtime type → GPU) และไดรเวอร์ CUDA ยังทำงาน"
    return ""


def check_disk_space(target_dir: str = "/content", required_gb: float | None = None) -> dict:
    cp = Path(target_dir)
    while not cp.exists() and cp != cp.parent:
        cp = cp.parent
    u = shutil.disk_usage(cp)
    t_gb, u_gb, f_gb = u.total / (1024**3), u.used / (1024**3), u.free / (1024**3)
    col = "green" if (required_gb is None or f_gb >= required_gb) else "bold red"
    extra = "" if required_gb is None else (f" — [bold green]✅ พอ ({required_gb:.1f}GB)[/bold green]" if f_gb >= required_gb else f" — [bold red]❌ ขาด {required_gb - f_gb:.1f}GB[/bold red]")
    Console().print(f"💾 [bold]Disk ({cp}):[/bold] ว่าง [{col}]{f_gb:.2f} GB[/{col}] / {t_gb:.2f} GB{extra}")
    return {"total_gb": t_gb, "used_gb": u_gb, "free_gb": f_gb}

_GPU_CACHE: dict[str, object] = {"t": 0.0, "devices": []}
GPU_CACHE_TTL = 10.0


def _find_smi() -> str | None:
    """หา nvidia-smi (แยกฟังก์ชันไว้ให้เทสต์แทนได้โดยไม่ต้องมี GPU จริง)"""
    return next(
        (c for c in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi", shutil.which("nvidia-smi")) if c and Path(c).exists()),
        None,
    )


def gpu_devices(ttl: float = GPU_CACHE_TTL) -> list[dict]:
    """รายการ GPU ทั้งหมด (index/ชื่อ/VRAM ว่าง-รวม/UUID) — cache สั้น ๆ กันเรียก nvidia-smi ถี่ตอน UI รีเฟรช"""
    now = time.monotonic()
    if ttl > 0 and _GPU_CACHE["devices"] and (now - float(_GPU_CACHE["t"])) < ttl:
        return list(_GPU_CACHE["devices"])  # type: ignore[arg-type]
    smi = _find_smi()
    devices: list[dict] = []
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=index,uuid,name,memory.total,memory.free,compute_cap", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            for ln in out.splitlines():
                if ln.strip():
                    parts = [x.strip() for x in ln.split(",")]
                    if len(parts) >= 6:
                        idx, uuid_s, nm, tot, fr, cc = parts[:6]
                    else:
                        idx, nm, tot, fr, cc = parts[:5]
                        uuid_s = f"GPU-{idx}"
                    devices.append({
                        "index": int(idx), "uuid": uuid_s, "name": nm,
                        "vram_total_gb": float(tot) / 1024, "vram_free_gb": float(fr) / 1024,
                        "compute_cap": cc,
                    })
        except Exception:  # noqa: BLE001
            devices = []
    _GPU_CACHE.update({"t": now, "devices": devices})
    return list(devices)


def _split_ratios(free: list[float]) -> list[float]:
    """สัดส่วน VRAM ว่างต่อการ์ด (ปัด 2 ตำแหน่ง แล้วปรับตัวสุดท้ายให้รวมได้ 1.00)"""
    total = sum(free) or 1.0
    ratios = [round(f / total, 2) for f in free]
    if ratios:
        ratios[-1] = round(1.0 - sum(ratios[:-1]), 2)
    return ratios


def plan_multi_gpu(model_gb: float, devices: list[dict] | None = None, overhead_gb: float = 1.2, reserve_gb: float = 0.6) -> dict:
    """วางแผนใช้ GPU ทุกใบให้เต็ม: แบ่ง layer ตาม VRAM ว่างจริง + เลือก main-gpu + บอกว่าโมเดลพอดีไหม"""
    devs = list(devices if devices is not None else gpu_devices())
    if not devs:
        return {
            "count": 0, "has_gpu": False, "tensor_split": [], "main_gpu": 0, "split_mode": "",
            "fits": False, "usable_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0, "per_gpu_gb": [],
            "label": "💻 CPU เท่านั้น (ไม่พบ GPU)",
        }
    free = [max(0.1, float(d.get("vram_free_gb", 0.0)) - reserve_gb) for d in devs]
    ratios = _split_ratios(free)
    usable = sum(free) - overhead_gb
    per_gpu = [round(model_gb * r, 2) for r in ratios]
    main_gpu = max(range(len(devs)), key=lambda k: free[k])
    names = " + ".join(str(d.get("name", f"GPU{i}")) for i, d in enumerate(devs))
    label = (
        f"🎮 {len(devs)}× GPU ({names}) · ใช้ได้ {usable:.1f} GB · "
        f"split {'/'.join(f'{r:.2f}' for r in ratios)} · main-gpu {main_gpu}"
    )
    return {
        "count": len(devs), "has_gpu": True, "tensor_split": ratios, "main_gpu": main_gpu,
        "split_mode": "layer", "fits": model_gb <= usable, "usable_gb": round(usable, 2),
        "free_gb": round(sum(float(d.get("vram_free_gb", 0.0)) for d in devs), 2),
        "total_gb": round(sum(float(d.get("vram_total_gb", 0.0)) for d in devs), 2),
        "per_gpu_gb": per_gpu, "label": label,
    }


def plan_ctx_size(free_gb: float, model_gb: float, kv_q8: bool = False, cap: int = 32768) -> int:
    """เลือก ctx ให้เต็ม VRAM ที่เหลือหลังโหลดน้ำหนัก (KV Q8_0 กินครึ่ง → เก็บ ctx ได้ประมาณสองเท่า)"""
    leftover = max(0.0, free_gb - model_gb - 1.5)
    if kv_q8:
        leftover *= 2
    for need, ctx in ((12.0, 32768), (6.0, 16384), (3.0, 8192), (1.2, 4096)):
        if leftover >= need:
            return min(ctx, cap)
    return 2048


def check_gpu_and_recommend(gb_per_layer: float = 0.5, kv_headroom_gb: float = 2.0, verbose: bool = True) -> dict:
    cores = os.cpu_count() or 2
    threads = max(1, cores - 1) if cores > 2 else cores
    ram_t, ram_a = 0.0, 0.0
    if Path("/proc/meminfo").exists():
        for ln in Path("/proc/meminfo").read_text().splitlines():
            pts = ln.split()
            if len(pts) >= 2 and ln.startswith("MemTotal:"):
                _, kb, *_ = pts; ram_t = int(kb) / (1024**2)
            elif len(pts) >= 2 and ln.startswith("MemAvailable:"):
                _, kb, *_ = pts; ram_a = int(kb) / (1024**2)
    gpus = gpu_devices()
    base = {"ram_total_gb": ram_t, "ram_avail_gb": ram_a, "recommended_threads": threads, "devices": gpus}
    if not gpus:
        return {**base, "has_gpu": False, "gpu_name": "CPU Only", "gpu_count": 0,
                "vram_total_gb": 0.0, "vram_free_gb": 0.0, "recommended_gpu_layers": 0, "recommended_ctx_size": 4096}
    tv, fv = sum(g["vram_total_gb"] for g in gpus), sum(g["vram_free_gb"] for g in gpus)
    names = ", ".join(f"{g['name']} ({g['vram_total_gb']:.0f}GB)" for g in gpus)
    layers = max(1, int(max(0.0, fv - kv_headroom_gb) / gb_per_layer))
    ctx = max(plan_ctx_size(fv, 0.0), 16384 if tv >= 28 else (8192 if tv >= 20 else 4096))
    if verbose:
        Console().print(f"🎮 [bold green]GPU:[/bold green] {names} | VRAM ว่าง {fv:.2f}/{tv:.2f} GB | RAM ว่าง {ram_a:.1f} GB")
        if len(gpus) > 1:
            Console().print(f"[dim]   {plan_multi_gpu(0.0, gpus)['label']}[/dim]")
    return {**base, "has_gpu": True, "gpu_name": names, "gpu_count": len(gpus),
            "vram_total_gb": tv, "vram_free_gb": fv, "recommended_gpu_layers": layers, "recommended_ctx_size": ctx}


# ---------------------------------------------------------------------------
# 🔬 ระบบตรวจสอบขนาดจริงและคำนวณ VRAM Fit ตามฮาร์ดแวร์และหัวไฟล์ GGUF จริง (ไม่ Hardcode)
# ---------------------------------------------------------------------------
_MODEL_SPEC_CACHE: dict[str, dict] = {}
_PROBE_LOCK = threading.Lock()
_PROBE_STARTED = False


def _num_or_avg(val: object, default: float = 0.0) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and val and all(isinstance(x, (int, float)) for x in val):
        return float(sum(val)) / len(val)
    return default


def parse_gguf_header_bytes(raw: bytes) -> dict:
    """อ่านค่าสถาปัตยกรรมจริงจากไบต์หัวไฟล์ GGUF v2/v3 (ใช้ได้ทั้งจากดิสก์และ HTTP Range 64KB)"""
    if len(raw) < 24 or raw[:4] != b"GGUF":
        return {}
    ver = struct.unpack_from("<I", raw, 4)[0]
    if ver not in (2, 3):
        return {}
    _n_tensors, n_kv = struct.unpack_from("<QQ", raw, 8)
    pos = 24
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    meta: dict = {}

    def _rv(vt: int, p: int):
        if vt in scalars:
            fmt = scalars[vt]
            sz = struct.calcsize(fmt)
            if p + sz > len(raw):
                raise EOFError
            return struct.unpack_from(fmt, raw, p)[0], p + sz
        if vt == 8:
            if p + 8 > len(raw):
                raise EOFError
            ln = struct.unpack_from("<Q", raw, p)[0]
            p += 8
            if p + ln > len(raw):
                raise EOFError
            return raw[p : p + ln].decode("utf-8", errors="ignore"), p + ln
        if vt == 9:
            if p + 12 > len(raw):
                raise EOFError
            et, cnt = struct.unpack_from("<IQ", raw, p)
            p += 12
            arr = []
            for _ in range(cnt):
                v, p = _rv(et, p)
                if len(arr) < 512:
                    arr.append(v)
            return arr, p
        raise ValueError(vt)

    try:
        for _ in range(min(int(n_kv), 5000)):
            if pos + 8 > len(raw):
                break
            klen = struct.unpack_from("<Q", raw, pos)[0]
            pos += 8
            if klen > 4096 or pos + klen + 4 > len(raw):
                break
            kname = raw[pos : pos + klen].decode("utf-8", errors="ignore")
            pos += klen
            vtype = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            if kname.startswith("tokenizer.ggml."):
                break
            val, pos = _rv(vtype, pos)
            if kname in ("general.architecture", "general.file_type") or any(
                kname.endswith(s)
                for s in (
                    ".block_count",
                    ".embedding_length",
                    ".attention.head_count",
                    ".attention.head_count_kv",
                    ".attention.key_length",
                    ".attention.value_length",
                    ".expert_count",
                )
            ):
                meta[kname] = val
    except (EOFError, ValueError, struct.error):
        pass

    n_layers = int(next((_num_or_avg(v) for k, v in meta.items() if k.endswith(".block_count")), 0))
    embd = _num_or_avg(next((v for k, v in meta.items() if k.endswith(".embedding_length")), 0))
    n_head = _num_or_avg(next((v for k, v in meta.items() if k.endswith(".attention.head_count")), 0))
    n_head_kv = _num_or_avg(next((v for k, v in meta.items() if k.endswith(".attention.head_count_kv")), n_head))
    k_len = _num_or_avg(next((v for k, v in meta.items() if k.endswith(".attention.key_length")), 0))
    v_len = _num_or_avg(next((v for k, v in meta.items() if k.endswith(".attention.value_length")), 0))
    if k_len <= 0 and embd > 0 and n_head > 0:
        k_len = embd / n_head
    if v_len <= 0:
        v_len = k_len
    n_experts = int(next((_num_or_avg(v) for k, v in meta.items() if k.endswith(".expert_count")), 0))
    kv_fp16_bpt = int(n_layers * n_head_kv * (k_len + v_len) * 2.0) if (n_layers > 0 and n_head_kv > 0 and k_len > 0) else 0
    return {
        "arch": str(meta.get("general.architecture") or ""),
        "n_layers": n_layers,
        "n_experts": n_experts,
        "kv_fp16_bytes_per_tok": kv_fp16_bpt,
    }


def measure_live_memory(ttl: float = 3.0) -> dict:
    """อ่านค่า VRAM ว่างจริงต่อการ์ด (รวม VRAM ที่โปรเซส llama-server ของเราจะคืนให้เมื่อสลับโมเดล) + RAM จริงจาก /proc/meminfo"""
    devs = gpu_devices(ttl=ttl)
    ram_total_gb, ram_avail_gb = 0.0, 0.0
    if Path("/proc/meminfo").is_file():
        for ln in Path("/proc/meminfo").read_text(errors="ignore").splitlines():
            pts = ln.split()
            if len(pts) >= 2 and ln.startswith("MemTotal:"):
                ram_total_gb = int(pts[1]) / (1024**2)
            elif len(pts) >= 2 and ln.startswith("MemAvailable:"):
                ram_avail_gb = int(pts[1]) / (1024**2)
    if not devs:
        return {
            "has_gpu": False, "gpu_count": 0, "effective_vram_gb": 0.0, "max_single_gpu_gb": 0.0,
            "raw_free_vram_gb": 0.0, "total_vram_gb": 0.0, "ram_avail_gb": ram_avail_gb, "ram_total_gb": ram_total_gb,
        }

    # ตรวจว่า llama-server กำลังถือ VRAM บนการ์ดไหนอยู่กี่ MB (เมื่อกดสลับโมเดล พื้นที่ส่วนนี้จะถูกคืนกลับมาใช้ได้จริง)
    reclaim_by_uuid: dict[str, float] = {}
    smi = _find_smi()
    if smi:
        with contextlib.suppress(Exception):
            out = subprocess.check_output(
                [smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL, timeout=4,
            ).strip()
            for ln in out.splitlines():
                pts = [x.strip() for x in ln.split(",")]
                if len(pts) >= 4:
                    guuid, _pid, pname, mb_s = pts[:4]
                    if "llama-server" in pname:
                        reclaim_by_uuid[guuid] = reclaim_by_uuid.get(guuid, 0.0) + (float(mb_s or 0) / 1024.0)

    cuda_ctx_reserve = 0.45  # CUDA driver/cublas context จริงต่อการ์ด
    per_gpu_eff: list[float] = []
    for d in devs:
        raw_f = float(d.get("vram_free_gb", 0.0))
        rec = reclaim_by_uuid.get(str(d.get("uuid", "")), 0.0)
        eff = min(float(d.get("vram_total_gb", raw_f)), raw_f + rec) - cuda_ctx_reserve
        per_gpu_eff.append(max(0.0, eff))

    return {
        "has_gpu": True,
        "gpu_count": len(devs),
        "effective_vram_gb": round(sum(per_gpu_eff), 2),
        "max_single_gpu_gb": round(max(per_gpu_eff), 2),
        "raw_free_vram_gb": round(sum(float(d.get("vram_free_gb", 0.0)) for d in devs), 2),
        "total_vram_gb": round(sum(float(d.get("vram_total_gb", 0.0)) for d in devs), 2),
        "ram_avail_gb": round(ram_avail_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
    }


def probe_model_spec(key: str, model_dir: str = MODEL_DIR, allow_network: bool = False) -> dict | None:
    """วัดขนาดไบต์จริงและโครงสร้างเลเยอร์ของโมเดลจากไฟล์บนดิสก์ หรือจาก Hugging Face API + GGUF Header Range 64KB"""
    sub = Path(model_dir) / key
    cfg = MODEL_CATALOG.get(key)
    if not cfg:
        return None
    is_im = cfg.get("kind") == "image"
    is_lora = cfg.get("kind") == "lora"

    # 1) ถ้ามีไฟล์บนดิสก์แล้ว — วัดขนาดจริงจาก stat() และอ่านไบนารีหัวไฟล์ GGUF บนดิสก์ 100%
    if sub.is_dir():
        if is_lora:
            sfs = [p for p in sub.rglob("*.safetensors") if p.is_file() and not p.name.endswith(".part") and p.stat().st_size > 0]
            if sfs:
                spec = {
                    "source": "disk", "kind": "lora",
                    "total_bytes": sum(p.stat().st_size for p in sfs),
                }
                with _PROBE_LOCK:
                    _MODEL_SPEC_CACHE[key] = spec
                return spec
        elif is_im:
            targets = cfg.get("targets") or {}
            if targets and all((sub / rel).is_file() for rel in targets):
                stage_b = {"diffusion_models": 0, "text_encoders": 0, "vae": 0, "other": 0}
                for rel, role in targets.items():
                    sz = (sub / rel).stat().st_size
                    stage_b[role if role in stage_b else "other"] += sz
                spec = {
                    "source": "disk", "kind": "image",
                    "total_bytes": sum(stage_b.values()),
                    "diff_gb": stage_b["diffusion_models"] / 1024**3,
                    "te_gb": stage_b["text_encoders"] / 1024**3,
                    "vae_gb": stage_b["vae"] / 1024**3,
                }
                with _PROBE_LOCK:
                    _MODEL_SPEC_CACHE[key] = spec
                return spec
        else:
            ggufs = [p for p in sorted(sub.rglob("*.gguf")) if p.is_file()]
            mains = [p for p in ggufs if not _is_aux_gguf(p)]
            if mains:
                main_b = sum(p.stat().st_size for p in mains)
                mm_b = sum(p.stat().st_size for p in ggufs if "mmproj" in p.name.lower())
                hdr = b""
                with contextlib.suppress(OSError):
                    with mains[0].open("rb") as fh:
                        hdr = fh.read(65536)
                arch = parse_gguf_header_bytes(hdr)
                spec = {
                    "source": "disk", "kind": "text",
                    "total_bytes": sum(p.stat().st_size for p in ggufs),
                    "main_gb": main_b / 1024**3,
                    "mmproj_gb": mm_b / 1024**3,
                    **arch,
                }
                with _PROBE_LOCK:
                    _MODEL_SPEC_CACHE[key] = spec
                return spec

    with _PROBE_LOCK:
        cached = _MODEL_SPEC_CACHE.get(key)
    if cached is not None or not allow_network:
        return cached

    # 2) ยังไม่ได้โหลดลงดิสก์ — ดึงขนาดไบต์จริงจาก Hugging Face repo_info + อ่านหัวไฟล์ GGUF 64KB ผ่าน HTTP Range
    try:
        from huggingface_hub import HfApi, hf_hub_url

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=token)
        sources = cfg.get("sources") or {}
        inc = list(cfg.get("include") or ["*"])
        exc = list(cfg.get("exclude") or [])
        by_repo: dict[str, list[str]] = {}
        for pat in inc:
            by_repo.setdefault(sources.get(pat, cfg["repo_id"]), []).append(pat)

        matched_files: list[tuple[str, str, int]] = []
        for repo_id, pats in by_repo.items():
            info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
            for sib in (info.siblings or []):
                fn = sib.rfilename
                bn = Path(fn).name
                if not any(fnmatch.fnmatch(fn, p) or fnmatch.fnmatch(bn, p) for p in pats):
                    continue
                if exc and any(fnmatch.fnmatch(fn, p) or fnmatch.fnmatch(Path(fn).name, p) for p in exc):
                    continue
                matched_files.append((repo_id, fn, int(getattr(sib, "size", 0) or 0)))

        if not matched_files:
            return None

        if is_lora:
            spec = {
                "source": "hf_live", "kind": "lora",
                "total_bytes": sum(sz for _, _, sz in matched_files),
            }
        elif is_im:
            targets = cfg.get("targets") or {}
            stage_b = {"diffusion_models": 0, "text_encoders": 0, "vae": 0, "other": 0}
            for _r, fn, sz in matched_files:
                role = targets.get(fn) or targets.get(Path(fn).name) or "other"
                stage_b[role if role in stage_b else "other"] += sz
            spec = {
                "source": "hf_live", "kind": "image",
                "total_bytes": sum(sz for _, _, sz in matched_files),
                "diff_gb": stage_b["diffusion_models"] / 1024**3,
                "te_gb": stage_b["text_encoders"] / 1024**3,
                "vae_gb": stage_b["vae"] / 1024**3,
            }
        else:
            mains_r = [(r, fn, sz) for r, fn, sz in matched_files if fn.lower().endswith(".gguf") and not _is_aux_gguf(Path(fn))]
            main_b = sum(sz for _, _, sz in mains_r)
            mm_b = sum(sz for _, fn, sz in matched_files if "mmproj" in fn.lower())
            arch = {"arch": "", "n_layers": 0, "n_experts": 0, "kv_fp16_bytes_per_tok": 0}
            if mains_r:
                r0, fn0, _ = sorted(mains_r, key=lambda x: x[1])[0]
                url = hf_hub_url(repo_id=r0, filename=fn0, repo_type="model")
                hdrs = {"User-Agent": "strata-vram-probe/1.0", "Range": "bytes=0-65535"}
                if token:
                    hdrs["Authorization"] = f"Bearer {token}"
                with contextlib.suppress(Exception):
                    req = urllib.request.Request(url, headers=hdrs)
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        arch = parse_gguf_header_bytes(resp.read(65536))
            spec = {
                "source": "hf_live", "kind": "text",
                "total_bytes": sum(sz for _, _, sz in matched_files),
                "main_gb": main_b / 1024**3,
                "mmproj_gb": mm_b / 1024**3,
                **arch,
            }
        with _PROBE_LOCK:
            _MODEL_SPEC_CACHE[key] = spec
        return spec
    except Exception:
        return None


def start_vram_probe_workers(keys: list[str] | None = None, model_dir: str = MODEL_DIR) -> None:
    """รันเธรดเบื้องหลังดึงขนาดไฟล์จริง + หัวไฟล์ GGUF จาก Hugging Face ขนานกันโดยไม่บล็อก UI"""
    global _PROBE_STARTED
    with _PROBE_LOCK:
        if _PROBE_STARTED:
            return
        _PROBE_STARTED = True
    target_keys = list(keys if keys is not None else ALL_PRESET_KEYS)

    def _runner():
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for k in target_keys:
                pool.submit(probe_model_spec, k, model_dir, True)

    threading.Thread(target=_runner, daemon=True).start()


def inspect_model_vram_fit(
    key: str,
    model_dir: str = MODEL_DIR,
    mem: dict | None = None,
    kv_q8: bool | None = None,
    ctx_tokens: int = 4096,
) -> dict:
    """คำนวณความเข้ากันได้ของ VRAM จริงจากข้อมูลที่วัดได้ (ไม่มีค่า Hardcode)"""
    hw_mem = mem if mem is not None else measure_live_memory()
    spec = probe_model_spec(key, model_dir=model_dir, allow_network=False)
    if spec is None:
        return {"fit": "probing", "fit_label": "⏳ กำลังวัดขนาดจริง...", "measured_gb": None, "source": "pending"}

    total_file_gb = round(float(spec["total_bytes"]) / 1024**3, 2)
    ram_budget = max(0.0, float(hw_mem["ram_avail_gb"]) - 1.5)

    # กรณีโมเดลเสริม LoRA (ขนาดเล็ก 40MB–700MB พอดี VRAM เสมอ)
    if spec.get("kind") == "lora":
        tot_b = int(spec["total_bytes"])
        return {
            "fit": "ok",
            "fit_label": f"🎮 พอดี VRAM (+LoRA {tot_b / 1024**3:.2f}GB)",
            "measured_gb": total_file_gb,
            "source": spec["source"],
        }

    # กรณีโมเดลสร้างภาพ (ComfyUI รันบน GPU เดี่ยวใบที่ว่างที่สุด ไม่ได้หารข้าม 2 ใบเหมือน llama.cpp)
    if spec["kind"] == "image":
        diff_gb = float(spec.get("diff_gb", 0.0))
        te_gb = float(spec.get("te_gb", 0.0))
        vae_gb = float(spec.get("vae_gb", 0.0))
        peak_gpu_gb = max(diff_gb, te_gb) + vae_gb + 1.15  # ช่วงพีคขณะถอดรหัสภาพ + activation จริง
        if not hw_mem["has_gpu"]:
            ok_cpu = (diff_gb + te_gb + vae_gb + 1.5) <= ram_budget
            return {
                "fit": "warn" if ok_cpu else "over",
                "fit_label": f"🖥️ โหมด CPU ({total_file_gb:.1f}/{ram_budget:.1f}GB RAM)" if ok_cpu else f"🚫 เกิน RAM ({total_file_gb:.1f}>{ram_budget:.1f}GB)",
                "measured_gb": total_file_gb, "source": spec["source"],
            }
        sgpu = float(hw_mem["max_single_gpu_gb"])
        if peak_gpu_gb <= sgpu:
            return {
                "fit": "ok",
                "fit_label": f"🎮 พอดี VRAM ({peak_gpu_gb:.1f}/{sgpu:.1f}GB)",
                "measured_gb": total_file_gb, "source": spec["source"],
            }
        if max(diff_gb, te_gb) <= (sgpu + ram_budget):
            return {
                "fit": "warn",
                "fit_label": f"⚠️ LowVRAM สลับ RAM ({peak_gpu_gb:.1f}/{sgpu:.1f}GB)",
                "measured_gb": total_file_gb, "source": spec["source"],
            }
        return {
            "fit": "over",
            "fit_label": f"🚫 เกิน VRAM+RAM ({peak_gpu_gb:.1f}>{sgpu + ram_budget:.1f}GB)",
            "measured_gb": total_file_gb, "source": spec["source"],
        }

    # กรณีโมเดลภาษา/Multimodal GGUF (llama-server รองรับแบ่งเลเยอร์รวมทุกการ์ด GPU)
    main_gb = float(spec.get("main_gb", 0.0))
    mm_gb = float(spec.get("mmproj_gb", 0.0))
    n_layers = int(spec.get("n_layers", 0))
    n_experts = int(spec.get("n_experts", 0))
    kv_bpt = int(spec.get("kv_fp16_bytes_per_tok", 0))
    use_q8 = bool(SPEC_STATE.get("kv_q8", True) if kv_q8 is None else kv_q8)

    if kv_bpt > 0:
        kv_scale = 0.53125 if use_q8 else 1.0  # Q8_0 = 34 bytes / 32 ค่า เทียบ FP16 = 64 bytes / 32 ค่า
        kv_gb = (kv_bpt * kv_scale * ctx_tokens) / (1024**3)
    else:
        # กรณีหัวไฟล์ถูกตัดก่อนถึงค่า head_kv คำนวณขั้นต่ำตามสัดส่วนพารามิเตอร์จริงของไฟล์
        kv_gb = max(0.25, main_gb * (0.035 if use_q8 else 0.068) * (ctx_tokens / 4096.0))

    compute_gb = 0.55 + (mm_gb if mm_gb > 0 else 0.0)
    need_vram_gb = main_gb + kv_gb + compute_gb

    if not hw_mem["has_gpu"]:
        if need_vram_gb <= ram_budget:
            return {"fit": "warn", "fit_label": f"🖥️ รันบน RAM ({need_vram_gb:.1f}/{ram_budget:.1f}GB)", "measured_gb": total_file_gb, "source": spec["source"]}
        return {"fit": "over", "fit_label": f"🚫 เกิน RAM ({need_vram_gb:.1f}>{ram_budget:.1f}GB)", "measured_gb": total_file_gb, "source": spec["source"]}

    eff_vram = float(hw_mem["effective_vram_gb"])
    lyr_txt = f" • {n_layers}/{n_layers} ชั้น" if n_layers > 0 else ""

    if need_vram_gb <= eff_vram:
        return {
            "fit": "ok",
            "fit_label": f"🎮 พอดี VRAM ({need_vram_gb:.1f}/{eff_vram:.1f}GB{lyr_txt})",
            "measured_gb": total_file_gb, "source": spec["source"],
        }

    offload_gb = need_vram_gb - eff_vram
    if n_experts > 1 and (main_gb * 0.28 + kv_gb + compute_gb) <= eff_vram and offload_gb <= ram_budget:
        return {
            "fit": "moe",
            "fit_label": f"⚡ พอดี GPU+MoE (GPU {eff_vram:.1f} + RAM {offload_gb:.1f}GB)",
            "measured_gb": total_file_gb, "source": spec["source"],
        }

    avail_for_weights = eff_vram - kv_gb - compute_gb
    if avail_for_weights >= 0.4 and offload_gb <= ram_budget:
        if n_layers > 0 and main_gb > 0:
            gpu_lyrs = max(1, min(n_layers - 1, int(avail_for_weights / (main_gb / n_layers))))
            return {
                "fit": "warn",
                "fit_label": f"⚠️ แบ่ง GPU {gpu_lyrs}/{n_layers} ชั้น (+RAM {offload_gb:.1f}GB)",
                "measured_gb": total_file_gb, "source": spec["source"],
            }
        return {
            "fit": "warn",
            "fit_label": f"⚠️ แบ่ง GPU+RAM (+RAM {offload_gb:.1f}GB)",
            "measured_gb": total_file_gb, "source": spec["source"],
        }

    return {
        "fit": "over",
        "fit_label": f"🚫 เกิน VRAM+RAM ({need_vram_gb:.1f}>{eff_vram + ram_budget:.1f}GB)",
        "measured_gb": total_file_gb, "source": spec["source"],
    }


def _write_wrapper(srv_bin: Path, dest: Path, wrapper: Path = Path("/usr/local/bin/llama-server")) -> Path:
    """สร้าง wrapper ที่ตัด path ของ Nix ออกจาก LD_LIBRARY_PATH และเปิด CUDA_SCALE_LAUNCH_QUEUES=4"""
    lib_dirs = os.pathsep.join(sorted({str(p.parent) for p in dest.rglob("*.so*")}))
    base = [d for d in (NVIDIA_LIB_DIRS, lib_dirs, str(srv_bin.parent)) if d]
    body = (
        "#!/usr/bin/env bash\n"
        f"{WRAPPER_REV}\n"
        "# ลบ lib ของ Nix ออกจาก LD_LIBRARY_PATH (glibc ของ Nix ไม่เข้ากับไบนารี prebuilt)\n"
        "unset LD_PRELOAD\n"
        'export CUDA_SCALE_LAUNCH_QUEUES="${CUDA_SCALE_LAUNCH_QUEUES:-4}"\n'
        '_keep=""\n'
        'while IFS= read -r _p; do\n'
        '  case "$_p" in /nix/store/*) continue;; esac\n'
        '  _keep="${_keep:+$_keep:}$_p"\n'
        'done < <(printf "%s\\n" "${LD_LIBRARY_PATH:-}" | tr ":" "\\n")\n'
        f'export LD_LIBRARY_PATH="{os.pathsep.join(base)}${{_keep:+:$_keep}}"\n'
        f'exec "{srv_bin}" "$@"\n'
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.unlink(missing_ok=True)
    wrapper.write_text(body, encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def install_llama_server(install_dir: str = LLAMA_CPP_DIR) -> str:
    wrapper = Path("/usr/local/bin/llama-server")
    if wrapper.exists():
        wtxt = wrapper.read_text(encoding="utf-8", errors="ignore")
        if ("\\" + S) in wtxt:
            wrapper.write_text(wtxt.replace("\\" + S, S), encoding="utf-8"); wrapper.chmod(0o755)
        old_bins = sorted(q for q in Path(install_dir).rglob("llama-server") if q.is_file())
        if WRAPPER_REV not in wtxt and old_bins:
            rprint("[dim]🔧 อัปเดต wrapper llama-server (ตัด lib ของ Nix + เปิด CUDA_SCALE_LAUNCH_QUEUES=4)[/dim]")
            return str(_write_wrapper(old_bins[0], Path(install_dir), wrapper))
        if subprocess.run([str(wrapper), "--version"], capture_output=True).returncode == 0:
            return str(wrapper)
    rprint("[bold cyan]⏳ กำลังติดตั้ง Prebuilt llama-server (CUDA 12)...[/bold cyan]")
    hw = check_gpu_and_recommend(verbose=False)
    gapi = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10", timeout=20, headers={"Accept": "application/vnd.github+json"})
    if gapi.status_code != 200:
        raise RuntimeError(f"ดึงรายการ release ของ llama.cpp ไม่สำเร็จ (HTTP {gapi.status_code}) — อาจติด rate limit ของ GitHub ลองรันใหม่ภายหลัง")
    rels = gapi.json()
    _pin = os.environ.get("LLAMA_CPP_TAG", "").strip()
    if _pin:
        rels = [r for r in rels if r.get("tag_name") == _pin] or rels
        rprint(f"[dim]📌 ใช้ release ที่กำหนด: {_pin}[/dim]")
    main_asset, cudart_asset = None, None
    prefs = ["bin-ubuntu-cuda-12", "bin-ubuntu-vulkan-x64", "bin-ubuntu-x64"] if hw["has_gpu"] else ["bin-ubuntu-x64"]
    for pat in prefs:
        for rel in rels:
            assets = rel.get("assets", [])
            for a in assets:
                nm = a.get("name", "")
                if nm.startswith("llama-") and pat in nm and "x64" in nm and (nm.endswith(".tar.gz") or nm.endswith(".zip")):
                    main_asset = a
                    if "cuda-12" in pat:
                        cudart_asset = next((c for c in assets if c.get("name", "").startswith("cudart-") and "ubuntu-cuda-12" in c.get("name", "") and "x64" in c.get("name", "")), None)
                    break
            if main_asset:
                break
        if main_asset:
            break
    if not main_asset:
        raise RuntimeError("ไม่พบไฟล์ Prebuilt llama-server")
    dest = Path(install_dir); shutil.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True, exist_ok=True)
    if cudart_asset:
        _download_and_extract(cudart_asset["browser_download_url"], cudart_asset["name"], dest,
                              label=f"CUDA runtime {cudart_asset['name']}")
    _download_and_extract(main_asset["browser_download_url"], main_asset["name"], dest,
                          label=f"llama-server {main_asset['name']}")
    rprint("[cyan]   🔧 ตั้งค่า wrapper ให้ชี้ไบนารีใหม่…[/cyan]")
    srv_bins = [p for p in dest.rglob("llama-server") if p.is_file()]
    if not srv_bins:
        raise RuntimeError("ไม่พบไบนารี llama-server")
    srv_bin = next(iter(srv_bins)); srv_bin.chmod(0o755)
    _write_wrapper(srv_bin, dest, wrapper)
    rprint(f"[bold green]✅ ติดตั้ง llama-server สำเร็จ ({main_asset['name']})![/bold green]")
    return str(wrapper)

def get_openai_client(port: int = DEFAULT_PORT, api_key: str = "strata", async_mode: bool = False, timeout: float = 300.0) -> OpenAI | AsyncOpenAI:
    url = f"http://127.0.0.1:{port}/v1"
    from openai import AsyncOpenAI, OpenAI

    return AsyncOpenAI(base_url=url, api_key=api_key, timeout=timeout) if async_mode else OpenAI(base_url=url, api_key=api_key, timeout=timeout)

_SRV_CACHE: dict[int, tuple[float, tuple[bool, str]]] = {}

_SRV_CACHE_TTL = 1.5

def _probe_server(port: int) -> tuple[bool, str]:
    try:
        if requests.get(f"http://127.0.0.1:{port}/health", timeout=1.5).status_code != 200:
            return False, "loading"
        rm = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
        if rm.status_code == 200 and "json" in rm.headers.get("content-type", "").lower():
            d = rm.json()
            if isinstance(d, dict) and isinstance(d.get("data"), list) and d["data"]:
                return True, str(next(iter(d["data"])).get("id", "default"))
    except Exception:
        pass
    return False, "offline"

def _check_server_online(port: int = DEFAULT_PORT, ttl: float = _SRV_CACHE_TTL) -> tuple[bool, str]:
    """เช็กสุขภาพ server แบบ cache สั้น ๆ (UI เรียกบ่อย แต่ไม่ควรยิง HTTP ซ้ำทุกครั้ง) — ttl=0 เพื่อบังคับเช็กใหม่"""
    now = time.monotonic()
    hit = _SRV_CACHE.get(port)
    if hit and ttl > 0 and (now - hit[0]) < ttl:
        return hit[1]
    res = _probe_server(port)
    _SRV_CACHE[port] = (now, res)
    return res

def _invalidate_server_cache(port: int | None = None):
    if port is None:
        _SRV_CACHE.clear()
    else:
        _SRV_CACHE.pop(port, None)

_GGUF_LAYERS: dict[str, int] = {}
_GGUF_EXPERTS: dict[str, int] = {}

VOCAB_FAMILY_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        (
            "lumimaid-v02",
            "lumimaid-v0.2",
            "anubis",
            "nevoria",
            "lexi-llama31",
            "stheno",
            "dolphin3-llama",
            "euryale",
            "llama-3.",
            "llama3.",
            "llama-3-",
        ),
        "llama3",
    ),
    (
        (
            "lumimaid",
            "cydonia",
            "skyfall",
            "magnum",
            "rocinante",
            "nemomix",
            "dolphin3-r1",
            "dolphin-24b",
            "venice",
            "mistral-small",
            "ministral",
            "nemo",
        ),
        "tekken",
    ),
    (("noromaid", "wizardlm", "mixtral", "tinyllama"), "llama2"),
    (("qwen", "qwq", "deepseek-v4-pro-4b"), "qwen"),
    (("gemma", "muse-glimmer", "glimmer"), "gemma"),
)


def infer_vocab_family(alias: str, model_path: str | Path | None = None) -> str:
    """เดาชุดคำศัพท์ (Tokenizer Family) ของโมเดลหลัก เพื่อจับคู่กับโมเดล Draft ที่ใช้คำศัพท์ตรงกัน"""
    repo = MODEL_CATALOG.get(alias, {}).get("repo_id", "") if alias else ""
    hint = f"{alias} {repo} {Path(model_path).name if model_path else ''}".lower()
    for needles, fam in VOCAB_FAMILY_RULES:
        if any(n in hint for n in needles):
            return fam
    return ""


def find_mtp_draft(model_dir: str | Path, main: str | Path | None = None) -> str:
    """หาไฟล์ MTP head/draft ที่มากับเรพอ (เช่น `mtp-gemma-4-26B-A4B-it.gguf`)"""
    base = Path(model_dir)
    if not base.is_dir():
        return ""
    main_name = Path(main).name if main else ""
    for f in sorted(base.glob("*.gguf")):
        if f.name == main_name or "mmproj" in f.name.lower():
            continue
        if f.name.lower().startswith("mtp") or "-mtp-draft" in f.name.lower():
            return str(f)
    return ""


def find_matching_eagle3(alias: str, main: str | Path | None = None, draft_base_dir: str | Path | None = None) -> str:
    """ค้นหาไฟล์หัวโมเดล EAGLE-3 ที่โหลดไว้ใน DRAFT_DIR ซึ่งตรงกับตระกูลโมเดลหลักปัจจุบัน"""
    dbase = Path(draft_base_dir or DRAFT_DIR)
    if not dbase.is_dir():
        return ""
    hint = f"{alias} {Path(main).name if main else ''}".lower()
    for dk, dv in DRAFT_CATALOG.items():
        if not dv.get("is_eagle3"):
            continue
        tmatch = str(dv.get("target_match") or "").lower()
        if tmatch and tmatch.replace("-", "") not in hint.replace("-", "").replace("_", ""):
            continue
        sub = dbase / dk
        if sub.is_dir():
            hits = sorted(p for p in sub.rglob("*.gguf") if p.is_file() and "mmproj" not in p.name.lower())
            if hits:
                return str(hits[0])
    return ""


def find_matching_draft(alias: str, main: str | Path | None = None, draft_base_dir: str | Path | None = None) -> tuple[str, bool]:
    """ค้นหาไฟล์ Draft Model (Uncensored/Heretic/Abliterated มาก่อน) ใน DRAFT_DIR ที่ชุดคำศัพท์ตรงกับโมเดลหลัก"""
    dbase = Path(draft_base_dir or DRAFT_DIR)
    if not dbase.is_dir():
        return "", False
    fam = infer_vocab_family(alias, main)
    if not fam:
        return "", False
    for dk, dv in DRAFT_CATALOG.items():
        if dv.get("is_eagle3") or dv.get("vocab_family") != fam:
            continue
        sub = dbase / dk
        if sub.is_dir():
            hits = sorted(p for p in sub.rglob("*.gguf") if p.is_file() and "mmproj" not in p.name.lower())
            if hits:
                return str(hits[0]), bool(dv.get("is_mtp"))
    return "", False


def _draft_file_vocab_family(draft_path: str | Path, draft_base_dir: str | Path | None = None) -> str:
    """ดูว่าไฟล์ draft ปัจจุบันมาจากพรีเซ็ตตระกูลคำศัพท์ใดใน DRAFT_CATALOG"""
    p = Path(draft_path)
    dbase = Path(draft_base_dir or DRAFT_DIR)
    for dk, dv in DRAFT_CATALOG.items():
        try:
            p.resolve().relative_to((dbase / dk).resolve())
            return str(dv.get("vocab_family") or "")
        except (OSError, ValueError):
            if p.parent.name == dk:
                return str(dv.get("vocab_family") or "")
    return ""


def resolve_compatible_draft(
    alias: str,
    main: str | Path | None = None,
    requested_draft: str = "",
    draft_base_dir: str | Path | None = None,
) -> tuple[str, bool, str]:
    """ตรวจและเลือกไฟล์ Draft ที่เข้าคู่ชุดคำศัพท์กับโมเดลหลัก — คืน (path, is_mtp, ข้อความแจ้งเตือน)"""
    main_fam = infer_vocab_family(alias, main)
    req = (requested_draft or "").strip()
    if req and Path(req).is_file():
        if "eagle3" in Path(req).name.lower():
            return req, False, ""
        req_fam = _draft_file_vocab_family(req, draft_base_dir)
        if not main_fam or not req_fam or main_fam == req_fam:
            is_m = any(
                dv.get("is_mtp")
                for dk, dv in DRAFT_CATALOG.items()
                if Path(req).parent.name == dk
            )
            return req, is_m, ""
        alt_path, alt_mtp = find_matching_draft(alias, main, draft_base_dir)
        if alt_path:
            return (
                alt_path,
                alt_mtp,
                f"🔄 สลับโมเดล Draft จากตระกูล '{req_fam}' → '{Path(alt_path).name}' (ตระกูล {main_fam}) ให้ตรงกับ {alias} อัตโนมัติ",
            )
        sug = next(
            (dk for dk, dv in DRAFT_CATALOG.items() if not dv.get("is_eagle3") and dv.get("vocab_family") == main_fam),
            "",
        )
        hint_sug = f" (แนะนำโหลด Draft '{sug}')" if sug else ""
        return (
            "",
            False,
            f"⚠️ ข้ามไฟล์ Draft '{Path(req).name}' (ตระกูล {req_fam} คนละชุดคำศัพท์กับ {alias} ตระกูล {main_fam}){hint_sug}",
        )
    auto_path, auto_mtp = find_matching_draft(alias, main, draft_base_dir)
    if auto_path:
        return auto_path, auto_mtp, f"🎯 พบโมเดล Draft ตระกูล '{main_fam}' ที่โหลดไว้แล้ว: {Path(auto_path).name}"
    return "", False, ""


def _gguf_read_value(f, vtype: int):
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if vtype in scalars:
        fmt = scalars[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:
        ln = struct.unpack("<Q", f.read(8))[0]
        return f.read(ln).decode("utf-8", errors="ignore")
    if vtype == 9:
        etype, cnt = struct.unpack("<IQ", f.read(12))
        for _ in range(min(cnt, 100000)):
            _gguf_read_value(f, etype)
        return None
    raise ValueError(f"GGUF metadata type {vtype} ไม่รู้จัก")

_GGUF_META: dict[str, dict] = {}


def gguf_metadata(path: str | Path, wanted: tuple[str, ...] = ("tokenizer.chat_template", "general.name", "general.architecture")) -> dict:
    """อ่าน metadata ที่ต้องการจากหัวไฟล์ GGUF (คืน {} ถ้าอ่านไม่ได้)"""
    key = str(path)
    if key in _GGUF_META:
        return _GGUF_META[key]
    found: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            ver = struct.unpack("<I", f.read(4))[0]
            if ver < 2 or ver > 3:
                return {}
            struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                if klen > 1_000_000:
                    return found
                kname = f.read(klen).decode("utf-8", errors="ignore")
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _gguf_read_value(f, vtype)
                if kname in wanted:
                    found[kname] = (val[:200] if isinstance(val, str) else val)
                if len(found) == len(wanted):
                    break
    except (OSError, struct.error, ValueError):
        pass
    _GGUF_META[key] = found
    return found


def has_chat_template(path: str | Path) -> bool:
    """GGUF มี jinja chat template ฝังอยู่หรือไม่ (ตัวชี้ว่า tool calling ใช้ได้จริง)"""
    tpl = gguf_metadata(path).get("tokenizer.chat_template")
    return isinstance(tpl, str) and len(tpl.strip()) > 20


CHAT_TEMPLATE_FAMILIES = (
    ("qwen", "chatml"), ("nemotron", "chatml"), ("minicpm", "chatml"), ("lfm", "chatml"),
    ("deepseek", "deepseek"), ("gemma", "gemma"), ("muse", "gemma"), ("glimmer", "gemma"),
    ("anubis", "llama3"), ("nevoria", "llama3"), ("lumimaid-v0", "llama3"), ("llama", "llama3"),
    ("ornith", "chatml"), ("sakura", "chatml"),
)


def guess_chat_template(model_path: str | Path) -> str:
    """เดา chat template จากชื่อไฟล์/สถาปัตยกรรม (ใช้เมื่อ GGUF ไม่มี template ฝังมา)"""
    meta = gguf_metadata(model_path)
    hint = f"{Path(model_path).name} {meta.get('general.name', '')} {meta.get('general.architecture', '')}".lower()
    for needle, fam in CHAT_TEMPLATE_FAMILIES:
        if needle in hint:
            return fam
    return ""


def find_chat_template_file(model_dir: str | Path) -> str:
    """หา `chat_template.jinja` ที่มากับเรพอ (บางเรพแยกไฟล์ ไม่ฝังใน GGUF)"""
    for name in ("chat_template.jinja", "chat_template.j2"):
        p = Path(model_dir) / name
        if p.is_file() and p.stat().st_size > 20:
            return str(p)
    return ""


def tool_call_flags(model_path: str | Path, help_txt: str, template_file: str = "") -> tuple[list[str], str]:
    """flag สำหรับ chat template/tool calling ของ agent — คืน (flags, ข้อความเตือน)"""
    if "--jinja" not in help_txt:
        return [], "⚠️ ไบนารี llama.cpp นี้ไม่มี --jinja → tool calling/JSON mode ของ agent อาจใช้ไม่ได้"
    flags = ["--jinja"]
    if has_chat_template(model_path):
        return flags, ""
    if template_file and "--chat-template-file" in help_txt:
        return [*flags, "--chat-template-file", str(template_file)], ""
    fam = guess_chat_template(model_path)
    if fam and "--chat-template" in help_txt:
        return [*flags, "--chat-template", fam], (
            f"⚠️ GGUF ไม่มี chat template ฝังมา — ใส่ --chat-template {fam} ให้ (tool calling อาจต้องทดสอบซ้ำ)"
        )
    return flags, "⚠️ GGUF ไม่มี chat template ฝังมา → tool calling อาจไม่ทำงาน (--jinja จะใช้ default)"


def _gguf_layer_count(path: str | Path) -> int:
    """อ่าน <arch>.block_count จากหัวไฟล์ GGUF (คืน 0 ถ้าอ่านไม่ได้) เพื่อคำนวณ --n-gpu-layers ให้ตรงจริง"""
    key = str(path)
    if key in _GGUF_LAYERS:
        return _GGUF_LAYERS[key]
    n_layers = 0
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return 0
            ver = struct.unpack("<I", f.read(4))[0]
            if ver < 2 or ver > 3:
                return 0
            struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]
                if klen > 1_000_000:
                    return 0
                kname = f.read(klen).decode("utf-8", errors="ignore")
                vtype = struct.unpack("<I", f.read(4))[0]
                val = _gguf_read_value(f, vtype)
                if kname.endswith(".block_count") and isinstance(val, int) and val > 0:
                    n_layers = val
                    break
    except (OSError, struct.error, ValueError):
        return 0
    if n_layers:
        _GGUF_LAYERS[key] = n_layers
    return n_layers


def is_moe_gguf(path: str | Path, alias: str = "") -> bool:
    """ตรวจว่าไฟล์ GGUF เป็นสถาปัตยกรรม MoE หรือไม่ (อ่าน .expert_count ในหัวไฟล์จริง)"""
    key = str(path)
    if key in _GGUF_EXPERTS:
        return _GGUF_EXPERTS[key] > 1
    n_exp = 0
    try:
        with open(path, "rb") as f:
            raw = f.read(65536)
        n_exp = int(parse_gguf_header_bytes(raw).get("n_experts", 0))
    except OSError:
        n_exp = 0
    if n_exp > 0:
        _GGUF_EXPERTS[key] = n_exp
        return n_exp > 1
    desc = MODEL_CATALOG.get(alias, {}).get("desc", "") if alias else ""
    hint = f"{alias} {Path(path).name} {desc}".lower()
    return any(tok in hint for tok in ("moe", "8x7b", "8x22b", "-a2b", "-a3b", "-a4b", "-a9b", "mix-128g"))


def plan_moe_offload(size_gb: float, usable_gb: float, n_layers: int) -> int:
    """คำนวณจำนวนชั้น Expert ที่ต้องย้ายลง CPU RAM (--n-cpu-moe) เพื่อเก็บ Attention ทุกชั้นไว้บน GPU 100%"""
    if size_gb <= usable_gb or n_layers <= 0:
        return 0
    excess = size_gb - usable_gb
    moe_per_layer = max(0.05, (size_gb * 0.72) / n_layers)
    return max(1, min(n_layers, math.ceil(excess / moe_per_layer)))


def _is_aux_gguf(p: Path) -> bool:
    """แยกเฉพาะไฟล์เสริมจริง ๆ (mmproj, ไฟล์ imatrix ดิบ, หัว mtp-* แยก, ไฟล์ draft, eagle3)"""
    nm = p.name.lower()
    if "mmproj" in nm or "eagle3" in nm:
        return True
    if nm in ("imatrix.gguf", "imatrix.dat") or nm.startswith("imatrix-") or nm.endswith("-imatrix.gguf"):
        return True
    if nm.startswith(("mtp-", "mtp_", "draft-", "draft_")):
        return True
    if any(k in nm for k in ("-mtp-draft", "-draft-", "_draft-", ".draft.")):
        return True
    if nm.endswith(("-draft.gguf", "_draft.gguf")):
        return True
    return False


def _pick_main_gguf(model_dir: str, hw: dict | None = None, prefer: str | None = None, verbose: bool = True) -> tuple[Path, Path | None, float]:
    """เลือก GGUF หลัก: prefer (ถ้าระบุ) > quant ใหญ่สุดที่โหลดไหว > quant เล็กสุด; shard ไม่ครบ = error ชัดเจน"""
    root = Path(model_dir)
    all_ggufs = [p for p in sorted(root.rglob("*.gguf")) if "_drafts" not in p.parts]
    if not all_ggufs:
        raise FileNotFoundError(f"ไม่พบไฟล์ .gguf ใน {model_dir}")
    mmprojs = [p for p in all_ggufs if "mmproj" in p.name.lower()]
    mains = [p for p in all_ggufs if not _is_aux_gguf(p)]
    if not mains:
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดลหลักใน {model_dir}")
    s_re = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf\Z", re.IGNORECASE)
    cands: list[tuple[float, Path]] = []
    broken: list[str] = []
    seen: set[Path] = set()
    for p in mains:
        m = s_re.match(p.name)
        if m:
            pfx, idx, tot_s = m.groups()
            if idx != "00001":
                continue
            shards = [p.parent / f"{pfx}-{i:05d}-of-{tot_s}.gguf" for i in range(1, int(tot_s) + 1)]
            missing = [s.name for s in shards if not s.exists()]
            if missing:
                broken.append(f"{p.name} (ขาด {len(missing)} shard)")
                continue
            if shards[0] in seen:
                continue
            seen.add(shards[0])
            cands.append((sum(s.stat().st_size for s in shards) / (1024**3), p))
        elif "-of-" not in p.name.lower():
            if p in seen:
                continue
            seen.add(p)
            cands.append((p.stat().st_size / (1024**3), p))
    if broken and verbose:
        rprint(f"[bold yellow]⚠️ ข้ามไฟล์ที่โหลดไม่ครบ: {', '.join(broken)}[/bold yellow]")
    if not cands:
        raise FileNotFoundError(f"ไฟล์ .gguf ใน {model_dir} ไม่สมบูรณ์ (shard ขาด) — โหลดใหม่ก่อน: {', '.join(broken) or 'ไม่พบไฟล์หลัก'}")
    mm_pick = next(iter(mmprojs), None)
    if len(cands) > 1 and verbose:
        rprint("[dim]📦 quant ที่พบ: " + ", ".join(f"{p.name}={sz:.2f}GB" for sz, p in sorted(cands, key=lambda c: c[0])) + "[/dim]")
    if prefer:
        pr = prefer.lower()
        hit = next(((sz, p) for sz, p in cands if pr in p.name.lower()), None)
        if hit:
            if verbose:
                rprint(f"[cyan]🎯 ใช้ quant ตามที่ระบุ: {hit[1].name} ({hit[0]:.2f} GB)[/cyan]")
            return hit[1], mm_pick, hit[0]
        if verbose:
            rprint(f"[yellow]⚠️ ไม่พบ quant ที่ตรงกับ '{prefer}' — เลือกอัตโนมัติแทน[/yellow]")
    hw = hw or check_gpu_and_recommend(verbose=False)
    budget = max(0.5, hw["vram_free_gb"] - 2.2) if hw["has_gpu"] else max(1.0, hw["ram_avail_gb"] * 0.9)
    fitting = [c for c in cands if c[0] <= budget]
    sz, chosen = max(fitting) if fitting else min(cands)
    if verbose:
        rprint(f"[dim]   → เลือก {chosen.name} ({sz:.2f} GB / งบหน่วยความจำ {budget:.2f} GB)[/dim]")
    return chosen, mm_pick, sz

def _llama_pids() -> list:
    """pid ของ llama-server ที่ค้างอยู่ (ของโปรเจกต์นี้) — ใช้คืน VRAM ก่อนสตาร์ทใหม่"""
    try:
        out = subprocess.run(["pgrep", "-af", "llama-server"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return []
    pids: list = []
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if LLAMA_SERVER_PROC and pid == LLAMA_SERVER_PROC.pid:
            continue
        if LLAMA_CPP_DIR in cmd or "llama-server" in Path(cmd.split(" ")[0]).name:
            pids.append(pid)
    return pids


def gpu_memory_users() -> list:
    """(pid, ชื่อ process, MB) ที่กำลังกิน VRAM อยู่ — ไว้แสดงตอน VRAM ไม่พอ"""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    users = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            users.append({"pid": int(parts[0]), "name": parts[1], "mb": int(float(parts[2] or 0))})
    return users


def load_timeout_for(size_gb: float) -> int:
    """เวลารอโหลดโมเดล — โมเดลใหญ่ (หรือ offload ลง RAM) ใช้เวลามากกว่า 4 นาที"""
    return int(min(900, max(240, 240 + (float(size_gb) - 8.0) * 40)))\


def release_gpu_memory(port: int | None = None, wait_s: float = 12.0) -> dict:
    """ฆ่า llama-server ที่ค้าง (ของโปรเจกต์นี้) แล้วรอให้ VRAM ถูกคืน"""
    killed = _kill_server_on_port(int(port)) if port else []
    extra = [pid for pid in _llama_pids() if pid not in killed]
    for pid in extra:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    if extra:
        time.sleep(1.0)
        for pid in extra:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
    deadline = time.monotonic() + wait_s
    free = 0.0
    while time.monotonic() < deadline:
        hw = check_gpu_and_recommend(verbose=False)
        free = float(hw["vram_free_gb"])
        if not hw["has_gpu"] or free >= 2.0:
            break
        time.sleep(1.0)
    return {"killed": killed + extra, "vram_free_gb": free}


def resolve_draft_max_flag(help_txt: str, kind: str = "draft") -> str:
    """ชื่อ option จำกัด token ของ speculative decoding ตามเวอร์ชัน llama.cpp"""
    text = help_txt or ""
    if "--spec-type" in text:
        order = (
            ["--spec-ngram-mod-n-max", "--spec-draft-n-max"]
            if kind == "ngram"
            else ["--spec-draft-n-max", "--spec-ngram-mod-n-max"]
        )
        for flag in order:
            if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text):
                return flag
        return ""
    for flag in ("--draft-max", "--draft"):
        if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text):
            return flag
    return ""


def pick_ngram_spec_type(help_txt: str) -> str:
    """เลือกโหมด N-gram ที่ดีที่สุดที่ไบนารีรองรับ: ngram-map-k (รุ่นใหม่ เดายกบล็อก) > ngram-mod > ngram-simple"""
    text = help_txt or ""
    for cand in ("ngram-map-k", "ngram-mod", "ngram-simple", "ngram-cache"):
        if cand in text:
            return cand
    return "ngram-mod"


def _kill_server_on_port(port: int) -> list[int]:
    """ฆ่าเฉพาะ process llama-server ที่ฟังพอร์ตที่กำหนด (ไม่แตะ process อื่นในเครื่อง)"""
    killed: list[int] = []
    try:
        out = subprocess.run(["pgrep", "-af", f"llama-server.*--port {port}"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return killed
    for line in out.splitlines():
        pid_s = line.strip().split(" ", 1)[0]
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if LLAMA_SERVER_PROC and pid == LLAMA_SERVER_PROC.pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM); killed.append(pid)
        except OSError:
            pass
    if killed:
        time.sleep(1.0)
    return killed

def stop_strata_server(port: int | None = None):
    global LLAMA_SERVER_PROC
    if LLAMA_SERVER_PROC and LLAMA_SERVER_PROC.poll() is None:
        LLAMA_SERVER_PROC.terminate()
        try:
            LLAMA_SERVER_PROC.wait(timeout=8)
        except subprocess.TimeoutExpired:
            LLAMA_SERVER_PROC.kill()
    LLAMA_SERVER_PROC = None
    _kill_server_on_port(int(port or DEFAULT_PORT))
    _invalidate_server_cache(int(port or DEFAULT_PORT))
    SPEC_STATE["active_label"] = "⏹️ ปิด Speculative"
    rprint("[bold yellow]🛑 หยุดการทำงานของ API Server เรียบร้อย[/bold yellow]")

def strata_server_status(port: int = DEFAULT_PORT):
    on, mid = _check_server_online(port)
    rprint(f"[bold {'green' if on else 'red'}]{'🟢 Online' if on else '🔴 Offline'} (:{port} • {mid} • {SPEC_STATE['active_label']})[/bold {'green' if on else 'red'}]")
    if LLAMA_LOG_FILE.exists():
        print("\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]))

def start_strata_server(model_dir: str | None = MODEL_DIR, port: int = DEFAULT_PORT, ctx_size: int | None = None, restart: bool = True, spec_mode: str | None = None, draft_max: int | None = None, draft_model: str | None = None, kv_q8: bool | None = None, quant: str | None = None, extra_args: list[str] | None = None, **_) -> OpenAI:
    global LLAMA_SERVER_PROC
    if not model_dir:
        raise ValueError("กรุณาระบุ model_dir")
    srv_bin = install_llama_server()
    if restart:
        stop_strata_server(port=port)
    hw = check_gpu_and_recommend(verbose=False)
    smode = (spec_mode if spec_mode is not None else str(SPEC_STATE["mode"])).lower().strip()
    dmax_in = int(draft_max if draft_max is not None else SPEC_STATE["draft_max"])
    raw_dmodel = (draft_model if draft_model is not None else str(SPEC_STATE["draft_model"])).strip()
    use_kv_q8 = bool(kv_q8 if kv_q8 is not None else SPEC_STATE["kv_q8"])

    main_gguf, mmproj_gguf, size_gb = _pick_main_gguf(model_dir, hw=hw, prefer=quant)

    if hw["has_gpu"] and int(hw.get("gpu_count", 1)) == 1 and hw["vram_free_gb"] < min(size_gb + 1.8, 12.5):
        with contextlib.suppress(Exception):
            from .imagegen import comfyui_online, stop_comfyui

            if comfyui_online():
                rprint("[yellow]🧹 การ์ดใบเดียวเหลือ VRAM ไม่พอ — พัก ComfyUI ชั่วคราวเพื่อคืน VRAM ให้ llama-server[/yellow]")
                stop_comfyui()
                time.sleep(2.0)
                _GPU_CACHE["t"] = 0.0
                hw = check_gpu_and_recommend(verbose=False)

    if hw["has_gpu"] and hw["vram_free_gb"] < 2.0:
        rprint(f"[yellow]🧹 VRAM ว่างแค่ {hw['vram_free_gb']:.2f} GB — เก็บกวาด llama-server ที่ค้างอยู่[/yellow]")
        info = release_gpu_memory(port)
        _GPU_CACHE["t"] = 0.0
        hw = check_gpu_and_recommend(verbose=False)
        if info["killed"]:\
            rprint(f"[dim]   ปิด process ค้าง: {', '.join(map(str, info['killed']))}[/dim]")
        if hw["vram_free_gb"] < 2.0:
            holders = ", ".join(f"{u['name']}(pid {u['pid']}, {u['mb']} MB)" for u in gpu_memory_users()[:5])
            rprint(
                f"[bold red]⚠️ VRAM ยังว่างแค่ {hw['vram_free_gb']:.2f} GB"
                f"{' — ตัวที่ถืออยู่: ' + holders if holders else ''}[/bold red]\n"
                "[yellow]   จะสตาร์ทแบบ offload ลง RAM (ช้ากว่าปกติมาก) — ลอง Runtime → Restart session "
                "ถ้าต้องการคืน VRAM ทั้งหมด[/yellow]"
            )

    SPEC_STATE["last_model_dir"] = str(model_dir)
    gpu_plan = plan_multi_gpu(size_gb, hw.get("devices"))
    SPEC_STATE["gpu_plan"] = gpu_plan
    if ctx_size:
        ctx_size = int(ctx_size)
    elif hw["has_gpu"]:
        ctx_size = plan_ctx_size(hw["vram_free_gb"], size_gb, use_kv_q8)
    else:
        ctx_size = int(hw["recommended_ctx_size"])
    SPEC_STATE["ctx_size"] = ctx_size
    usable_vram = max(0.0, float(gpu_plan["usable_gb"])) if hw["has_gpu"] else 0.0
    if hw["has_gpu"]:
        rprint(f"[dim]{gpu_plan['label']}[/dim]")
        if gpu_plan["count"] > 1:
            per = " / ".join(f"{g:.1f}" for g in gpu_plan["per_gpu_gb"])
            rprint(f"[dim]   น้ำหนักโมเดล ~{size_gb:.1f} GB → แบ่ง {per} GB/การ์ด (เปิด CUDA_SCALE_LAUNCH_QUEUES=4)[/dim]")
    if hw["has_gpu"] and size_gb > usable_vram:
        rprint(
            f"[yellow]⚠️ โมเดล {size_gb:.1f} GB ใหญ่กว่า VRAM ที่ใช้ได้ ({usable_vram:.1f} GB) — "
            "จะจัดสรร offload ลง RAM อัตโนมัติ (ถ้าเป็น MoE จะเก็บ Attention บน GPU 100%)[/yellow]"
        )
    load_timeout = load_timeout_for(size_gb)
    rprint(f"[dim]🧠 ctx-size = {ctx_size} tokens (ปรับตาม VRAM อัตโนมัติ) • รอโหลดสูงสุด {load_timeout}s[/dim]")
    help_txt = subprocess.run([srv_bin, "--help"], capture_output=True, text=True).stdout
    alias_name = Path(model_dir).name

    if smode != "off":
        dmodel_in, draft_is_mtp, draft_note = resolve_compatible_draft(alias_name, main_gguf, raw_dmodel)
        if draft_note:
            rprint(f"[dim]   {draft_note}[/dim]")
    else:
        dmodel_in, draft_is_mtp = raw_dmodel, False

    has_draft = bool(dmodel_in and Path(dmodel_in).is_file())
    is_e3_draft = has_draft and ("eagle3" in Path(dmodel_in).name.lower())
    auto_e3 = find_matching_eagle3(alias_name, main_gguf)
    cat_mtp = bool(MODEL_CATALOG.get(alias_name, {}).get("has_mtp")) or ("mtp" in main_gguf.name.lower())
    mtp_draft = find_mtp_draft(model_dir, main_gguf)

    if smode == "eagle3":
        eff_mode = "eagle3" if (is_e3_draft or auto_e3) else ("draft_file" if has_draft else ("mtp" if (cat_mtp or mtp_draft) else "ngram"))
    elif smode == "auto":
        if is_e3_draft or auto_e3:
            eff_mode = "eagle3"
        elif has_draft:
            eff_mode = "draft_file_mtp" if draft_is_mtp else "draft_file"
        elif cat_mtp or mtp_draft:
            eff_mode = "mtp"
        else:
            eff_mode = "ngram"
    elif smode == "draft":
        eff_mode = "eagle3" if is_e3_draft else (("draft_file_mtp" if draft_is_mtp else "draft_file") if has_draft else ("mtp" if cat_mtp else "ngram"))
    elif smode == "mtp":
        eff_mode = "draft_file_mtp" if has_draft else "mtp"
    elif smode == "ngram":
        eff_mode = "ngram"
    else:
        eff_mode = "off"

    def _build_cmd(target_eff: str) -> tuple[list[str], str]:
        cmd = [srv_bin, "--model", str(main_gguf), "--alias", alias_name, "--host", "127.0.0.1", "--port", str(port), "--ctx-size", str(ctx_size), "--threads", str(hw["recommended_threads"])]
        moe_tag = ""
        if hw["has_gpu"]:
            multi = int(gpu_plan.get("count", 0)) > 1
            usable = max(1.0, float(gpu_plan["usable_gb"]) if multi else hw["vram_free_gb"] - 2.2)
            n_layers = _gguf_layer_count(main_gguf) or max(8, int(size_gb / 0.5))
            use_moe_cpu = (
                size_gb > usable
                and bool(SPEC_STATE.get("moe_cpu_auto", True))
                and is_moe_gguf(main_gguf, alias_name)
                and "--n-cpu-moe" in help_txt
            )
            if multi:
                if "--split-mode" in help_txt:
                    cmd.extend(["--split-mode", str(gpu_plan.get("split_mode") or "layer")])
                if "--tensor-split" in help_txt:
                    cmd.extend(["--tensor-split", ",".join(f"{r:.2f}" for r in gpu_plan["tensor_split"])])
                if "--main-gpu" in help_txt:
                    cmd.extend(["--main-gpu", str(gpu_plan["main_gpu"])])
            elif "--fit" in help_txt and not use_moe_cpu:
                cmd.extend(["--fit", "on", "--fit-target", "1024"])

            if use_moe_cpu:
                n_cpu_moe = plan_moe_offload(size_gb, usable, n_layers)
                cmd.extend(["--n-gpu-layers", str(n_layers), "--n-cpu-moe", str(n_cpu_moe)])
                if "--moe-cache-mib" in help_txt and usable >= 4.0:
                    cmd.extend(["--moe-cache-mib", "512"])
                moe_tag = f" + MoE-CPU:{n_cpu_moe}"
                rprint(f"[dim]   🧠 MoE Smart Offload: คง Attention {n_layers} ชั้นบน GPU + ย้าย Expert {n_cpu_moe} ชั้นลง RAM[/dim]")
            elif multi or "--fit" not in help_txt:
                gb_per_layer = size_gb / max(1, n_layers)
                n_gpu = n_layers if size_gb <= usable else max(1, min(n_layers, int(usable / max(gb_per_layer, 0.01))))
                cmd.extend(["--n-gpu-layers", str(n_gpu)])
                where = "ทุกการ์ด" if multi else "GPU"
                rprint(f"[dim]   🧮 offload {n_gpu}/{n_layers} layer ลง{where} (โมเดล {size_gb:.2f}GB • ใช้ได้ {usable:.2f}GB)[/dim]")
            if "--flash-attn" in help_txt:
                cmd.extend(["--flash-attn", "on"])
        else:
            cmd.extend(["--n-gpu-layers", "0"])
        if "--ctx-checkpoints" in help_txt:
            cmd.extend(["--ctx-checkpoints", "16"])
        tpl_flags, tpl_warn = tool_call_flags(main_gguf, help_txt, find_chat_template_file(model_dir))
        cmd.extend(tpl_flags)
        if tpl_warn:
            rprint(f"[yellow]{tpl_warn}[/yellow]")
        if use_kv_q8 and "--cache-type-k" in help_txt:
            cmd.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])

        no_mm = False; lbl = "⏹️ ปิด Speculative"
        e3_file = dmodel_in if is_e3_draft else auto_e3
        if target_eff == "eagle3" and e3_file and "draft-eagle3" in help_txt:
            dm = dmax_in if dmax_in > 0 else 4
            dmax_flag = resolve_draft_max_flag(help_txt, "draft")
            cmd.extend(["--model-draft", e3_file, "--spec-type", "draft-eagle3"])
            if "-ngld" in help_txt:
                cmd.extend(["-ngld", "99"])
            if dmax_flag:
                cmd.extend([dmax_flag, str(dm)])
            if "--spec-draft-p-min" in help_txt:
                cmd.extend(["--spec-draft-p-min", "0.5"])
            no_mm = True
            lbl = f"🦅 EAGLE-3 ({Path(e3_file).name[:18]}.., max={dm})"
        elif target_eff == "mtp" and not mtp_draft and "--spec-type" in help_txt:
            dm = dmax_in if dmax_in > 0 else 3
            dmax_flag = resolve_draft_max_flag(help_txt, "draft")
            cmd.extend(["--spec-type", "draft-mtp", *([dmax_flag, str(dm)] if dmax_flag else [])])
            no_mm = True; lbl = f"🚀 MTP (max={dm if dmax_flag else 'default'})"
        elif target_eff == "ngram" and "--spec-type" in help_txt:
            ng = pick_ngram_spec_type(help_txt)
            is_map = ng.startswith("ngram-map")
            dm = dmax_in if dmax_in > 0 else (24 if is_map else 16)
            dmax_flag = resolve_draft_max_flag(help_txt, "draft" if is_map else "ngram")
            cmd.extend(["--spec-type", ng, *([dmax_flag, str(dm)] if dmax_flag else [])])
            lbl = f"📚 N-gram ({ng}, max={dm if dmax_flag else 'default'})"
        elif target_eff in ("mtp", "draft_file", "draft_file_mtp") and not has_draft and mtp_draft and "--model-draft" in help_txt:
            cmd.extend(["--model-draft", mtp_draft])
            if "-ngld" in help_txt:
                cmd.extend(["-ngld", "99"])
            dmax_flag = resolve_draft_max_flag(help_txt, "draft")
            dm = dmax_in if dmax_in > 0 else 3
            if dmax_flag:
                cmd.extend([dmax_flag, str(dm)])
            if "--spec-type" in help_txt:
                cmd.extend(["--spec-type", "draft-mtp"])
            no_mm = True
            lbl = f"🚀 MTP built-in ({Path(mtp_draft).name[:20]}, max={dm})"
        elif target_eff in ("draft_file", "draft_file_mtp") and has_draft:
            dm = dmax_in if dmax_in > 0 else (3 if target_eff == "draft_file_mtp" else 4)
            dmax_flag = resolve_draft_max_flag(help_txt, "draft")
            cmd.extend(["--model-draft", dmodel_in])
            if "-ngld" in help_txt:
                cmd.extend(["-ngld", "99"])
            if dmax_flag:
                cmd.extend([dmax_flag, str(dm)])
            if target_eff == "draft_file_mtp" and "--spec-type" in help_txt:
                cmd.extend(["--spec-type", "draft-mtp"]); no_mm = True
            lbl = f"⚡ Draft ({Path(dmodel_in).name[:16]}.., max={dm})"
        if use_kv_q8:
            lbl += " + KV:Q8"
        if moe_tag:
            lbl += moe_tag
        if no_mm:
            if "--no-mmproj" in help_txt:
                cmd.append("--no-mmproj")
        elif mmproj_gguf and hw["vram_free_gb"] >= size_gb + 3.0:
            cmd.extend(["--mmproj", str(mmproj_gguf)])
        cmd.extend(str(a) for a in (extra_args or []))
        return cmd, lbl

    def _spawn(target_eff: str) -> tuple[bool, str]:
        global LLAMA_SERVER_PROC
        cmd_run, lbl_run = _build_cmd(target_eff)
        rprint(f"[bold cyan]🚀 เปิด llama-server (:{port}):[/bold cyan] [yellow]{main_gguf.name}[/yellow] ({size_gb:.2f}GB) • [magenta]{lbl_run}[/magenta]")
        with open(LLAMA_LOG_FILE, "w", encoding="utf-8") as log_f:
            LLAMA_SERVER_PROC = subprocess.Popen(
                cmd_run, stdout=log_f, stderr=subprocess.STDOUT, text=True, env=server_env()
            )
        deadline = time.monotonic() + load_timeout
        while time.monotonic() < deadline:
            if LLAMA_SERVER_PROC.poll() is not None:
                tail = "\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:])
                return False, (
                    f"process จบตัวก่อนพร้อม (exit={LLAMA_SERVER_PROC.returncode})\n{tail}{start_error_hint(tail)}"
                )
            is_ok, loaded_id = _check_server_online(port, ttl=0.0)
            if is_ok:
                SPEC_STATE["active_label"] = lbl_run
                elapsed = int(load_timeout - (deadline - time.monotonic()))
                rprint(f"[bold green]✅ Server พร้อมใน {elapsed}s! (model='{loaded_id}' • {lbl_run})[/bold green]")
                if hw["has_gpu"] and int(hw.get("gpu_count", 1)) > 1:
                    used = " · ".join(
                        f"GPU{d['index']} {d['vram_total_gb'] - d['vram_free_gb']:.1f}/{d['vram_total_gb']:.0f}GB"
                        for d in gpu_devices(ttl=0.0)
                    )
                    rprint(f"[dim]   🎮 ใช้หน่วยความจำจริง: {used} (ถ้าใบใดใบหนึ่งนิ่ง ~0 GB ให้ดู log ว่า offload ไม่ครบ)[/dim]")
                return True, loaded_id
            elapsed = int(load_timeout - (deadline - time.monotonic()))
            if elapsed and elapsed % 10 == 0:
                rprint(f"[dim]   ⏳ กำลังโหลดโมเดลลง GPU/RAM... ({elapsed}s)[/dim]")
            time.sleep(1.0)
        if LLAMA_SERVER_PROC and LLAMA_SERVER_PROC.poll() is None:
            LLAMA_SERVER_PROC.terminate()
            try:
                LLAMA_SERVER_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                LLAMA_SERVER_PROC.kill()
        tail = "\n".join(LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:])
        return False, f"โหลดโมเดลเกิน {load_timeout} วินาที\n{tail}{start_error_hint(tail)}"

    if eff_mode == "eagle3":
        extra_fb = [*([ "draft_file"] if has_draft and not is_e3_draft else []), *(["mtp"] if (cat_mtp or mtp_draft) else []), "ngram"]
    elif eff_mode == "draft_file_mtp":
        extra_fb = ["draft_file", "ngram"]
    elif eff_mode in ("mtp", "draft_file"):
        extra_fb = ["ngram"]
    else:
        extra_fb = []
    fallbacks = [eff_mode, *extra_fb]
    if eff_mode != "off":
        fallbacks.append("off")
    last_err = ""
    for idx, m_try in enumerate(dict.fromkeys(fallbacks)):
        if idx > 0:
            rprint(f"[bold yellow]⚠️ โหมดก่อนหน้าไม่พร้อม กำลังสลับเป็น '{m_try}' อัตโนมัติ...[/bold yellow]")
            _kill_server_on_port(port)
        ok, info = _spawn(m_try)
        if ok:
            return get_openai_client(port=port)  # type: ignore[return-value]
        last_err = info
    raise RuntimeError(f"llama-server สตาร์ทไม่สำเร็จทุกโหมด:\n{last_err}")

def switch_strata_model(option: str, model_dir: str = MODEL_DIR, port: int = DEFAULT_PORT) -> OpenAI:
    if option not in MODEL_CATALOG:
        raise ValueError(f"เลือกโมเดลจาก: {ALL_PRESET_KEYS}")
    if MODEL_CATALOG[option].get("kind") == "image":
        raise ValueError(
            f"'{option}' เป็นโมเดลสร้างภาพ (ไม่ใช่ text model) — llama-server รันไม่ได้ "
            "ให้ใช้แท็บ 1 ปุ่ม '🎨 สร้างภาพ' (ComfyUI) แทน"
        )
    return start_strata_server(model_dir=str(Path(model_dir) / option), port=port, restart=True)

_SERVER_LOCK = threading.RLock()


def _guarded(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not _SERVER_LOCK.acquire(blocking=False):
            raise RuntimeError(f"{name}: มีคำสั่งอื่นกำลังจัดการ server อยู่ — รอให้เสร็จแล้วลองใหม่")
        try:
            return fn(*args, **kwargs)
        finally:
            _SERVER_LOCK.release()

    return wrapper


start_strata_server = _guarded("start_strata_server", start_strata_server)
stop_strata_server = _guarded("stop_strata_server", stop_strata_server)


def get_last_model_dir() -> str | None:
    """โมเดลที่สตาร์ทล่าสุด (เก็บใน SPEC_STATE แทน global แยก)"""
    return SPEC_STATE.get("last_model_dir")
