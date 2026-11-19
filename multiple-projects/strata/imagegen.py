"""สร้างภาพด้วย ComfyUI + GGUF (รันในเครื่อง + แสดงผลในเซลล์ที่ 3 โดยตรง ไม่ใช้ Tunnel)

- ป้องกันปัญหา `AssertionError: Torch not compiled with CUDA enabled` อัตโนมัติ:
  1) ถ้ารันไทม์มี GPU แต่ PyTorch เป็นรุ่น CPU-only จะติดตั้ง PyTorch CUDA 12.8 ให้อัตโนมัติ
  2) ตอนลง requirements.txt ของ ComfyUI จะล็อกไม่ให้ pip เขียนทับ PyTorch CUDA เดิมด้วยรุ่น CPU
  3) ถ้ารันไทม์เปิดมาเป็นโหมด CPU (ไม่ได้เลือก GPU) จะสลับส่ง `--cpu` แทน `--lowvram` ไม่ให้แครช
- ยกเลิกการเปิดผ่าน Cloudflare Tunnel ทั้งหมด — แสดงหน้าจอ ComfyUI ในเซลล์ที่ 3 ผ่าน `launch_comfyui_cell()`
"""
from __future__ import annotations

import base64
import contextlib
import functools
import gzip
import io
import json
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from rich import print as rprint

from .config import (
    COMFY_DIR,
    COMFY_MODEL_SUBDIRS,
    COMFY_PORT,
    IMAGE_PRESET_KEYS,
    MODEL_CATALOG,
    MODEL_DIR,
    SPEC_STATE,
    STRATA_HOME,
)
from .platforms import detect as detect_platform
from .utils import _dir_bytes, _remember, run_stream

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"
GGUF_NODE_REPO = "https://github.com/leejet/ComfyUI-GGUF"
GGUF_NODE_DIRNAME = "ComfyUI-GGUF"
FAST_CACHE_DIRNAME = "strata_accel"
LEGACY_FRONTEND_SPEC = "Comfy-Org/ComfyUI_legacy_frontend@latest"
CHROME_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

FAST_CACHE_NODE_CODE = '''"""Strata Adaptive Residual Step-Cache (TeaCache / FBCache style) สำหรับทุกสถาปัตยกรรม DiT ใน ComfyUI."""


class StrataFastCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "threshold": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.5, "step": 0.01}),
                "warmup_steps": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "strata/speed"

    def apply(self, model, threshold: float = 0.12, warmup_steps: int = 3):
        if threshold <= 0.0:
            return (model,)
        m = model.clone()
        slots: dict = {}

        def _wrapper(apply_model, args):
            x = args["input"]
            t = args["timestep"]
            c = args["c"]
            cond_key = tuple(args.get("cond_or_uncond") or [0])
            t_val = float(t[0].detach().cpu().item()) if hasattr(t, "__getitem__") else float(t)
            st = slots.get(cond_key)
            if st is None or t_val > st["last_t"] + 1e-5:
                st = {"last_t": t_val, "step": 0, "prev_delta": None, "rel_diff": 1.0, "skipped": False}
                slots[cond_key] = st
            st["step"] += 1
            st["last_t"] = t_val
            can_skip = (
                st["step"] > int(warmup_steps)
                and t_val > 0.14
                and not st["skipped"]
                and st["prev_delta"] is not None
                and st["rel_diff"] < float(threshold)
                and st["prev_delta"].shape == x.shape
            )
            if can_skip:
                st["skipped"] = True
                return x + st["prev_delta"].to(device=x.device, dtype=x.dtype)
            out = apply_model(x, t, **c)
            delta = (out - x).detach()
            if st["prev_delta"] is not None and st["prev_delta"].shape == delta.shape:
                prev = st["prev_delta"].to(device=delta.device, dtype=delta.dtype)
                denom = float(delta.abs().mean().item()) + 1e-6
                st["rel_diff"] = float((delta - prev).abs().mean().item()) / denom
            else:
                st["rel_diff"] = 1.0
            st["prev_delta"] = delta
            st["skipped"] = False
            return out

        m.set_model_unet_function_wrapper(_wrapper)
        return (m,)\n

NODE_CLASS_MAPPINGS = {"StrataFastCache": StrataFastCache}
NODE_DISPLAY_NAME_MAPPINGS = {"StrataFastCache": "Strata FastCache (Adaptive Step Cache)"}
'''

_COMFY_PROC: subprocess.Popen | None = None

_COMFY_LOG: object | None = None


# ---------------------------------------------------------------------------
# แคตตาล็อก / ไฟล์
# ---------------------------------------------------------------------------
def image_preset(key: str) -> dict:
    cfg = MODEL_CATALOG.get(key)
    if cfg is None or cfg.get("kind") != "image":
        raise ValueError(f"'{key}' ไม่ใช่โมเดลสร้างภาพ (มี: {', '.join(IMAGE_PRESET_KEYS)})")
    return cfg


def include_patterns(key: str) -> list[str]:
    return list(image_preset(key)["targets"])


def file_sources(key: str) -> dict[str, str]:
    cfg = image_preset(key)
    extra = cfg.get("sources") or {}
    return {rel: str(extra.get(rel) or cfg["repo_id"]) for rel in cfg["targets"]}


def _comfy_models_root(comfy_dir: str | Path = COMFY_DIR) -> Path:
    return Path(comfy_dir) / "models"


def comfy_target_path(key: str, rel: str, comfy_dir: str | Path = COMFY_DIR) -> Path:
    sub = image_preset(key)["targets"][rel]
    if sub not in COMFY_MODEL_SUBDIRS:
        raise ValueError(f"โฟลเดอร์ปลายทาง '{sub}' ไม่รองรับ (ต้องเป็น {COMFY_MODEL_SUBDIRS})")
    return _comfy_models_root(comfy_dir) / sub / Path(rel).name


def image_model_status(key: str, model_dir: str | Path | None = None) -> dict:
    cfg = image_preset(key)
    base = Path(model_dir or MODEL_DIR) / key
    missing = [rel for rel in cfg["targets"] if not (base / rel).is_file()]
    gb = _dir_bytes(base) / (1024**3) if base.is_dir() else 0.0
    return {
        "key": key,
        "dir": str(base),
        "downloaded": not missing,
        "missing": missing,
        "gb": round(gb, 2),
        "approx_gb": cfg["approx_gb"],
    }


# ---------------------------------------------------------------------------
# ติดตั้งโมเดลภาพ + ผูกเข้า ComfyUI + จับคู่ชื่อไฟล์จริงบน Hugging Face อัตโนมัติ
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _list_hf_repo_files(repo_id: str) -> tuple[str, ...]:
    with contextlib.suppress(Exception):
        from huggingface_hub import HfApi

        return tuple(HfApi().list_repo_files(repo_id=repo_id, repo_type="model"))
    return ()


def resolve_remote_image_file(wanted_rel: str, repo_files: tuple[str, ...] | list[str], subdir: str = "") -> str:
    if not repo_files or wanted_rel in repo_files:
        return wanted_rel
    w_name = Path(wanted_rel).name.lower()
    w_ext = Path(wanted_rel).suffix.lower()
    for rf in repo_files:
        if Path(rf).name.lower() == w_name:
            return rf
    norm_w = re.sub(r"[^a-z0-9]", "", w_name)
    for rf in repo_files:
        if re.sub(r"[^a-z0-9]", "", Path(rf).name.lower()) == norm_w:
            return rf
    cands = [rf for rf in repo_files if rf.lower().endswith(w_ext) and "mmproj" not in rf.lower()]
    if not cands:
        return wanted_rel
    q_match = re.search(r"(iq\d_[a-z0-9]+|q\d_[a-z0-9_]+|q8_0|fp8_e4m3fn_scaled|fp8_e4m3fn|int8_convrot|bf16|fp16)", w_name)
    if q_match:
        qtag = q_match.group(1)
        q_hits = [rf for rf in cands if qtag in rf.lower()]
        if subdir == "vae":
            q_hits = [rf for rf in q_hits if "vae" in rf.lower() or "/vae/" in f"/{rf.lower()}"] or q_hits
        elif subdir == "diffusion_models":
            q_hits = [rf for rf in q_hits if "vae" not in rf.lower() and "text_encoder" not in rf.lower()] or q_hits
        if q_hits:
            return sorted(q_hits, key=len)[0]
    if subdir == "vae":
        v_hits = [rf for rf in cands if "vae" in rf.lower() or Path(rf).name.lower() == "ae.safetensors"]
        if v_hits:
            return sorted(v_hits, key=len)[0]
    return wanted_rel


def _clean_broken_model_symlinks(comfy_dir: str | Path = COMFY_DIR) -> int:
    mroot = _comfy_models_root(comfy_dir)
    if not mroot.is_dir():
        return 0
    removed = 0
    for sub in mroot.iterdir():
        if not sub.is_dir():
            continue
        for item in list(sub.iterdir()):
            if item.is_symlink() and not item.exists():
                with contextlib.suppress(OSError):
                    item.unlink()
                    removed += 1
    return removed


def _find_existing_companion(fname: str, subdir: str, base: Path, comfy_dir: str | Path = COMFY_DIR) -> Path | None:
    comfy_cand = _comfy_models_root(comfy_dir) / subdir / fname
    if comfy_cand.is_file() and comfy_cand.resolve().stat().st_size > 1024 * 1024:
        return comfy_cand.resolve()
    parent = base.parent
    if not parent.is_dir():
        return None
    for sib in parent.iterdir():
        if sib == base or not sib.is_dir() or sib.name.startswith("_"):
            continue
        for hit in sib.rglob(fname):
            if hit.is_file() and hit.resolve().stat().st_size > 1024 * 1024:
                return hit.resolve()
    return None


def _reuse_shared_companions(key: str, base: Path, comfy_dir: str | Path = COMFY_DIR) -> None:
    cfg = image_preset(key)
    for rel, subdir in cfg["targets"].items():
        if subdir == "diffusion_models":
            continue
        dst = base / rel
        if dst.is_file():
            continue
        found = _find_existing_companion(Path(rel).name, subdir, base, comfy_dir=comfy_dir)
        if found is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            os.symlink(found, dst)
            rprint(f"[dim]♻️ ใช้ไฟล์ร่วมที่มีอยู่แล้ว: {Path(rel).name}[/dim]")


def link_image_model(key: str, model_dir: str | Path | None = None, comfy_dir: str | Path = COMFY_DIR) -> dict[str, str]:
    _clean_broken_model_symlinks(comfy_dir)
    cfg = image_preset(key)
    base = (Path(model_dir or MODEL_DIR) / key)
    linked: dict[str, str] = {}
    for rel in cfg["targets"]:
        src = (base / rel).resolve()
        if not src.is_file():
            continue
        dst = comfy_target_path(key, rel, comfy_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                linked[rel] = str(dst)
                continue
            dst.unlink()
        os.symlink(src, dst)
        linked[rel] = str(dst)
    return linked


def link_all_downloaded_image_models(model_dir: str | Path | None = None, comfy_dir: str | Path = COMFY_DIR) -> list[str]:
    """ผูกโมเดลภาพทุกตัวที่ดาวน์โหลดครบแล้วเข้า ComfyUI/models/* พร้อมกัน"""
    ready_keys: list[str] = []
    for k in IMAGE_PRESET_KEYS:
        if image_model_status(k, model_dir)["downloaded"]:
            link_image_model(k, model_dir=model_dir, comfy_dir=comfy_dir)
            ready_keys.append(k)
    return ready_keys


def install_image_model(key: str, model_dir: str | Path | None = None, force: bool = False, **_: object) -> dict:
    from .download import download_model_files

    cfg = image_preset(key)
    base = Path(model_dir or MODEL_DIR) / key
    base.mkdir(parents=True, exist_ok=True)
    if not force:
        _reuse_shared_companions(key, base)
    by_repo: dict[str, list[str]] = {}
    alias_pairs: list[tuple[str, str]] = []
    for rel, repo in file_sources(key).items():
        if not force and (base / rel).is_file():
            continue
        rfiles = _list_hf_repo_files(repo)
        actual_rel = resolve_remote_image_file(rel, rfiles, cfg["targets"][rel]) if rfiles else rel
        if actual_rel != rel:
            rprint(f"[dim]🔎 จับคู่ชื่อไฟล์จริงในเรพอ: '{rel}' → '{actual_rel}'[/dim]")
            alias_pairs.append((actual_rel, rel))
        by_repo.setdefault(repo, []).append(actual_rel)
    total_dl, total_gb = 0, 0.0
    if not by_repo:
        rprint(f"[dim]⏭️ ไฟล์ของ {key} ครบทุกชิ้นแล้ว[/dim]")
    for repo, rels in by_repo.items():
        rprint(f"[bold magenta]🎨 โมเดลสร้างภาพ:[/bold magenta] [cyan]{key}[/cyan] -> {repo} ({len(rels)} ไฟล์)")
        part = download_model_files(
            include=rels, exclude=None, model_dir=str(base), repo_id=repo, force=force
        )
        total_dl += int(part.get("downloaded", 0))
        total_gb += float(part.get("required_gb", 0.0))
    for remote_rel, wanted_rel in alias_pairs:
        src_p = (base / remote_rel).resolve()
        dst_p = base / wanted_rel
        if src_p.is_file() and not dst_p.is_file():
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            if dst_p.is_symlink():
                dst_p.unlink()
            os.symlink(src_p, dst_p)
    linked = link_image_model(key, base.parent)
    st = image_model_status(key, base.parent)
    if st["missing"]:
        raise FileNotFoundError(f"ยังขาดไฟล์: {', '.join(st['missing'])} (ลองดาวน์โหลดใหม่หรือตรวจชื่อไฟล์ในเรพอ)")
    rprint(f"[bold green]✅ พร้อมสร้างภาพ[/bold green] ({st['gb']:.2f} GB • ผูกเข้า ComfyUI {len(linked)} ไฟล์)")
    return {"downloaded": total_dl, "required_gb": round(total_gb, 2), **st, "linked": linked}


# ---------------------------------------------------------------------------
# PyTorch CUDA Verification & Protection (แก้ AssertionError: Torch not compiled with CUDA enabled)
# ---------------------------------------------------------------------------
def _probe_torch_cuda() -> dict:
    """ตรวจสถานะ PyTorch ของ sys.executable ใน env สะอาด (ไม่ปน /nix/store)"""
    code = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    print(json.dumps({'installed': True, 'ver': torch.__version__, "
        "'cuda_ver': torch.version.cuda, 'cuda_avail': bool(torch.cuda.is_available())}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'installed': False, 'err': str(e)}))\n"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=25,
            env=comfy_runtime_env(),
        ).stdout.strip()
        last = out.splitlines()[-1] if out else "{}"
        return json.loads(last)
    except Exception as e:  # noqa: BLE001
        return {"installed": False, "err": str(e)}


def ensure_torch_runtime() -> dict:
    """ตรวจสอบและซ่อมแซม PyTorch ให้ตรงกับฮาร์ดแวร์:
    - ถ้ามี NVIDIA GPU แต่ PyTorch เป็นรุ่น CPU-only (`cuda_ver=None`): ติดตั้ง PyTorch CUDA 12.8 ให้อัตโนมัติ!
    - ถ้าไม่มี GPU (เปิด Colab เป็น CPU Runtime): แจ้งเตือนชัดเจนและให้ ComfyUI รันด้วย `--cpu` แทนการแครช
    """
    has_hw_gpu = False
    with contextlib.suppress(Exception):
        from .server import gpu_devices

        has_hw_gpu = bool(gpu_devices(ttl=0.0))

    st = _probe_torch_cuda()
    if has_hw_gpu and (not st.get("installed") or not st.get("cuda_ver")):
        rprint(
            "[bold yellow]🔧 ตรวจพบการ์ดจอ NVIDIA GPU แต่ PyTorch ปัจจุบันเป็นรุ่น CPU-only "
            f"({st.get('ver', 'ยังไม่ติดตั้ง')}) — กำลังติดตั้ง PyTorch CUDA 12.8 ให้อัตโนมัติ...[/bold yellow]"
        )
        installed_ok = False
        for idx_url in (
            "https://download.pytorch.org/whl/cu128",
            "https://download.pytorch.org/whl/cu126",
            "https://download.pytorch.org/whl/cu124",
        ):
            try:
                _pip_install_comfy(
                    "--upgrade",
                    "--force-reinstall",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    idx_url,
                )
                st = _probe_torch_cuda()
                if st.get("cuda_ver"):
                    installed_ok = True
                    rprint(f"[bold green]✅ ติดตั้ง PyTorch CUDA สำเร็จ ({st.get('ver')} • CUDA {st.get('cuda_ver')})[/bold green]")
                    break
            except Exception:
                continue
        if not installed_ok:
            rprint("[yellow]⚠️ ติดตั้ง PyTorch CUDA ไม่สำเร็จ — จะรัน ComfyUI ในโหมด --cpu ชั่วคราว[/yellow]")
    elif not has_hw_gpu:
        rprint(
            "[bold yellow]⚠️ รันไทม์ปัจจุบันไม่ได้เปิดใช้งาน GPU (เป็นโหมด CPU Runtime) — "
            "ComfyUI จะรันในโหมด `--cpu` เพื่อไม่ให้เกิดข้อผิดพลาด CUDA "
            "(แนะนำไปที่เมนู Runtime → Change runtime type → เลือก T4 GPU เพื่อวาดภาพเร็วขึ้น ~20 เท่า)[/bold yellow]"
        )
    return {"has_hw_gpu": has_hw_gpu, **st}


def _can_use_cuda_for_comfy() -> bool:
    """คืน True เฉพาะเมื่อมี GPU จริงและ PyTorch คอมไพล์มาพร้อม CUDA พร้อมใช้งาน"""
    with contextlib.suppress(Exception):
        from .server import gpu_devices

        if not gpu_devices():
            return False
    st = _probe_torch_cuda()
    return bool(st.get("installed") and st.get("cuda_ver") and st.get("cuda_avail"))


# ---------------------------------------------------------------------------
# ComfyUI runtime
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _comfy_help_flags(comfy_dir_s: str) -> str:
    main_py = Path(comfy_dir_s) / "main.py"
    if not main_py.is_file():
        return ""
    with contextlib.suppress(Exception):
        p = subprocess.run(
            [sys.executable, str(main_py), "--help", "--cpu"],
            cwd=comfy_dir_s,
            capture_output=True,
            text=True,
            timeout=25,
            env=comfy_runtime_env(),
        )
        return (p.stdout or "") + "\n" + (p.stderr or "")
    return ""


def comfyui_command(
    port: int = COMFY_PORT,
    lowvram: bool = True,
    comfy_dir: str | Path = COMFY_DIR,
    ui_mode: str | None = None,
    use_cuda: bool | None = None,
) -> list[str]:
    """สร้างคำสั่งสตาร์ท ComfyUI — ถ้าไม่มี CUDA จะสลับใส่ `--cpu` อัตโนมัติ ป้องกัน AssertionError"""
    mode = (ui_mode or str(SPEC_STATE.get("comfy_ui_mode") or "modern")).lower().strip()
    help_txt = _comfy_help_flags(str(Path(comfy_dir)))
    cmd = [
        sys.executable,
        "main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        str(port),
        "--enable-cors-header",
        "*",
        "--preview-method",
        "none",
    ]
    if not help_txt or "--enable-compress-response-body" in help_txt:
        cmd.append("--enable-compress-response-body")
    if mode == "classic" and (not help_txt or "--front-end-version" in help_txt):
        cmd.extend(["--front-end-version", LEGACY_FRONTEND_SPEC])

    # ถ้าระบุ use_cuda=None: ในเทสต์ที่ไม่มี main.py ให้คงพฤติกรรม lowvram เดิม; ถ้ารันจริงให้เช็ก CUDA จริง!
    if use_cuda is None:
        cuda_ok = True if not (Path(comfy_dir) / "main.py").is_file() else _can_use_cuda_for_comfy()
    else:
        cuda_ok = bool(use_cuda)

    if not cuda_ok:
        cmd.append("--cpu")
    elif lowvram:
        cmd.append("--lowvram")
        for opt in ("--cache-none", "--disable-smart-memory", "--fp16-vae"):
            if not help_txt or opt in help_txt:
                cmd.append(opt)
    return cmd


TARBALL_BRANCHES = ("master", "main")


def _say(text: str) -> None:
    _remember(text)
    print(text)


def _run_git(args: list[str], cwd: Path | None = None, timeout: int = 900) -> None:
    cmd = ["git", *args]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"รัน `{' '.join(cmd)}` ไม่ได้: {e}") from e
    for line in ((p.stdout or "") + (p.stderr or "")).splitlines()[-15:]:
        _say(line)
    if p.returncode != 0:
        raw = (p.stderr or p.stdout or "").strip().splitlines()
        tail = " | ".join(x.strip() for x in raw[-3:] if x.strip()) or "(git ไม่ได้บอกสาเหตุ)"
        raise RuntimeError(f"`{' '.join(cmd)}` ล้มเหลว (exit {p.returncode}): {tail}")


def _git(args: list[str], cwd: Path | None = None) -> None:
    _run_git(args, cwd=cwd)


def _repo_slug(repo_url: str) -> str:
    s = repo_url.rstrip("/").removesuffix(".git")
    return s.split("github.com/")[-1] if "github.com/" in s else s


def _extract_tarball(data: bytes, dest: Path) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            m.name = parts[1]
            members.append(m)
        try:
            tar.extractall(dest, members=members, filter="data")
        except TypeError:
            tar.extractall(dest, members=members)


def _fetch_tarball(repo_url: str, dest: Path) -> None:
    slug = _repo_slug(repo_url)
    last = ""
    for br in TARBALL_BRANCHES:
        url = f"https://codeload.github.com/{slug}/tar.gz/refs/heads/{br}"
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            last = f"{br}: {e}"
            continue
        _say(f"⬇️ ดาวน์โหลด tarball {slug}@{br} (~{len(data) / 1024**2:.0f} MB)")
        _extract_tarball(data, dest)
        return
    raise RuntimeError(f"ดาวน์โหลด {slug} เป็น tarball ไม่สำเร็จ ({last})")


def _clear_incomplete(dest: Path) -> None:
    dest = Path(dest)
    if not dest.exists():
        return
    try:
        empty = not any(dest.iterdir())
    except OSError:
        empty = False
    if empty:
        dest.rmdir()
        return
    rprint(f"[yellow]🧹 ล้างโฟลเดอร์เดิมที่ไม่สมบูรณ์: {dest}[/yellow]")
    shutil.rmtree(dest, ignore_errors=True)


def _clone_or_tarball(repo_url: str, dest: Path, marker: str = "main.py") -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stash_root: Path | None = None
    if dest.is_dir() and not (dest / marker).exists() and (dest / "models").is_dir():
        stash_root = Path(tempfile.mkdtemp(prefix="strata_comfy_models_"))
        shutil.move(str(dest / "models"), str(stash_root / "models"))
        shutil.rmtree(dest, ignore_errors=True)
    try:
        try:
            _run_git(["clone", "--depth", "1", repo_url, str(dest)])
            if (dest / marker).exists():
                return
            rprint(f"[yellow]⚠️ clone เสร็จแต่ไม่พบ {marker} — ลองดาวน์โหลด tarball แทน[/yellow]")
        except RuntimeError as e:
            rprint(f"[yellow]⚠️ git clone ไม่สำเร็จ: {e}[/yellow]")
            rprint("[yellow]→ ลองดาวน์โหลดเป็น tarball จาก codeload แทน[/yellow]")
        _clear_incomplete(dest)
        _fetch_tarball(repo_url, dest)
        if not (dest / marker).exists():
            raise RuntimeError(f"ติดตั้ง {_repo_slug(repo_url)} แล้วแต่ไม่พบ {marker} ใน {dest}")
    finally:
        if stash_root and (stash_root / "models").is_dir():
            if dest.is_dir():
                for sub in (stash_root / "models").iterdir():
                    tgt_sub = dest / "models" / sub.name
                    tgt_sub.mkdir(parents=True, exist_ok=True)
                    for item in sub.iterdir():
                        dst_item = tgt_sub / item.name
                        if not dst_item.exists() and not dst_item.is_symlink():
                            shutil.move(str(item), str(dst_item))
            shutil.rmtree(stash_root, ignore_errors=True)


def _is_legacy_gguf_node(node_dir: Path) -> bool:
    git_cfg = node_dir / ".git" / "config"
    if git_cfg.is_file():
        txt = git_cfg.read_text(encoding="utf-8", errors="ignore")
        if "city96/ComfyUI-GGUF" in txt:
            return True
    loader = node_dir / "loader.py"
    if loader.is_file():
        ltxt = loader.read_text(encoding="utf-8", errors="ignore").lower()
        if "qwen" not in ltxt:
            return True
    return False


def _pip_install_comfy(*pip_args: str) -> None:
    py = shlex.quote(sys.executable)
    joined = " ".join(shlex.quote(a) for a in pip_args)
    try:
        run_stream(f"{py} -m pip install -q {joined}")
    except subprocess.CalledProcessError:
        run_stream(f"{py} -m pip install -q --break-system-packages {joined}")


def _install_req_preserving_cuda_torch(req_file: Path) -> None:
    """ติดตั้ง requirements.txt โดยกรอง torch/torchvision/torchaudio ออกถ้าเครื่องมี PyTorch CUDA อยู่แล้ว!
    (ป้องกัน pip ไปดึง torch รุ่น CPU-only จาก PyPI มาทับของเดิมจนเกิด Torch not compiled with CUDA enabled)
    """
    st = _probe_torch_cuda()
    if st.get("installed") and st.get("cuda_ver"):
        lines = req_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        kept = [
            ln for ln in lines
            if not re.match(r"^\s*(torch|torchvision|torchaudio)\b", ln, flags=re.IGNORECASE)
        ]
        filtered = req_file.parent / ".strata_requirements_notorch.txt"
        filtered.write_text("\n".join(kept) + "\n", encoding="utf-8")
        try:
            _pip_install_comfy("-r", str(filtered))
        finally:
            filtered.unlink(missing_ok=True)
    else:
        _pip_install_comfy("-r", str(req_file))


def ensure_fast_cache_node(comfy_dir: str | Path = COMFY_DIR) -> tuple[Path, bool]:
    init_py = Path(comfy_dir) / "custom_nodes" / FAST_CACHE_DIRNAME / "__init__.py"
    init_py.parent.mkdir(parents=True, exist_ok=True)
    if init_py.is_file() and init_py.read_text(encoding="utf-8", errors="ignore") == FAST_CACHE_NODE_CODE:
        return init_py, False
    init_py.write_text(FAST_CACHE_NODE_CODE, encoding="utf-8")
    return init_py, True


def ensure_comfyui(pull: bool = True, comfy_dir: str | Path = COMFY_DIR) -> Path:
    root = Path(comfy_dir)
    if not (root / "main.py").is_file():
        _clone_or_tarball(COMFY_REPO, root, marker="main.py")
    elif pull:
        try:
            _run_git(["pull", "--ff-only"], cwd=root)
        except RuntimeError as e:
            rprint(f"[yellow]⚠️ อัปเดต ComfyUI ไม่ได้ (ใช้ของเดิมต่อ): {e}[/yellow]")
    node = root / "custom_nodes" / GGUF_NODE_DIRNAME
    if (node / "nodes.py").is_file() and _is_legacy_gguf_node(node):
        rprint("[yellow]🔄 พบ ComfyUI-GGUF รุ่นเก่า (city96) — กำลังสลับเป็น leejet/ComfyUI-GGUF เพื่อรองรับ Qwen-Image-2.1[/yellow]")
        shutil.rmtree(node, ignore_errors=True)
    if not (node / "nodes.py").is_file():
        _clear_incomplete(node)
        _clone_or_tarball(GGUF_NODE_REPO, node, marker="nodes.py")
    ensure_fast_cache_node(root)
    _clean_broken_model_symlinks(root)
    ensure_torch_runtime()
    req = root / "requirements.txt"
    node_req = node / "requirements.txt"
    try:
        if req.is_file():
            _install_req_preserving_cuda_torch(req)
        if node_req.is_file():
            _install_req_preserving_cuda_torch(node_req)
        _pip_install_comfy("--upgrade", "gguf", "comfyui-frontend-package")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ติดตั้ง dependency ของ ComfyUI ไม่สำเร็จ (exit {e.returncode}) — "
            "ลองรัน `%s -m pip install -r %s` เองแล้วดู error แรก" % (sys.executable, req)
        ) from e
    # ตรวจซ้ำอีกครั้งหลังลง requirements เผื่อแพ็กเกจย่อยไปแตะ torch
    ensure_torch_runtime()
    _comfy_help_flags.cache_clear()
    rprint(f"[bold green]✅ ComfyUI พร้อม[/bold green] ที่ {root}")
    return root


CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"

_TUNNEL_PROC: subprocess.Popen | None = None

_TUNNEL_LOG_PATH: Path | None = None


def comfyui_log_path(comfy_dir: str | Path = COMFY_DIR) -> Path:
    return Path(comfy_dir) / "comfyui.log"


def parse_comfy_port(log_text: str, fallback: int = COMFY_PORT) -> int:
    hits = re.findall(r"To see the GUI go to: http://[^:]+:(\d+)", log_text)
    if not hits:
        hits = re.findall(r"http://(?:0\.0\.0\.0|127\.0\.0\.1|\[::\]):(\d+)", log_text)
    return int(hits[-1]) if hits else fallback


def parse_cloudflared_url(text: str) -> str:
    hits = re.findall(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com", text)
    return hits[-1] if hits else ""


def _decode_http_bytes(raw: bytes, content_encoding: str = "") -> str:
    if raw[:2] == b"\x1f\x8b" or "gzip" in (content_encoding or "").lower():
        with contextlib.suppress(Exception):
            raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="ignore")


def _local_root_serves_html(port: int = COMFY_PORT, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/", headers={"User-Agent": CHROME_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not (200 <= int(r.status) < 400):
                return False
            txt = _decode_http_bytes(r.read(8192), r.headers.get("Content-Encoding", "")).lower()
            return "<html" in txt or "<!doctype" in txt
    except Exception:  # noqa: BLE001
        return False


def comfy_object_info_ok(port: int = COMFY_PORT, timeout: float = 15.0) -> tuple[bool, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{int(port)}/object_info/KSampler", headers={"User-Agent": CHROME_UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (200 <= int(r.status) < 400), ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def public_url_works(url: str, timeout: float = 12.0) -> bool:
    if not url:
        return False
    headers = {"User-Agent": CHROME_UA}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            if not (200 <= int(r.status) < 400):
                return False
            txt = _decode_http_bytes(r.read(8192), r.headers.get("Content-Encoding", "")).lower()
            return "<html" in txt or "<!doctype" in txt or len(txt) > 20
    except Exception:  # noqa: BLE001
        return False


def comfy_allows_cross_site(port: int = COMFY_PORT) -> bool:
    req = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/system_stats",
        headers={"User-Agent": CHROME_UA, "Sec-Fetch-Site": "cross-site", "Origin": "https://colab.research.google.com"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2.5) as r:
            return 200 <= int(r.status) < 400
    except Exception:  # noqa: BLE001
        return False


def _kill_matching_pids(pattern: str) -> list[int]:
    killed: list[int] = []
    try:
        out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return killed
    my_pid = os.getpid()
    for line in out.splitlines():
        pid_s = line.strip().split(" ", 1)[0]
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        if pid == my_pid:
            continue
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
    if killed:
        time.sleep(1.5)
        for pid in killed:
            with contextlib.suppress(OSError):
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
    return killed


def ensure_cloudflared(bin_dir: str | Path | None = None) -> Path:
    target = Path(bin_dir or Path(MODEL_DIR).parent / "bin") / "cloudflared"
    if target.is_file() and os.access(target, os.X_OK):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(CLOUDFLARED_URL, timeout=180) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.chmod(0o755)
    tmp.replace(target)
    return target


def start_tunnel(port: int = COMFY_PORT, bin_dir: str | Path | None = None, wait_s: int = 60, force_new: bool = False) -> str:
    """เก็บไว้เพื่อความเข้ากันได้ย้อนหลังเท่านั้น (ระบบหลักเปลี่ยนมาแสดงในเซลล์ที่ 3 โดยไม่ใช้ Tunnel แล้ว)"""
    return f"http://127.0.0.1:{int(port)}"


def stop_tunnel() -> bool:
    global _TUNNEL_PROC
    had = False
    if _TUNNEL_PROC is not None and _TUNNEL_PROC.poll() is None:
        _TUNNEL_PROC.terminate()
        with contextlib.suppress(Exception):
            _TUNNEL_PROC.wait(timeout=5)
        had = True
    _TUNNEL_PROC = None
    extra = _kill_matching_pids("cloudflared.*tunnel")
    return bool(had or extra)


def open_colab_port_iframe(port: int = COMFY_PORT, height: int = 820) -> bool:
    """แสดงหน้าจอ ComfyUI แบบเต็มความกว้างภายในเซลล์ที่ 3 โดยตรง (ไม่ต้องเปิดผ่าน Tunnel)"""
    try:
        from google.colab import output  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        output.serve_kernel_port_as_iframe(int(port), height=str(height))
        return True
    except Exception:
        return False


def open_colab_port_window(port: int = COMFY_PORT) -> bool:
    return open_colab_port_iframe(port=port)


def ensure_public_url(port: int, prefer: str = "local") -> tuple[str, str]:
    """คืน URL ภายในเครื่อง (ยกเลิกการใช้ Tunnel เป็นค่าเริ่มต้น)"""
    if prefer == "colab":
        info = detect_platform()
        if not info["proxy_ports"]:
            raise RuntimeError(f"{info['label']} ไม่มี port proxy")
        return comfyui_public_url(port), "colab"
    return f"http://127.0.0.1:{int(port)}", "local"


def comfyui_online(port: int = COMFY_PORT) -> bool:
    import urllib.error

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=3) as r:
            return r.status < 500
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def comfyui_public_url(port: int = COMFY_PORT) -> str:
    return f"http://127.0.0.1:{int(port)}"


def comfy_gpu_env() -> dict:
    env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    try:
        from .server import gpu_devices
    except Exception:  # noqa: BLE001
        return env
    devices = gpu_devices()
    if len(devices) < 2:
        return env
    best = max(devices, key=lambda d: float(d.get("vram_free_gb", 0.0)))
    env["CUDA_VISIBLE_DEVICES"] = str(best["index"])
    rprint(
        f"[dim]🎨 ปัก ComfyUI ไปที่ GPU{best['index']} ({best['name']} • ว่าง {best['vram_free_gb']:.1f} GB) "
        "เพื่อไม่ให้แย่ง VRAM กับ llama-server[/dim]"
    )
    return env


def comfy_runtime_env() -> dict:
    """ตัด /nix/store ออกจากทั้ง LD_LIBRARY_PATH และ PYTHONPATH เพื่อไม่ให้ไลบรารี Nix ตีกับ PyTorch CUDA"""
    env = dict(os.environ)
    with contextlib.suppress(Exception):
        from .server import sanitize_ld_library_path

        env["LD_LIBRARY_PATH"] = sanitize_ld_library_path(env.get("LD_LIBRARY_PATH"))
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = sanitize_ld_library_path(env.get("PYTHONPATH"))
    if "/nix/store/" in (env.get("LD_PRELOAD") or ""):
        env.pop("LD_PRELOAD", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.update(comfy_gpu_env())
    return env


def _free_single_gpu_if_needed() -> None:
    with contextlib.suppress(Exception):
        from .server import _llama_pids, gpu_devices, release_gpu_memory, stop_strata_server

        devs = gpu_devices(ttl=0.0)
        if len(devs) == 1 and float(devs[0].get("vram_free_gb", 0.0)) < 10.0 and _llama_pids():
            rprint("[yellow]🧹 การ์ดใบเดียวต้องใช้ VRAM โหลดโมเดลภาพ — พัก llama-server ชั่วคราวให้อัตโนมัติ[/yellow]")
            stop_strata_server()
            release_gpu_memory()


def _ensure_cors_ready_comfy(port: int = COMFY_PORT, ui_mode: str | None = None) -> None:
    """ล้าง cloudflared tunnel เดิมที่ค้างอยู่ และตรวจว่า ComfyUI ในเครื่องพร้อมใช้งาน"""
    stop_tunnel()
    if not comfyui_online(port):
        return
    if _local_root_serves_html(port, timeout=3.0) and comfy_allows_cross_site(port):
        return
    stop_comfyui(port=port)


def _spawn_comfy_process(root: Path, port: int, lowvram: bool, wait_s: int, ui_mode: str, use_cuda: bool | None = None) -> None:
    global _COMFY_PROC, _COMFY_LOG
    _clean_broken_model_symlinks(root)
    ensure_fast_cache_node(root)
    if use_cuda is not False:
        _free_single_gpu_if_needed()
    log = comfyui_log_path(root)
    _close_comfy_log()
    _COMFY_LOG = log.open("w", encoding="utf-8")
    _COMFY_PROC = subprocess.Popen(
        comfyui_command(port, lowvram=lowvram, comfy_dir=root, ui_mode=ui_mode, use_cuda=use_cuda),
        cwd=str(root),
        stdout=_COMFY_LOG,
        stderr=subprocess.STDOUT,
        env=comfy_runtime_env(),
    )
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if comfyui_online(port):
            return
        if _COMFY_PROC.poll() is not None:
            tail = log.read_text(encoding="utf-8", errors="ignore").splitlines()[-18:] if log.is_file() else []
            raise RuntimeError("ComfyUI ออกก่อนพร้อมใช้งาน:\n" + "\n".join(tail))
        time.sleep(1.5)
    raise TimeoutError(f"ComfyUI ไม่ขึ้นภายใน {wait_s}s — ดู log ที่ {log}")


def start_comfyui(
    port: int = COMFY_PORT,
    lowvram: bool = True,
    wait_s: int = 180,
    comfy_dir: str | Path = COMFY_DIR,
    public: str = "local",
    force_restart: bool = False,
    ui_mode: str | None = None,
) -> str:
    """สตาร์ทเซิร์ฟเวอร์ ComfyUI ในเครื่อง (ไม่ใช้ Tunnel — แสดงผลผ่าน Iframe ในเซลล์ที่ 3)"""
    stop_tunnel()
    root = Path(comfy_dir)
    mode = (ui_mode or str(SPEC_STATE.get("comfy_ui_mode") or "modern")).lower().strip()
    SPEC_STATE["comfy_ui_mode"] = mode
    if force_restart:
        stop_comfyui(port=port)
    else:
        _ensure_cors_ready_comfy(port, ui_mode=mode)

    if not comfyui_online(port):
        if not (root / "main.py").is_file():
            raise FileNotFoundError(f"ยังไม่มี ComfyUI ที่ {root} — เรียก ensure_comfyui() ก่อน")
        ensure_torch_runtime()
        cuda_ok = _can_use_cuda_for_comfy()
        try:
            _spawn_comfy_process(root, port, lowvram=lowvram, wait_s=wait_s, ui_mode=mode, use_cuda=cuda_ok)
        except RuntimeError as err:
            err_txt = str(err)
            # ถ้า CUDA ล้มเหลว (เช่นไดรเวอร์/PyTorch ไม่มี CUDA) ให้สลับรันด้วย --cpu อัตโนมัติทันที ไม่ปล่อยให้แครช!
            if cuda_ok and ("CUDA" in err_txt or "cuda" in err_txt):
                rprint("[yellow]⚠️ สตาร์ทโหมด CUDA ไม่ผ่าน — สลับไปรันโหมด `--cpu` อัตโนมัติเพื่อไม่ให้แครช[/yellow]")
                stop_comfyui(port=port)
                _spawn_comfy_process(root, port, lowvram=False, wait_s=wait_s, ui_mode="modern", use_cuda=False)
            elif mode == "classic":
                rprint("[yellow]⚠️ โหลดหน้าเว็บ Classic ไม่สำเร็จ — สลับเป็นหน้าเว็บมาตรฐานในตัวเครื่องแทน[/yellow]")
                stop_comfyui(port=port)
                SPEC_STATE["comfy_ui_mode"] = "modern"
                _spawn_comfy_process(root, port, lowvram=lowvram, wait_s=wait_s, ui_mode="modern", use_cuda=cuda_ok)
            else:
                raise

        if not _local_root_serves_html(port, timeout=6.0) and SPEC_STATE.get("comfy_ui_mode") == "classic":
            stop_comfyui(port=port)
            SPEC_STATE["comfy_ui_mode"] = "modern"
            _spawn_comfy_process(root, port, lowvram=lowvram, wait_s=wait_s, ui_mode="modern", use_cuda=_can_use_cuda_for_comfy())

    actual = port
    with contextlib.suppress(OSError):
        actual = parse_comfy_port(comfyui_log_path(root).read_text(encoding="utf-8", errors="ignore"), port)
    if actual != port and not comfyui_online(actual):
        actual = port
    local_url = f"http://127.0.0.1:{int(actual)}"
    rprint(f"[bold green]🎨 เซิร์ฟเวอร์ ComfyUI พร้อมใช้งานที่ {local_url} (พร้อมแสดงผลในเซลล์ที่ 3 — ไม่ใช้ Tunnel)[/bold green]")
    return local_url


def launch_comfyui_cell(
    model_key: str = "auto",
    port: int = COMFY_PORT,
    height: int = 820,
    ui_mode: str = "modern",
    force_restart: bool = False,
    model_dir: str | Path = MODEL_DIR,
) -> str:
    """ทางเข้าหลักสำหรับ 'เซลล์ที่ 3' ในโน้ตบุ๊ก:
    1) ติดตั้ง/ตรวจสอบ ComfyUI + ซ่อมแซม PyTorch CUDA อัตโนมัติถ้าเป็น CPU-only
    2) ผูกโมเดลภาพทุกตัวที่โหลดไว้แล้วเข้า ComfyUI (หรือโหลดตัวที่ระบุถ้ายังไม่มี)
    3) เปิด ComfyUI ในเครื่องแล้วฝังหน้าจอลงในเซลล์ที่ 3 โดยตรง (ไม่ผ่าน Cloudflare Tunnel)
    """
    from IPython.display import HTML, display

    ensure_comfyui(pull=False)
    mk = (model_key or "auto").strip()
    if mk and mk != "auto":
        if not image_model_status(mk, model_dir)["downloaded"]:
            install_image_model(mk, model_dir=model_dir)
        link_image_model(mk, model_dir=model_dir)
    ready = link_all_downloaded_image_models(model_dir=model_dir)
    if ready:
        rprint(f"[bold cyan]🔗 ผูกโมเดลภาพที่โหลดไว้แล้วเข้า ComfyUI:[/bold cyan] {', '.join(ready)}")
    else:
        rprint("[yellow]ℹ️ ยังไม่มีโมเดลภาพที่ดาวน์โหลดไว้ — เลือก IMAGE_MODEL ด้านบนหรือกด 📥 โหลด ในเซลล์ที่ 2 ก่อนวาดภาพครับ[/yellow]")

    url = start_comfyui(port=port, force_restart=force_restart, ui_mode=ui_mode, public="local")
    actual_port = int(url.rsplit(":", 1)[-1]) if ":" in url else int(port)
    dev_mode = "🎮 GPU CUDA" if _can_use_cuda_for_comfy() else "🖥️ CPU Mode (แนะนำเปิด Runtime → T4 GPU)"
    display(
        HTML(
            "<div style='background:#0f172a;color:#f8fafc;padding:10px 14px;border-radius:10px;"
            "border:1px solid #10b981;font-family:sans-serif;margin-bottom:8px;font-size:13px;'>"
            f"<b style='color:#34d399;'>🎨 ComfyUI Embedded Studio (พอร์ต {actual_port} • {dev_mode} • ไม่ใช้ Tunnel)</b>"
            f"<span style='color:#94a3b8;margin-left:10px;'>โมเดลที่พร้อมใช้: {', '.join(ready) if ready else 'ยังไม่ได้โหลด'}</span>"
            "</div>"
        )
    )
    if not open_colab_port_iframe(actual_port, height=height):
        display(
            HTML(
                f"<iframe src='http://127.0.0.1:{actual_port}/' width='100%' height='{int(height)}' "
                "style='border:1px solid #334155;border-radius:10px;background:#111827;'></iframe>"
            )
        )
    return url


def _close_comfy_log() -> None:
    global _COMFY_LOG
    if _COMFY_LOG is not None:
        with contextlib.suppress(Exception):
            _COMFY_LOG.close()  # type: ignore[attr-defined]
        _COMFY_LOG = None


def stop_comfyui(port: int = COMFY_PORT) -> bool:
    global _COMFY_PROC
    had = False
    if _COMFY_PROC is not None and _COMFY_PROC.poll() is None:
        _COMFY_PROC.terminate()
        try:
            _COMFY_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _COMFY_PROC.kill()
        had = True
    _COMFY_PROC = None
    _close_comfy_log()
    extra = _kill_matching_pids(f"main\\.py.*--port {int(port)}")
    stop_tunnel()
    if had or extra:
        rprint("[bold yellow]⏹ หยุด ComfyUI และล้างโปรเซสค้างเรียบร้อย[/bold yellow]")
        return True
    return False


# ---------------------------------------------------------------------------
# สร้างภาพจากแท็บแชท (ComfyUI API) — คืนไฟล์ PNG ให้ UI ฝังใน bubble ได้เลย
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("STRATA_OUTPUT_DIR", str(Path(STRATA_HOME) / "outputs")))

IMAGE_SAMPLING: dict[str, dict] = {
    "z-image-turbo": {"family": "z-image", "clip_type": "lumina2", "steps": 9, "cfg": 1.0,
                      "sampler": "euler", "scheduler": "simple", "width": 1024, "height": 1024},
    "chroma": {"family": "chroma", "clip_type": "chroma", "steps": 22, "cfg": 4.0,
               "sampler": "euler", "scheduler": "beta", "width": 1024, "height": 1024},
    "qwen-image-turbo": {"family": "qwen-image", "clip_type": "qwen_image", "steps": 8, "cfg": 1.0,
                         "sampler": "euler", "scheduler": "simple", "width": 1024, "height": 1024},
    "qwen-image": {"family": "qwen-image", "clip_type": "qwen_image", "steps": 20, "cfg": 3.0,
                   "sampler": "euler", "scheduler": "simple", "width": 1024, "height": 1024},
    "flux3": {"family": "flux3", "clip_type": "flux3", "steps": 16, "cfg": 1.0,
              "sampler": "euler", "scheduler": "simple", "width": 1024, "height": 1024},
    "flux": {"family": "flux", "clip_type": "flux", "steps": 8, "cfg": 1.0,
             "sampler": "euler", "scheduler": "simple", "width": 1024, "height": 1024},
}


def sampling_for(key: str) -> dict:
    cfg = image_preset(key)
    hint = f"{key} {cfg['repo_id']}".lower()
    if "z-image" in hint or "z_image" in hint:
        return dict(IMAGE_SAMPLING["z-image-turbo"])
    if "chroma" in hint:
        return dict(IMAGE_SAMPLING["chroma"])
    if "flux-3" in hint or "flux3" in hint:
        return dict(IMAGE_SAMPLING["flux3"])
    if "flux" in hint:
        return dict(IMAGE_SAMPLING["flux"])
    if "turbo" in hint:
        return dict(IMAGE_SAMPLING["qwen-image-turbo"])
    return dict(IMAGE_SAMPLING["qwen-image"])


def _preferred_clip_device() -> str:
    if _can_use_cuda_for_comfy():
        with contextlib.suppress(Exception):
            from .server import gpu_devices

            devs = gpu_devices()
            if devs and max(float(d.get("vram_total_gb", 0.0)) for d in devs) >= 8.0:
                return "default"
    return "cpu"


def _comfy_post(port: int, path: str, payload: dict, timeout: float = 30.0) -> dict:
    import requests

    r = requests.post(f"http://127.0.0.1:{port}{path}", json=payload, timeout=timeout)
    if r.status_code >= 400:
        detail = r.text[:800]
        with contextlib.suppress(Exception):
            body = r.json()
            err = body.get("error") or {}
            nodes = body.get("node_errors") or {}
            detail = json.dumps({"error": err, "node_errors": nodes}, ensure_ascii=False)[:1200]
        raise RuntimeError(f"ComfyUI ปฏิเสธ workflow (HTTP {r.status_code}): {detail}")
    return r.json()


def _comfy_get(port: int, path: str, timeout: float = 10.0):
    import requests

    r = requests.get(f"http://127.0.0.1:{port}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def ensure_gguf_nodes(comfy_dir: str | Path = COMFY_DIR) -> bool:
    target = Path(comfy_dir) / "custom_nodes" / GGUF_NODE_DIRNAME
    if (target / "__init__.py").is_file() and not _is_legacy_gguf_node(target):
        return True
    with contextlib.suppress(Exception):
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        _clone_or_tarball(GGUF_NODE_REPO, target, marker="__init__.py")
        node_req = target / "requirements.txt"
        if node_req.is_file():
            _install_req_preserving_cuda_torch(node_req)
        rprint("[dim]🔌 ติดตั้ง custom node leejet/ComfyUI-GGUF เรียบร้อย[/dim]")
        return (target / "__init__.py").is_file()
    return False


def image_workflow(key: str, prompt: str, negative: str = "", width: int | None = None,
                   height: int | None = None, steps: int | None = None, seed: int | None = None,
                   clip_device: str = "cpu", fast_cache: bool = False, cache_threshold: float = 0.12,
                   tiled_vae: bool = False) -> dict:
    cfg = image_preset(key)
    sp = sampling_for(key)
    seed = int(seed if (seed is not None and int(seed) >= 0) else random.randint(1, 2**31 - 1))
    w, h = int(width or sp["width"]), int(height or sp["height"])
    eff_steps = int(steps if (steps is not None and int(steps) > 0) else sp["steps"])
    names = {rel: Path(rel).name for rel in cfg["targets"]}
    unet = next((names[r] for r in names if cfg["targets"][r] == "diffusion_models"), "")
    clip = next((names[r] for r in names if cfg["targets"][r] == "text_encoders"), "")
    vae = next((names[r] for r in names if cfg["targets"][r] == "vae"), "")
    if not unet or not clip:
        raise RuntimeError(f"preset {key} ไม่มีไฟล์ diffusion model/text encoder ให้ใช้")
    if not vae:
        raise RuntimeError(f"preset {key} ไม่มีไฟล์ VAE สำหรับถอดรหัสภาพ (ต้องระบุไฟล์ในโฟลเดอร์ vae)")

    if clip.lower().endswith(".gguf"):
        clip_node = {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip, "type": sp["clip_type"]}}
    else:
        clip_inputs: dict = {"clip_name": clip, "type": sp["clip_type"]}
        if clip_device:
            clip_inputs["device"] = clip_device
        clip_node = {"class_type": "CLIPLoader", "inputs": clip_inputs}

    latent_cls = "EmptySD3LatentImage" if sp["family"] in ("qwen-image", "z-image", "chroma", "flux", "flux3") else "EmptyLatentImage"
    use_cache = bool(fast_cache) and eff_steps >= 12 and float(cache_threshold) > 0.0
    model_ref = ["10", 0] if use_cache else ["1", 0]
    decode_node = (
        {
            "class_type": "VAEDecodeTiled",
            "inputs": {"samples": ["6", 0], "vae": ["8", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8},
        }
        if tiled_vae
        else {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["8", 0]}}
    )
    graph: dict = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        "2": clip_node,
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative or "", "clip": ["2", 0]}},
        "5": {"class_type": latent_cls, "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": model_ref, "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
            "seed": seed, "steps": eff_steps, "cfg": float(sp["cfg"]),
            "sampler_name": sp["sampler"], "scheduler": sp["scheduler"], "denoise": 1.0,
        }},
        "7": decode_node,
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "strata"}},
    }
    if use_cache:
        graph["10"] = {
            "class_type": "StrataFastCache",
            "inputs": {"model": ["1", 0], "threshold": float(cache_threshold), "warmup_steps": 3},
        }
    return graph


def generate_image(key: str, prompt: str, negative: str = "", width: int | None = None, height: int | None = None,
                   steps: int | None = None, seed: int | None = None, port: int = COMFY_PORT,
                   wait_s: float = 420.0, comfy_dir: str | Path = COMFY_DIR,
                   fast_cache: bool = True, cache_threshold: float = 0.12,
                   on_progress: Callable[[str], None] | None = None) -> dict:
    import requests

    def _notify(msg: str) -> None:
        if on_progress is not None:
            with contextlib.suppress(Exception):
                on_progress(msg)

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("ต้องมีคำสั่ง (prompt) ก่อนสร้างภาพ")
    st = image_model_status(key)
    if not st["downloaded"]:
        raise RuntimeError(
            f"โมเดลภาพ {key} ยังโหลดไม่ครบ (ขาด: {', '.join(st['missing'])}) — กด 📥 โหลด ในแท็บ 1 ก่อน"
        )
    croot = Path(comfy_dir)
    if not (croot / "main.py").is_file():
        _notify("🔧 กำลังติดตั้ง ComfyUI ครั้งแรก...")
        ensure_comfyui(pull=False, comfy_dir=croot)
    node_dir = croot / "custom_nodes" / GGUF_NODE_DIRNAME
    need_node_fix = not (node_dir / "__init__.py").is_file() or _is_legacy_gguf_node(node_dir)
    _, wrote_cache = ensure_fast_cache_node(croot)
    if need_node_fix:
        _notify("🔌 กำลังติดตั้งโหนด leejet/ComfyUI-GGUF...")
        ensure_gguf_nodes(croot)
    if need_node_fix or (wrote_cache and fast_cache):
        if comfyui_online(port):
            _notify("🔄 รีสตาร์ท ComfyUI เพื่อโหลดโหนดใหม่...")
            stop_comfyui(port=port)
    link_image_model(key, comfy_dir=croot)
    _free_single_gpu_if_needed()
    if not comfyui_online(port):
        _notify("🚀 กำลังเปิดเซิร์ฟเวอร์ ComfyUI...")
        start_comfyui(port=port, comfy_dir=croot, public="local")

    c_dev = _preferred_clip_device()
    sp_cfg = sampling_for(key)
    eff_steps = int(steps if (steps is not None and int(steps) > 0) else sp_cfg["steps"])
    graph = image_workflow(
        key, prompt, negative, width, height, steps, seed,
        clip_device=c_dev, fast_cache=fast_cache, cache_threshold=cache_threshold, tiled_vae=True,
    )
    used_cache = "10" in graph
    t0 = time.perf_counter()
    _notify(f"📤 ส่งคำสั่งเข้าคิว ({eff_steps} steps • Text Encoder={c_dev})...")
    try:
        res = _comfy_post(port, "/prompt", {"prompt": graph, "client_id": f"strata-{os.getpid()}"})
    except RuntimeError as err:
        err_s = str(err)
        if any(tok in err_s for tok in ("StrataFastCache", "device", "VAEDecodeTiled", "temporal_size")):
            keep_dev = "" if "device" in err_s else c_dev
            keep_fc = False if "StrataFastCache" in err_s else fast_cache
            keep_tile = False if ("VAEDecodeTiled" in err_s or "temporal_size" in err_s) else True
            graph = image_workflow(
                key, prompt, negative, width, height, steps, seed,
                clip_device=keep_dev, fast_cache=keep_fc, cache_threshold=cache_threshold, tiled_vae=keep_tile,
            )
            used_cache = "10" in graph
            res = _comfy_post(port, "/prompt", {"prompt": graph, "client_id": f"strata-{os.getpid()}"})
        else:
            raise
    pid = res.get("prompt_id", "")
    if not pid:
        raise RuntimeError(f"ComfyUI ไม่คืน prompt_id: {res}")

    deadline = time.time() + wait_s
    files: list[dict] = []
    dead_strikes = 0
    while time.time() < deadline:
        sec_now = int(time.perf_counter() - t0)
        try:
            hist = _comfy_get(port, f"/history/{pid}", timeout=8.0).get(pid) or {}
            dead_strikes = 0
        except requests.Timeout:
            _notify(f"🎨 กำลังโหลดน้ำหนักโมเดลเข้าหน่วยความจำ… ({sec_now}s)")
            continue
        except requests.ConnectionError as cerr:
            dead_strikes += 1
            if dead_strikes >= 3 and (_COMFY_PROC is None or _COMFY_PROC.poll() is not None):
                log_p = comfyui_log_path(croot)
                tail = "\n".join(log_p.read_text(encoding="utf-8", errors="ignore").splitlines()[-15:]) if log_p.is_file() else ""
                raise RuntimeError(
                    "เซิร์ฟเวอร์ ComfyUI ปิดตัวเองระหว่างวาดภาพ (ลองใช้โมเดล z-image-turbo-heretic ~6.6GB หรือลดขนาดเป็น 768×768):\n"
                    + (tail or str(cerr))
                ) from cerr
            time.sleep(1.0)
            continue

        status = hist.get("status") or {}
        if status.get("status_str") == "error":
            msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
            raise RuntimeError(f"ComfyUI รันไม่สำเร็จ: {json.dumps(msgs, ensure_ascii=False)[:800]}")
        outs = hist_outputs(hist)
        if outs:
            files = outs
            break
        _notify(f"🎨 กำลังวาดภาพด้วย {key} ({eff_steps} steps)… ผ่านไป {sec_now}s")
        time.sleep(1.0)
    if not files:
        raise TimeoutError(f"ComfyUI ไม่สร้างภาพเสร็จใน {wait_s:.0f}s (ดู log ที่ {comfyui_log_path(croot)})")
    image = files[0]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / f"{key}_{time.strftime('%Y%m%d_%H%M%S')}_{image['filename']}"
    with requests.get(f"http://127.0.0.1:{port}/view", timeout=120,
                      params={"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                              "type": image.get("type", "output")}) as r:
        r.raise_for_status()
        dest.write_bytes(r.content)
    elapsed = time.perf_counter() - t0
    fc_note = " • ⚡FastCache" if used_cache else ""
    rprint(f"[bold green]✅ สร้างภาพเสร็จใน {elapsed:.1f}s{fc_note}[/bold green] → {dest}")
    return {
        "path": str(dest), "seconds": round(elapsed, 1), "seed": graph["6"]["inputs"]["seed"],
        "prompt_id": pid, "key": key, "fast_cache": used_cache,
    }


def hist_outputs(hist: dict) -> list[dict]:
    out: list[dict] = []
    for node_out in (hist.get("outputs") or {}).values():
        for img in node_out.get("images", []) or []:
            if img.get("filename"):
                out.append(img)
    return out


def list_generated(limit: int = 12) -> list[str]:
    if not OUTPUT_DIR.is_dir():
        return []
    files = sorted((p for p in OUTPUT_DIR.glob("*.png") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


def image_data_uri(path: str | Path, max_width: int = 640) -> str:
    p = Path(path)
    data = p.read_bytes()
    mime = "image/png"
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        pass
    else:
        with contextlib.suppress(Exception), Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                im = im.resize((max_width, int(im.height * max_width / im.width)))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def comfyui_status(port: int = COMFY_PORT) -> dict:
    return {
        "online": comfyui_online(port),
        "port": port,
        "url": comfyui_public_url(port) if comfyui_online(port) else "",
        "installed": (Path(COMFY_DIR) / "main.py").is_file(),
        "models_gb": round(_dir_bytes(Path(COMFY_DIR) / "models") / (1024**3), 2) if Path(COMFY_DIR).is_dir() else 0.0,
    }


__all__ = [
    "_extract_tarball",
    "comfy_allows_cross_site",
    "comfy_object_info_ok",
    "comfy_runtime_env",
    "ensure_cloudflared",
    "ensure_fast_cache_node",
    "ensure_public_url",
    "ensure_torch_runtime",
    "file_sources",
    "launch_comfyui_cell",
    "link_all_downloaded_image_models",
    "open_colab_port_iframe",
    "open_colab_port_window",
    "parse_cloudflared_url",
    "parse_comfy_port",
    "public_url_works",
    "resolve_remote_image_file",
    "start_tunnel",
    "stop_tunnel",
    "_fetch_tarball",
    "_run_git",
    "comfy_target_path",
    "comfyui_command",
    "comfyui_online",
    "comfyui_public_url",
    "comfyui_status",
    "ensure_comfyui",
    "image_model_status",
    "image_preset",
    "include_patterns",
    "install_image_model",
    "link_image_model",
    "start_comfyui",
    "stop_comfyui",
]

if __name__ == "__main__":  # pragma: no cover
    for k in IMAGE_PRESET_KEYS:
        st = image_model_status(k)
        print(f"{k}: {st['gb']}/{st['approx_gb']} GB • ครบ={st['downloaded']}")
