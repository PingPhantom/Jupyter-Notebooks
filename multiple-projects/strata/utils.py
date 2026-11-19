"""ยูทิลิตี้พื้นฐาน: รันคำสั่ง shell อย่างปลอดภัย (ไม่ใช้ PTY ที่ทำให้โปรเซสค้างบน Colab), ยกเลิกกลางคันได้, วัดขนาดโฟลเดอร์แบบ cache."""
from __future__ import annotations

import contextlib
import importlib
import os
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections import deque
from collections.abc import Callable
from pathlib import Path

import requests

LOG_RING: deque = deque(maxlen=600)  # บรรทัด log ล่าสุด (ให้ UI แสดงแบบเรียลไทม์ผ่าน HTML ที่ thread-safe 100%)
STREAM_HOOKS: list[Callable[[str, bool], None]] = []

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
_RING_LOCK = threading.Lock()
_LAST_WAS_PROGRESS = False
_ACTIVE_PROCS: set[subprocess.Popen] = set()
_PROCS_LOCK = threading.Lock()


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def recent_log(n: int = 12) -> list[str]:
    """n บรรทัดล่าสุดจากคำสั่งที่รันผ่าน run_quiet/run_stream"""
    with _RING_LOCK:
        return list(LOG_RING)[-int(n):]


def _notify_hooks(clean: str, is_cr: bool) -> None:
    for fn in list(STREAM_HOOKS):
        with contextlib.suppress(Exception):
            fn(clean, is_cr)


def _remember(line: str, *, replace_progress: bool = False) -> None:
    global _LAST_WAS_PROGRESS
    clean = strip_ansi(str(line)).rstrip("\r\n")
    if not clean.strip():
        return
    with _RING_LOCK:
        if replace_progress and _LAST_WAS_PROGRESS and LOG_RING:
            LOG_RING[-1] = clean
        else:
            LOG_RING.append(clean)
        _LAST_WAS_PROGRESS = bool(replace_progress)
    _notify_hooks(clean, bool(replace_progress))


def abort_active_streams() -> int:
    """สั่งหยุดคำสั่ง shell เบื้องหลังที่กำลังรันค้างอยู่ทั้งหมดทันที (ใช้คู่กับปุ่ม ⏹️ ยกเลิกงาน / ปลดล็อก)"""
    with _PROCS_LOCK:
        procs = list(_ACTIVE_PROCS)
    killed = 0
    for p in procs:
        if p.poll() is None:
            with contextlib.suppress(Exception):
                p.terminate()
                killed += 1
    if killed:
        time.sleep(0.4)
        for p in procs:
            if p.poll() is None:
                with contextlib.suppress(Exception):
                    p.kill()
    return killed


def run_quiet(cmd: str):
    p = subprocess.run(["/bin/bash", "-c", cmd], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd)


def _consume_stream_chunk(buf: str, chunk: str, *, echo_stdout: bool) -> str:
    buf += chunk
    while True:
        idx_r = buf.find("\r")
        idx_n = buf.find("\n")
        if idx_r == -1 and idx_n == -1:
            break
        if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
            if idx_r + 1 < len(buf) and buf[idx_r + 1] == "\n":
                seg = buf[:idx_r]
                buf = buf[idx_r + 2:]
                clean = strip_ansi(seg)
                if clean.strip():
                    _remember(clean, replace_progress=False)
                    if echo_stdout:
                        print(clean, flush=True)
            else:
                seg = buf[:idx_r]
                buf = buf[idx_r + 1:]
                clean = strip_ansi(seg)
                if clean.strip():
                    _remember(clean, replace_progress=True)
        else:
            seg = buf[:idx_n]
            buf = buf[idx_n + 1:]
            clean = strip_ansi(seg)
            if clean.strip():
                _remember(clean, replace_progress=False)
                if echo_stdout:
                    print(clean, flush=True)
    return buf


def run_stream(cmd: str, cwd: str | Path | None = None):
    """รันคำสั่งแบบ PIPE มาตรฐาน (ไม่ใช้ PTY เพื่อกัน TTY hang) และพิมพ์ออก stdout เฉพาะบน main thread (กันชน IPython ZMQ)"""
    global _LAST_WAS_PROGRESS
    where = f" (ใน {cwd})" if cwd else ""
    shown = "\n".join(f"$ {line}" for line in cmd.strip().splitlines())
    _LAST_WAS_PROGRESS = False
    _remember(f"{shown}{where}")
    on_main = threading.current_thread() is threading.main_thread()

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_XET": "1",
    }
    p = subprocess.Popen(
        ["/bin/bash", "-c", cmd],
        cwd=str(cwd) if cwd else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    with _PROCS_LOCK:
        _ACTIVE_PROCS.add(p)
    buf = ""
    try:
        if p.stdout:
            while True:
                chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(512)
                if not chunk:
                    break
                buf = _consume_stream_chunk(buf, chunk.decode("utf-8", errors="ignore"), echo_stdout=on_main)
        if buf.strip():
            clean = strip_ansi(buf)
            _remember(clean, replace_progress=False)
            if on_main:
                print(clean, flush=True)
        p.wait()
    finally:
        with _PROCS_LOCK:
            _ACTIVE_PROCS.discard(p)
        _LAST_WAS_PROGRESS = False

    _remember(f"(exit {p.returncode})")
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def uv_add(*packages: str):
    if not packages:
        return
    for pkg in packages:
        if not re.fullmatch(r"[A-Za-z0-9_.\-\[\]=<>!,~*+/]+", pkg):
            raise ValueError(f"ชื่อแพ็กเกจไม่ถูกต้อง: {pkg!r}")
    s = " ".join(shlex.quote(p) for p in packages)
    run_quiet(f"uv add --quiet {s} && uv export --format requirements-txt --no-hashes --no-emit-project --quiet -o requirements.txt")
    importlib.invalidate_caches()
    print(f"✅ ติดตั้ง {s} เรียบร้อย!")


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if not p.is_symlink() and p.is_file())


_DIR_CACHE: dict[str, tuple[float, int, int]] = {}

_DIR_CACHE_TTL = 30.0


def _dir_stats(path: Path, ttl: float = _DIR_CACHE_TTL) -> dict:
    key = str(Path(path))
    now = time.monotonic()
    hit = _DIR_CACHE.get(key)
    if hit and ttl > 0 and (now - hit[0]) < ttl:
        return {"bytes": hit[1], "gguf": hit[2], "cached\": True" if False else "cached": True}
    p = Path(path)
    total = gguf = 0
    if p.is_file():
        total = p.stat().st_size
    elif p.exists():
        for f in p.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            try:
                total += f.stat().st_size
            except OSError:
                continue
            if f.suffix.lower() == ".gguf":
                gguf += 1
    _DIR_CACHE[key] = (now, total, gguf)
    return {"bytes": total, "gguf": gguf, "cached": False}


def _invalidate_dir_cache(prefix: Path | None = None):
    if prefix is None:
        _DIR_CACHE.clear()
        return
    pref = str(Path(prefix)).rstrip("/")
    for k in [k for k in _DIR_CACHE if k == pref or k.startswith(pref + "/")]:
        _DIR_CACHE.pop(k, None)


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024**3):.2f} GB" if n >= 1024**3 else (f"{n / (1024**2):.2f} MB" if n >= 1024**2 else f"{n / 1024:.1f} KB")


def _read_net_counters() -> tuple[int, int]:
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8", errors="ignore").splitlines()[2:]
    except OSError:
        return 0, 0
    rx = tx = 0
    for line in lines:
        iface, sep, data = line.partition(":")
        if not sep or iface.strip() == "lo":
            continue
        parts = data.split()
        if len(parts) < 9:
            continue
        try:
            rx += int(parts[0])
            tx += int(parts[8])
        except ValueError:
            continue
    return rx, tx


class NetworkMonitor:
    def __init__(self, reader=_read_net_counters, clock=time.monotonic):
        self._read = reader
        self._clock = clock
        self._prev: tuple[float, int, int] | None = None

    def sample(self) -> dict:
        now = self._clock()
        try:
            rx, tx = self._read()
        except Exception:
            rx = tx = 0
        prev = self._prev
        self._prev = (now, rx, tx)
        down = up = 0.0
        if prev is not None:
            dt = now - prev[0]
            if dt > 0 and rx >= prev[1] and tx >= prev[2]:
                down = (rx - prev[1]) / dt
                up = (tx - prev[2]) / dt
        return {
            "down_bps": down,
            "up_bps": up,
            "rx_bytes": rx,
            "tx_bytes": tx,
            "ok": prev is not None,
        }

    def reset(self) -> None:
        self._prev = None


_NET_MONITOR = NetworkMonitor()


def net_bandwidth() -> dict:
    return _NET_MONITOR.sample()


def _fmt_rate(bps: float) -> str:
    bps = max(0.0, float(bps))
    if bps < 1024:
        return f"{bps:.0f} B/s"
    if bps < 1024**2:
        return f"{bps / 1024:.0f} KB/s"
    if bps < 1024**3:
        return f"{bps / 1024**2:.2f} MB/s"
    return f"{bps / 1024**3:.2f} GB/s"


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    members = []
    for m in tf.getmembers():
        target = (root / m.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"ไฟล์ใน tar ชี้นอกโฟลเดอร์ปลายทาง: {m.name}")
        if m.isfile() or m.isdir():
            members.append(m)
    if sys.version_info >= (3, 12):
        tf.extractall(root, filter="data")
    else:
        tf.extractall(root, members=members)


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for info in zf.infolist():
        target = (root / info.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"ไฟล์ใน zip ชี้นอกโฟลเดอร์ปลายทาง: {info.filename}")
    zf.extractall(root)


def _size_txt(num_bytes: float) -> str:
    """แปลงไบต์เป็นข้อความสั้น ๆ (ใช้กับการ์ดความคืบหน้า)"""
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.0f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{int(num_bytes)} B"


def download_progress(
    url: str,
    dest_path: Path,
    *,
    label: str = "",
    prefix: str = "   ",
    every: float = 1.0,
) -> Path:
    """ดาวน์โหลดแบบสตรีม พร้อมพิมพ์ % ความคืบหน้าทุก ๆ ``every`` วินาที

    ใช้ ``Content-Length`` ของเซิร์ฟเวอร์เป็นตัวหาร — ถ้าเซิร์ฟเวอร์ไม่ส่งขนาดมา จะพิมพ์
    ปริมาณที่โหลดไปแล้วอย่างเดียว (ไม่เดา % ให้เข้าใจผิด)
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    label = label or dest_path.name
    started = time.perf_counter()
    done = 0
    last_print = 0.0
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or r.headers.get("content-length") or 0)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                now = time.perf_counter()
                if now - last_print < every or (total and done >= total):
                    continue  # บรรทัด 100% มีอยู่แล้วในสรุปท้ายฟังก์ชัน
                last_print = now
                speed = done / max(now - started, 1e-6)
                if total:
                    pct = 100.0 * done / total
                    bar_len = 20
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    _progress_print(f"{prefix}⬇️ {label}: {bar} {pct:5.1f}% "
                                    f"({_size_txt(done)}/{_size_txt(total)} · {_size_txt(speed)}/s)")
                else:
                    _progress_print(f"{prefix}⬇️ {label}: {_size_txt(done)} · {_size_txt(speed)}/s")
    if total and done < total:
        _progress_print(f"{prefix}⚠️ {label}: โหลดได้ {_size_txt(done)} จาก {_size_txt(total)} (เซิร์ฟเวอร์ปิดการเชื่อมต่อก่อน)")
    elif total:
        _progress_print(f"{prefix}✅ {label}: 100.0% ({_size_txt(done)}) ใช้เวลา {time.perf_counter() - started:.0f}s")
    return dest_path


def _progress_print(text: str) -> None:
    """พิมพ์บรรทัดความคืบหน้า — ใช้ rich ถ้ามี ไม่งั้น print ธรรมดา"""
    try:
        from rich import print as rprint

        rprint(f"[cyan]{text}[/cyan]")
    except Exception:  # noqa: BLE001
        print(text, flush=True)


def _download_and_extract(url: str, name: str, dest: Path, label: str = ""):
    arc = dest / name
    download_progress(url, arc, label=label or name, prefix="   ")
    try:
        _progress_print(f"   📦 แตกไฟล์ {label or name} → {dest} …")
        if name.endswith(".tar.gz"):
            with tarfile.open(arc, "r:gz") as tf:
                _safe_extract_tar(tf, dest)
        elif name.endswith(".zip"):
            with zipfile.ZipFile(arc, "r") as zf:
                _safe_extract_zip(zf, dest)
    finally:
        arc.unlink(missing_ok=True)
