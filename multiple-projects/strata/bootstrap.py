"""เตรียมสภาพแวดล้อมสำหรับ Colab: Nix (ปักหมุด nixpkgs) → devshell → uv venv

หลักการ: ไม่มี `curl | sh` (ดาวน์โหลดสคริปต์มาตรวจก่อนรัน + log sha256 ให้ปักหมุดได้),
ไม่ลบไฟล์ระบบของ Colab (ใช้ symlink), ป้องกัน .venv ทับ ipywidgets/comm/torch ของ Colab,
และเรียกซ้ำได้ปลอดภัย (idempotent)
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from rich import print as rprint

from .config import (
    CUSTOM_HF_REPO,
    MODEL_DIR,
    NVIDIA_BIN_DIRS,
    NVIDIA_LIB_DIRS,
    STRATA_HOME,
)
from .flake_template import FLAKE_TEMPLATE
from .utils import run_quiet, run_stream

NIXPKGS_REV = "d4f25a3e562008648d45bcf2d70bcc7422cad84e"
NIX_INSTALL_URL = "https://nixos.org/nix/install"
FLAKE_DIR = Path(STRATA_HOME) / "flake"
ENV_DUMP = "import os, json; print(json.dumps(dict(os.environ)))"
FLAKE_MARKER = "Strata Studio devshell"

FALLBACK_DEPS = [
    "numpy>=1.26,<3",
    "requests>=2.31,<3",
    "rich>=13.7,<16",
    "openai>=1.55,<4",
    "fastembed>=0.4,<1",
    "pypdf>=4.2,<8",
]

# แพ็กเกจที่ Google Colab ผูกเข้ากับ Kernel Comm และ GPU มาให้ตั้งแต่บูต — ห้ามให้ .venv มาบังเด็ดขาด
COLAB_SHADOW_PACKAGES = (
    "ipywidgets",
    "jupyterlab_widgets",
    "jupyterlab-widgets",
    "widgetsnbextension",
    "comm",
    "ipykernel",
    "ipython",
)

NIX_CONF = """experimental-features = nix-command flakes
sandbox = false
filter-syscalls = false
build-users-group =
max-jobs = auto
cores = 0
http-connections = 50
"""

SETUP_LOG = Path(STRATA_HOME) / "setup.log"


def _is_colab_runtime() -> bool:
    # Kaggle บางอิมเมจติด COLAB_* มาด้วย → ต้องกันไม่ให้เข้าใจผิดว่าเป็น Colab
    if any(k.startswith("KAGGLE_") for k in os.environ) or Path("/kaggle/working").is_dir():
        return False
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def setup_log_path() -> str:
    return str(SETUP_LOG)


def prepare_setup_log() -> str:
    global SETUP_LOG
    candidates = [Path(STRATA_HOME), Path(tempfile.gettempdir()) / "strata"]
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            log = base / "setup.log"
            log.write_text("", encoding="utf-8")
        except OSError:
            continue
        SETUP_LOG = log
        return str(SETUP_LOG)
    return str(SETUP_LOG)


def _emit(msg: str, plain: str | None = None) -> None:
    rprint(msg)
    text = plain if plain is not None else re.sub(r"\[/?[a-zA-Z0-9 #]+\]", "", msg)
    with contextlib.suppress(OSError):
        SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SETUP_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")


def _log(msg: str) -> None:
    _emit(f"[cyan]{msg}[/cyan]", plain=msg)


def write_nix_conf() -> None:
    _require_root("เขียน /etc/nix/nix.conf")
    Path("/etc/nix").mkdir(parents=True, exist_ok=True)
    Path("/etc/nix/nix.conf").write_text(NIX_CONF, encoding="utf-8")


def ensure_paths() -> None:
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH") or NVIDIA_LIB_DIRS
    extra = [
        str(Path.home() / ".local/bin"),
        str(Path.home() / ".nix-profile/bin"),
        "/nix/var/nix/profiles/default/bin",
        "/usr/local/nvidia/bin",
        "/usr/local/cuda/bin",
    ]
    for p in reversed(extra):
        if p not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = f"{p}{os.pathsep}{os.environ.get('PATH', '')}"


PIP_ONLY_DEPS = [
    "numpy>=1.26,<3",
    "requests>=2.31,<3",
    "rich>=13.7,<16",
    "openai>=1.55,<4",
    "huggingface-hub>=0.26,<2",
    "pydantic>=2,<3",
]

_STEP_COUNTER = {"i": 0, "total": 0, "t0": 0.0}


def _plan_steps(use_nix: bool) -> list[str]:
    if use_nix:
        return [
            "เตรียมโฟลเดอร์",
            "PATH/ตัวแปรระบบ",
            "ติดตั้ง Nix",
            "ตั้งค่า shell ใหม่",
            "เขียน flake.nix",
            "โหลด devshell (นานสุด)",
            "เชื่อมเครื่องมือ",
            "สร้าง venv + deps",
        ]
    return ["เตรียมโฟลเดอร์", "PATH/ตัวแปรระบบ", "ติดตั้งแพ็กเกจ Python (pip)"]


def environment_plan(nix: str = "auto", name: str | None = None) -> dict:
    if name is None:
        from .platforms import detect

        name = detect()["name"]
    mode = str(nix).lower()
    if mode in ("always", "true", "yes", "1"):
        use_nix = True
    elif mode in ("never", "false", "no", "0"):
        use_nix = False
    else:
        use_nix = name != "kaggle"
    notes = []
    if name == "kaggle" and use_nix:
        forced = os.environ.get("STRATA_FORCE_NIX", "").strip().lower() in ("1", "true", "yes", "on")
        if forced:
            notes.append(
                "⚠️ STRATA_FORCE_NIX เปิดอยู่ — ใช้ Nix บน Kaggle (ติดตั้งใหม่ทุก session 5-20 นาที) "
                "ถ้าไม่จำเป็นถอด env นี้ออกเพื่อกลับไปโหมด pip-only"
            )
        else:
            use_nix = False
            notes.append(
                "Kaggle ไม่มี systemd และ /nix ถูกล้างทุก session — บังคับโหมด pip-only ให้เลย "
                "(จะใช้ Nix ก็ต่อเมื่อตั้ง env STRATA_FORCE_NIX=1)"
            )
    if not use_nix:
        notes.append("โหมด pip-only: ไม่ต้องใช้ Nix/Docker — llama.cpp โหลดไบนารี prebuilt CUDA ให้เอง")
    return {"name": name, "nix": mode, "use_nix": use_nix, "steps": _plan_steps(use_nix), "notes": notes}


def _step(label: str) -> None:
    _STEP_COUNTER["i"] += 1
    total = _STEP_COUNTER["total"] or 1
    el = time.perf_counter() - (_STEP_COUNTER["t0"] or time.perf_counter())
    _emit(f"▶️ [{_STEP_COUNTER['i']}/{total}] {label} ({el:.0f}s)")


def _run_quiet_heartbeat(cmd: str, label: str, every: float = 20.0) -> None:
    started = time.perf_counter()
    proc = subprocess.Popen(["/bin/bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while proc.poll() is None:
        time.sleep(every)
        if proc.poll() is None:
            rprint(f"[dim]   … {label}: ยังทำงานอยู่ ({time.perf_counter() - started:.0f}s)[/dim]", flush=True)
    out = proc.stdout.read() if proc.stdout else ""
    if proc.returncode != 0:
        print(out, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _require_root(what: str) -> None:
    if os.geteuid() == 0:
        return
    raise RuntimeError(
        f'{what} ต้องใช้สิทธิ์ root — Kaggle/Colab รันเป็น root อยู่แล้ว ถ้ารันใน container อื่น '
        'ให้ใช้โหมด pip-only (USE_NIX="never") แทน ซึ่งไม่ต้องใช้ Nix/Docker เลย'
    )


def install_nix() -> None:
    if shutil.which("nix"):
        return
    _require_root("ติดตั้ง Nix (เขียน /nix และ /etc/nix)")
    _log("⏳ กำลังติดตั้ง Nix Package Manager...")
    resp = requests.get(NIX_INSTALL_URL, timeout=120)
    resp.raise_for_status()
    script = resp.content
    digest = hashlib.sha256(script).hexdigest()
    rprint(f"[dim]   installer sha256 = {digest}[/dim]")

    expected = os.environ.get("NIX_INSTALL_SHA256", "").strip().lower()
    if expected and digest != expected:
        raise RuntimeError(f"sha256 ของสคริปต์ติดตั้ง Nix ไม่ตรงที่คาดไว้ (ได้ {digest}) — ยกเลิกเพื่อความปลอดภัย")
    if not script.startswith(b"#!") or b"nix" not in script[:8192].lower():
        raise RuntimeError("เนื้อหาที่ดาวน์โหลดมาไม่ใช่สคริปต์ติดตั้ง Nix — ยกเลิกเพื่อความปลอดภัย")

    installer = Path("/tmp/nix-install.sh")
    installer.write_bytes(script)
    installer.chmod(0o700)
    _run_quiet_heartbeat(
        f"groupadd -r nixbld 2>/dev/null || true && sh {shlex.quote(str(installer))} --no-daemon",
        "ติดตั้ง Nix",
    )


def render_flake(repo_id: str = CUSTOM_HF_REPO) -> str:
    py_ver = f"python{sys.version_info.major}{sys.version_info.minor}"
    return (
        FLAKE_TEMPLATE
        .replace("@NIXPKGS_REV@", NIXPKGS_REV)
        .replace("@PY_VER@", py_ver)
        .replace("@CUSTOM_HF_REPO@", repo_id)
        .replace("@MODEL_DIR@", MODEL_DIR)
    )


def write_flake(repo_id: str = CUSTOM_HF_REPO, flake_dir: Path | None = None) -> Path:
    target_dir = Path(flake_dir or FLAKE_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    content = render_flake(repo_id)
    target = target_dir / "flake.nix"
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")

    for legacy in (Path("flake.nix"), Path("default.nix")):
        try:
            if legacy.is_file():
                text = legacy.read_text(encoding="utf-8", errors="ignore")
                if FLAKE_MARKER in text and "@NIXPKGS_REV@" not in text:
                    legacy.unlink()
        except OSError:
            pass
    return target


NIX_BIN_DIRS = ("~/.nix-profile/bin", "/nix/var/nix/profiles/default/bin")
NIX_BINARIES = ("nix", "nix-store", "nix-env", "nix-build", "nix-shell", "nix-instantiate", "nix-channel", "nix-hash")
SHELL_MARK = "# >>> strata nix >>>"
SHELL_SNIPPET = (
    f"{SHELL_MARK}\n"
    "# ให้ `nix` ใช้ได้ในทุก shell (Colab terminal, bash -lc, ฯลฯ)\n"
    'export PATH="$HOME/.nix-profile/bin:/nix/var/nix/profiles/default/bin:$PATH"\n'
    f"# <<< strata nix <<<\n"
)


def persist_shell_env(profile_dir: Path | None = None, bashrc: Path | None = None) -> dict:
    written: list = []
    pdir = Path(profile_dir or "/etc/profile.d")
    try:
        pdir.mkdir(parents=True, exist_ok=True)
        target = pdir / "strata-nix.sh"
        if not target.exists() or target.read_text(encoding="utf-8", errors="ignore") != SHELL_SNIPPET:
            target.write_text(SHELL_SNIPPET, encoding="utf-8")
            written.append(str(target))
    except OSError:
        pass

    rc = Path(bashrc or (Path.home() / ".bashrc"))
    try:
        old = rc.read_text(encoding="utf-8", errors="ignore") if rc.exists() else ""
        if SHELL_MARK not in old:
            rc.parent.mkdir(parents=True, exist_ok=True)
            rc.write_text(old + ("\n" if old and not old.endswith("\n") else "") + SHELL_SNIPPET, encoding="utf-8")
            written.append(str(rc))
    except OSError:
        pass
    return {"written": written, "snippet": SHELL_SNIPPET}


def link_nix_binaries(bin_dir: Path | None = None) -> list:
    dst_dir = Path(bin_dir or "/usr/local/bin")
    linked: list = []
    for name in NIX_BINARIES:
        src = shutil.which(name)
        if not src:
            src = next((str(Path(d).expanduser() / name) for d in NIX_BIN_DIRS if (Path(d).expanduser() / name).exists()), None)
        if not src or not Path(src).exists():
            continue
        dst = dst_dir / name
        try:
            if dst.exists() and dst.resolve() == Path(src).resolve():
                continue
            dst.unlink(missing_ok=True)
            dst.symlink_to(src)
            linked.append(str(dst))
        except OSError:
            continue
    return linked


def link_tools() -> None:
    for name, real in (("bash", "/bin/bash"), ("sh", "/bin/sh")):
        dst = Path("/usr/local/bin") / name
        try:
            if Path(real).exists() and (not dst.exists() or dst.resolve() != Path(real)):
                dst.unlink(missing_ok=True)
                dst.symlink_to(real)
        except OSError:
            pass
    for tool in ("uv", "ruff", "jq", "rg", "hf", "download-deepseek"):
        src = shutil.which(tool)
        if not src:
            continue
        dst = Path("/usr/local/bin") / tool
        try:
            if not dst.exists() or dst.resolve() != Path(src).resolve():
                dst.unlink(missing_ok=True)
                dst.symlink_to(src)
        except OSError:
            pass


REQUIRED_TOOLS = ("uv", "hf", "download-deepseek", "jq", "rg", "ruff")


def missing_tools() -> list:
    return [t for t in REQUIRED_TOOLS if not shutil.which(t)]


def _devshell_error(returncode: int, stderr: str, flake_dir: Path) -> str:
    tail = "\n".join((stderr or "").strip().splitlines()[-15:]) or "(ไม่มี stderr)"
    hints = (
        "ตรวจ `nix --version` และ /etc/nix/nix.conf ว่ามี `experimental-features = nix-command flakes`",
        f"ลองรันเองเพื่อดู error เต็ม: nix develop {flake_dir} --command python3 -c 'print(1)'",
        "ถ้า error พูดถึง git/tracked: flake ต้องอยู่นอก git repo (โค้ดนี้เขียนไว้ที่ "
        f"{flake_dir} แล้ว — ถ้าค่า FLAKE_DIR ยังชี้เข้า repo ให้ตั้ง env STRATA_HOME)",
    )
    return (
        f"`nix develop` ล้มเหลว (exit {returncode})\n--- stderr ---\n{tail}\n--- วิธีแก้ ---\n"
        + "\n".join(f"• {h}" for h in hints)
    )


def _nix_timeout() -> int:
    raw = os.environ.get("STRATA_NIX_TIMEOUT", "").strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return 1800  # 30 นาที — กันค้างไม่จบ (nix develop ครั้งแรกโหลด store เยอะ)


def activate_devshell(flake_dir: Path | None = None) -> dict:
    flake = Path(flake_dir or FLAKE_DIR)
    if not (flake / "flake.nix").is_file():
        write_flake(flake_dir=flake)
    timeout = _nix_timeout()
    _log(f"⏳ กำลังโหลด Nix devshell จาก {flake} (ครั้งแรกอาจใช้เวลาหลายนาที, เพดาน {timeout}s)...")
    try:
        proc = subprocess.run(
            ["nix", "develop", str(flake), "--command", "python3", "-c", ENV_DUMP],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"`nix develop` ใช้เวลาเกิน {timeout}s — ยกเลิกการรอ\n--- วิธีแก้ ---\n"
            '• ใช้โหมด pip-only เร็วกว่า: ตั้ง USE_NIX="never" (หรือไม่ต้องตั้งอะไรบน Kaggle ก็เป็น pip-only อยู่แล้ว)\n'
            "• ถ้าอยากรอนานขึ้น ตั้ง env STRATA_NIX_TIMEOUT=3600"
        ) from None
    if proc.returncode != 0:
        raise RuntimeError(_devshell_error(proc.returncode, proc.stderr, flake))
    try:
        env_dump = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            "`nix develop` รันจบแต่ไม่ได้คืนค่าสภาพแวดล้อม (stdout ไม่ใช่ JSON) — devshell อาจถูก interrupt\n"
            f"--- stdout (ท้าย) ---\n{(proc.stdout or '').strip()[-500:] or '(ว่าง)'}"
        ) from None
    for key, val in env_dump.items():
        if key in ("_", "SHLVL", "PWD", "OLDPWD"):
            continue
        if key == "PATH":
            os.environ["PATH"] = f"/usr/bin:/bin:{val}:{NVIDIA_BIN_DIRS}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin"
        elif key == "LD_LIBRARY_PATH":
            os.environ["LD_LIBRARY_PATH"] = f"{val}:{NVIDIA_LIB_DIRS}"
        else:
            os.environ[key] = val
    return {"flake_dir": str(flake), "vars": len(env_dump)}


def nix_diagnostics() -> dict:
    info: dict = {"flake_dir": str(FLAKE_DIR), "flake_exists": (FLAKE_DIR / "flake.nix").is_file()}
    for key, cmd in (
        ("version", ["nix", "--version"]),
        ("conf", ["nix", "config", "show", "experimental-features"]),
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True)
            info[key] = (out.stdout or out.stderr).strip()[:200]
        except OSError as e:
            info[key] = f"เรียกไม่ได้: {e}"
    info["store_ok"] = Path("/nix/store").is_dir()
    return info


def _strip_colab_shadow_packages(venv_path: Path) -> None:
    if not _is_colab_runtime():
        return
    uv_bin = shutil.which("uv")
    venv_py = venv_path / "bin" / "python"
    if uv_bin and venv_py.is_file():
        with contextlib.suppress(Exception):
            subprocess.run(
                [uv_bin, "pip", "uninstall", "--quiet", "--python", str(venv_py), *COLAB_SHADOW_PACKAGES],
                capture_output=True,
                text=True,
                check=False,
            )


def ensure_venv() -> None:
    if Path("pyproject.toml").exists():
        _log("⏳ กำลังติดตั้งแพ็กเกจ Python ตาม pyproject.toml (uv sync)...")
        run_stream("set -e\nuv sync --quiet\n")
    else:
        _log("⏳ ไม่พบ pyproject.toml — สร้างโปรเจกต์ใหม่แล้วติดตั้ง dependency ชุดมาตรฐาน...")
        deps = " ".join(shlex.quote(d) for d in FALLBACK_DEPS)
        run_stream(
            "set -e\n"
            "if [ ! -f pyproject.toml ]; then uv init --vcs none --no-readme --quiet; fi\n"
            f"uv add --quiet {deps}\n"
        )
    _strip_colab_shadow_packages(Path(".venv").resolve())
    run_quiet("uv export --format requirements-txt --no-hashes --no-emit-project --quiet -o requirements.txt || true")


def activate_venv() -> str:
    venv_path = Path(".venv").resolve()
    _strip_colab_shadow_packages(venv_path)
    sites = glob.glob(str(venv_path / "lib" / "python*" / "site-packages"))
    if not sites:
        raise RuntimeError(f"ไม่พบ site-packages ใน {venv_path} — uv sync ล้มเหลว?")
    venv_site = sites[0]
    site.addsitedir(venv_site)
    while venv_site in sys.path:
        sys.path.remove(venv_site)
    if _is_colab_runtime():
        sys.path.append(venv_site)
    else:
        sys.path.insert(0, venv_site)
    os.environ["VIRTUAL_ENV"] = str(venv_path)
    os.environ["PATH"] = f"{venv_path / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    importlib.invalidate_caches()
    return venv_site


def prepare_dirs() -> None:
    for d in (STRATA_HOME, MODEL_DIR, f"{MODEL_DIR}/_drafts"):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            rprint(f"[yellow]⚠️ สร้างโฟลเดอร์ {d} ไม่ได้: {exc} — ลองตั้ง STRATA_HOME/STRATA_MODEL_DIR ใหม่[/yellow]")


def ensure_environment(skip_devshell: bool = False, nix: str = "auto") -> dict:
    plan = environment_plan(nix)
    _STEP_COUNTER.update({"i": 0, "total": len(plan["steps"]), "t0": time.perf_counter()})
    prepare_dirs()
    prepare_setup_log()
    _emit(
        f"🚀 เตรียมสภาพแวดล้อม: โหมด {'Nix devshell' if plan['use_nix'] else 'pip-only (เร็ว)'}"
        f" • แพลตฟอร์ม {plan['name']} • {len(plan['steps'])} ขั้นตอน • log: {SETUP_LOG}"
    )
    for note in plan.get("notes", []):
        _emit(f"ℹ️ {note}")

    _step("เตรียมโฟลเดอร์")
    prepare_dirs()
    _step("ตั้งค่า PATH/ตัวแปรระบบ")
    ensure_paths()

    use_nix = plan["use_nix"]
    if use_nix:
        _step("ติดตั้ง Nix (3-8 นาที ถ้ายังไม่มี)")
        write_nix_conf()
        install_nix()
        _step("ตั้งค่า shell ใหม่ให้มี nix")
        persist_shell_env()
        link_nix_binaries()
        _step("เขียน flake.nix")
        write_flake()
        if not skip_devshell:
            _step("โหลด devshell (ขั้นที่ช้าที่สุด)")
            try:
                activate_devshell()
            except KeyboardInterrupt:
                use_nix = False
                rprint(
                    "[yellow]⏹️ ยกเลิกการรอ Nix devshell — สลับไปโหมด pip-only ให้อัตโนมัติ "
                    "(ครั้งหน้าตั้ง USE_NIX=\"never\" เพื่อข้ามขั้นนี้ไปเลย)[/yellow]"
                )
            except RuntimeError:
                rprint("[bold red]❌ โหลด Nix devshell ไม่สำเร็จ — ข้อมูลดีบัก:[/bold red]")
                for key, val in nix_diagnostics().items():
                    rprint(f"   [dim]{key}[/dim] = {val}")
                raise
        if use_nix:
            _step("เชื่อมเครื่องมือเข้า PATH")
            link_tools()
            missing = missing_tools()
            if missing:
                rprint(
                    f"[yellow]⚠️ ไม่พบเครื่องมือใน PATH: {', '.join(missing)} — ถ้าเพิ่งรันเซลล์ 1 ไม่จบ "
                    "ให้รันใหม่ หรือโหลดโมเดลจะพังด้วย `command not found`[/yellow]"
                )
            _step("สร้าง venv + ติดตั้ง dependency")
            ensure_venv()
            venv_site = activate_venv()
            mode = "nix"
    if not use_nix:
        _step("ติดตั้งแพ็กเกจ Python ด้วย pip")
        pkgs = [*PIP_ONLY_DEPS]
        if not _is_colab_runtime():
            pkgs.append("ipywidgets>=8.1,<9")
        _pip_install_deps(pkgs)
        venv_site = _current_site_packages()
        missing = [t for t in ("llama-server",) if not shutil.which(t)]
        mode = "pip"

    _emit(f"✅ สภาพแวดล้อมพร้อม (โหมด {mode} • {venv_site} • {time.perf_counter() - _STEP_COUNTER['t0']:.0f}s)")
    return {
        "mode": mode,
        "platform": plan["name"],
        "steps": plan["steps"],
        "nixpkgs": NIXPKGS_REV if plan["use_nix"] else "",
        "venv": venv_site,
        "model_dir": MODEL_DIR,
        "flake_dir": str(FLAKE_DIR),
        "missing_tools": missing,
        "nix_in_path": bool(shutil.which("nix")),
    }


def _current_site_packages() -> str:
    import site as _site

    paths = list(_site.getsitepackages() if hasattr(_site, "getsitepackages") else [])
    for p in sys.path:
        if p and p.endswith("site-packages") and Path(p).is_absolute():
            paths.append(p)
    default_sp = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    return paths[0] if paths else str(default_sp)


def _pip_freeze_pins() -> dict[str, str]:
    """เวอร์ชันของทุกแพ็กเกจที่ติดตั้งอยู่ตอนนี้ (ชื่อ → เวอร์ชัน) เอาไปทำ constraints"""
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True, text=True, check=False,
    ).stdout
    pins: dict[str, str] = {}
    for ln in out.splitlines():
        ln = ln.strip()
        if "==" not in ln or ln.startswith(("-e", "#", "pip==")):
            continue
        name, _, ver = ln.partition("==")
        if not name or not ver or "@" in ver or " " in name or name.startswith("-"):
            continue
        pins[name] = ver
    return pins


def pip_constraints_file() -> Path | None:
    """สร้างไฟล์ constraints จากเวอร์ชันที่ติดตั้งอยู่ เพื่อไม่ให้ pip อัปเกรดของที่แพลตฟอร์มผูกมา

    เคสจริงบน Kaggle: ``huggingface-hub`` ดึง ``fsspec`` ใหม่ → ``datasets`` พัง
    (fsspec<=2026.2.0) และ ``fastembed`` ดึง ``pillow`` 13 → ``gradio`` พัง (pillow<13)
    """
    pins = _pip_freeze_pins()
    if not pins:
        return None
    path = Path(tempfile.gettempdir()) / "strata-pip-constraints.txt"
    with contextlib.suppress(OSError):
        path.write_text("".join(f"{n}=={v}\n" for n, v in sorted(pins.items())), encoding="utf-8")
        return path
    return None


def _pip_install_deps(dep_list: list[str] | None = None) -> None:
    use_deps = dep_list if dep_list is not None else PIP_ONLY_DEPS
    base = f"{shlex.quote(sys.executable)} -m pip install -q"
    deps = " ".join(shlex.quote(d) for d in use_deps)
    names = " ".join(d.split(">")[0].split("=")[0].split("[")[0] for d in use_deps)
    rprint(f"[dim]📦 pip install: {names}[/dim]")
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "hf_transfer"],
            capture_output=True, text=True, check=False,
        )
    constraints = pip_constraints_file()
    attempts: list[tuple[str, str]] = []
    if constraints is not None:
        attempts.append((f"{base} -c {shlex.quote(str(constraints))} {deps}", "ล็อกเวอร์ชันที่ติดตั้งอยู่"))
    attempts.append((f"{base} {deps}", "ไม่ล็อกเวอร์ชัน"))
    for cmd, label in attempts:
        try:
            run_stream(f"set -e\n{base} --upgrade pip\n{cmd}")
            break
        except subprocess.CalledProcessError:
            rprint(f"[yellow]⚠️ pip install ไม่ผ่าน ({label}) — ลองวิธีถัดไป[/yellow]")
    else:
        rprint("[yellow]⚠️ pip ปกติไม่ผ่าน (อาจเป็น externally-managed) — ลองใหม่ด้วย --break-system-packages[/yellow]")
        run_stream(f"set -e\n{base} --break-system-packages {deps}")
    importlib.invalidate_caches()


def _llama_binary_hint() -> None:
    if not shutil.which("llama-server"):
        rprint("[dim]ℹ️ ยังไม่มี llama-server ใน PATH — จะถูกดาวน์โหลดอัตโนมัติตอนเปิด Studio (prebuilt CUDA)[/dim]")
