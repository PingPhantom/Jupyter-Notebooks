"""ตัวช่วยสำหรับใช้ใน Colab/Jupyter: ลงทะเบียน magic และเปิด Studio UI 3 แท็บ (พร้อม HTML5 Mobile/Desktop Touch UI บน Colab)"""

from __future__ import annotations

import base64
import contextlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from IPython.core.getipython import get_ipython
from IPython.core.magic import register_cell_magic
from IPython.display import HTML, JSON, display
from rich.console import Console
from rich.syntax import Syntax


def register_magics() -> None:
    """ลงทะเบียน `%%ruff` — ฟอร์แมต/จัดเรียงโค้ดในเซลล์ก่อนรัน (ข้ามการรันด้วย `--no-run`)"""

    @register_cell_magic
    def ruff(line, cell):  # noqa: ANN001
        no_run = "--no-run" in line.split()
        config = PyprojectConfig.path()
        chk = subprocess.run(
            ["ruff", "check", "--fix", "--quiet", "--config", config, "--stdin-filename", "cell.py", "-"],
            input=cell,
            text=True,
            capture_output=True,
            check=False,
        )
        fixed = chk.stdout if chk.returncode in (0, 1) and chk.stdout else cell
        fmt = subprocess.run(
            ["ruff", "format", "--quiet", "--config", config, "--stdin-filename", "cell.py", "-"],
            input=fixed,
            text=True,
            capture_output=True,
            check=False,
        )
        code = fmt.stdout if fmt.returncode == 0 else fixed
        if code.strip() != cell.strip() or no_run:
            Console().print(Syntax(code.strip(), "python", theme="monokai", line_numbers=True))
        if not no_run:
            get_ipython().run_cell(code)


class PyprojectConfig:
    @staticmethod
    def path() -> str:
        for base in (Path.cwd(), *Path.cwd().parents):
            candidate = base / "pyproject.toml"
            if candidate.is_file():
                return str(candidate)
        return str(Path.cwd() / "pyproject.toml")


def _ver(module: str) -> str:
    loaded = sys.modules.get(module)
    if loaded is not None and getattr(loaded, "__version__", None):
        return str(loaded.__version__)
    with contextlib.suppress(importlib.metadata.PackageNotFoundError, ValueError):
        return importlib.metadata.version(module)
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001
        return "-"
    return str(getattr(mod, "__version__", "?"))


def widget_versions() -> dict:
    return {
        "ipywidgets": _ver("ipywidgets"),
        "jupyterlab_widgets": _ver("jupyterlab_widgets"),
        "widgetsnbextension": _ver("widgetsnbextension"),
    }


def widget_plan(versions: dict, platform: str | None = None) -> list[str]:
    iw = str(versions.get("ipywidgets", "-"))
    if platform == "colab":
        if not iw or iw == "-":
            return ["ipywidgets>=7.7,<8"]
        return []
    if not iw or iw == "-":
        return ["ipywidgets>=8.1,<9", "jupyterlab_widgets>=3,<4"]
    major = iw.split(".")[0]
    want = {
        "7": ("jupyterlab_widgets>=2,<3", "widgetsnbextension>=3,<4"),
        "8": ("jupyterlab_widgets>=3,<4", "widgetsnbextension>=4,<5"),
    }.get(major)
    if want is None:
        return ["ipywidgets", "jupyterlab_widgets"]
    plan = []
    jlw = str(versions.get("jupyterlab_widgets", "-"))
    if jlw == "-" or jlw.split(".")[0] != want[0].split(">=")[1].split(",")[0].split(".")[0]:
        plan.append(want[0])
    wsnb = str(versions.get("widgetsnbextension", "-"))
    if wsnb == "-" or wsnb.split(".")[0] != want[1].split(">=")[1].split(",")[0].split(".")[0]:
        plan.append(want[1])
    return plan


def _enable_colab_widgets_if_needed(versions: dict, platform: str) -> None:
    if platform != "colab":
        return
    major = str(versions.get("ipywidgets", "-")).split(".")[0]
    if major.isdigit() and int(major) >= 8:
        with contextlib.suppress(Exception):
            from google.colab import output as colab_output  # type: ignore[import-not-found]

            colab_output.enable_custom_widget_manager()


def _install_widget_pkgs(plan: list[str]) -> bool:
    venv = os.environ.get("VIRTUAL_ENV", "")
    uv_bin = shutil.which("uv")
    venv_py = Path(venv) / "bin" / "python" if venv else None
    if venv_py and venv_py.is_file() and uv_bin:
        res = subprocess.run(
            [uv_bin, "pip", "install", "--quiet", "--python", str(venv_py), *plan],
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            return True
    res = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *plan],
        check=False,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "--upgrade", *plan],
            check=False,
            capture_output=True,
            text=True,
        )
    return res.returncode == 0


def ensure_widgets(install: bool = True, platform: str | None = None) -> dict:
    if platform is None:
        from .platforms import detect

        platform = detect()["name"]
    before = widget_versions()
    _enable_colab_widgets_if_needed(before, platform)
    plan = widget_plan(before, platform=platform)
    if plan and install:
        rprint = _rprint()
        rprint(f"[yellow]🧩 วิดเจ็ตอาจไม่แสดงผล: {before} → ติดตั้ง {', '.join(plan)}[/yellow]")
        for mod_name in ("jupyterlab_widgets", "widgetsnbextension"):
            sys.modules.pop(mod_name, None)
        ok = _install_widget_pkgs(plan)
        importlib.invalidate_caches()
        if not ok:
            pkgs = " ".join(f'"{p}"' for p in plan)
            rprint(f"[yellow]⚠️ ติดตั้งวิดเจ็ตอัตโนมัติไม่สำเร็จ — ลองรัน: %pip install -q {pkgs}[/yellow]")
    after = widget_versions()
    _enable_colab_widgets_if_needed(after, platform)
    return {"before": before, "plan": plan, "after": after}


def _rprint():
    from rich import print as rprint

    return rprint


def diagnose(verbose: bool = True) -> dict:
    from .config import DEFAULT_PORT, MODEL_DIR, STRATA_HOME
    from .platforms import detect

    info = detect()
    plat = info["name"]
    versions = widget_versions()
    plan = widget_plan(versions, platform=plat)
    disk = {}
    try:
        base = Path(MODEL_DIR) if Path(MODEL_DIR).exists() else Path(info["home"])
        du = shutil.disk_usage(base if base.exists() else Path.home())
        disk = {"path": str(base), "free_gb": round(du.free / 1024**3, 2), "total_gb": round(du.total / 1024**3, 2)}
    except OSError:
        disk = {"path": str(MODEL_DIR), "free_gb": 0.0, "total_gb": 0.0}
    out = {
        "platform": info,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "strata_home": STRATA_HOME,
        "model_dir": MODEL_DIR,
        "port": DEFAULT_PORT,
        "widgets": versions,
        "widget_plan": plan,
        "disk": disk,
        "is_notebook": get_ipython() is not None,
    }
    if verbose:
        from rich.table import Table

        colab_note = " (Colab จัดการเอง)" if plat == "colab" else ""
        iw_note = " (Colab native)" if plat == "colab" and versions["ipywidgets"].startswith("7.") else ""
        t = Table(title="strata diagnose", show_lines=False)
        t.add_column("รายการ", style="cyan")
        t.add_column("ค่า")
        t.add_row("แพลตฟอร์ม", info["label"])
        t.add_row("python", out["python"])
        t.add_row("STRATA_HOME", out["strata_home"])
        t.add_row("MODEL_DIR", out["model_dir"])
        t.add_row("พอร์ต", str(out["port"]))
        t.add_row("ipywidgets", f"{versions['ipywidgets']}{iw_note}")
        t.add_row("jupyterlab_widgets", f"{versions['jupyterlab_widgets']}{colab_note}")
        t.add_row("widgetsnbextension", f"{versions['widgetsnbextension']}{colab_note}")
        t.add_row("ดิสก์ว่าง", f"{disk['free_gb']} / {disk['total_gb']} GB ({disk['path']})")
        t.add_row("อยู่ใน notebook", "ใช่" if out["is_notebook"] else "ไม่")
        Console().print(t)
    return out


def text_dashboard(model_dir: str | None = None) -> None:
    from rich.table import Table

    from .config import IMAGE_PRESET_KEYS, MODEL_CATALOG, MODEL_DIR
    from .utils import _dir_bytes, _fmt_bytes

    base = Path(model_dir or MODEL_DIR)
    con = Console()
    t = Table(title="📦 โมเดลในแคตตาล็อก (โหมดข้อความ)")
    t.add_column("#", justify="right")
    t.add_column("key", style="cyan")
    t.add_column("ขนาด", justify="right")
    t.add_column("สถานะ")
    for i, (key, cfg) in enumerate(MODEL_CATALOG.items(), 1):
        if key == "custom":
            continue
        sub = base / key
        have = _dir_bytes(sub) if sub.is_dir() else 0
        kind = "🎨" if key in IMAGE_PRESET_KEYS else "💬"
        t.add_row(
            str(i),
            f"{kind} {key}",
            f"~{cfg['approx_gb']:.1f} GB",
            f"✅ {_fmt_bytes(have)}" if have else "⬜ ยังไม่โหลด",
        )
    con.print(t)


# ---------------------------------------------------------------------------
# HTML5 Responsive Touch Studio 3 แท็บ สำหรับ Google Colab (รองรับทั้งมือถือ iOS/Android & Desktop 100%)
# เก็บเป็นสตริงธรรมดา (ไม่ใช้ f-string) เพื่อไม่ให้วงเล็บปีกกา {} ของ CSS/JS ชนกับ Python เด็ดขาด
# ---------------------------------------------------------------------------
_TOUCH_STATE: dict = {
    "job": "",
    "notice": "",
    "notice_kind": "info",
    "img_status": "",
    "last_image": None,
    "history": [],
}

_TOUCH_HTML_TEMPLATE = """
<div id="st-app">
<style>
  #st-app { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; background: #0b1120; border: 1px solid #334155; border-radius: 14px; padding: 12px; max-width: 100%; box-sizing: border-box; }
  #st-app * { box-sizing: border-box; touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
  .st-hdr { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
  .st-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; font-size: 12px; }
  .st-bdg { padding: 3px 9px; border-radius: 99px; background: #1e293b; border: 1px solid #334155; }
  .st-btn { min-height: 42px; padding: 8px 13px; border-radius: 9px; border: 1px solid #475569; background: #1e293b; color: #f8fafc; font-size: 13.5px; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 5px; user-select: none; }
  .st-btn:active { transform: scale(0.97); opacity: 0.9; }
  .st-btn.pri { background: #2563eb; border-color: #3b82f6; }
  .st-btn.suc { background: #059669; border-color: #10b981; }
  .st-btn.wrn { background: #d97706; border-color: #f59e0b; }
  .st-btn.dng { background: #dc2626; border-color: #ef4444; }
  .st-btn.act { background: #7c3aed; border-color: #a78bfa; }
  .st-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 10px; }
  .st-inp, .st-sel, .st-ta { width: 100%; min-height: 42px; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 9px; padding: 8px 10px; font-size: 14px; }
  .st-row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
  .st-card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px; margin-bottom: 8px; }
  .st-mgrid { display: flex; flex-direction: column; gap: 8px; max-height: 540px; overflow-y: auto; padding-right: 2px; }
  .st-mrow { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
  @media (min-width: 720px) { .st-mrow { flex-direction: row; justify-content: space-between; align-items: center; } }
  .st-macts { display: flex; flex-wrap: wrap; gap: 6px; }
  .st-tag { display: inline-block; padding: 1px 6px; border-radius: 6px; font-size: 10.5px; margin-left: 4px; }
  .st-feed { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px; min-height: 240px; max-height: 440px; overflow-y: auto; margin-bottom: 8px; display: flex; flex-direction: column; gap: 8px; }
  .st-msg-u { align-self: flex-end; background: #1d4ed8; color: #fff; padding: 8px 12px; border-radius: 12px; max-width: 88%; font-size: 14px; white-space: pre-wrap; }
  .st-msg-b { align-self: flex-start; background: #1e293b; border: 1px solid #334155; padding: 9px 12px; border-radius: 12px; max-width: 94%; font-size: 14px; line-height: 1.55; white-space: pre-wrap; }
  .st-log { background: #060b14; color: #94a3b8; font-family: ui-monospace, monospace; font-size: 11px; padding: 8px 10px; border-radius: 8px; border: 1px solid #1e293b; max-height: 150px; overflow-y: auto; margin-top: 8px; }
  .st-hide { display: none !important; }
</style>

  <div class="st-hdr">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
      <b style="font-size:15px;">📱🎛️ Strata Touch Studio (<span id="st-plat"></span> • __TOTAL_MODELS__ Models • 3 แท็บ)</b>
      <div style="display:flex;gap:6px;">
        <button class="st-btn dng" id="st-cancel-btn" onclick="stCall('cancel')">⏹️ ยกเลิกงาน</button>
        <button class="st-btn" onclick="stPoll(true)">🔄 รีเฟรช</button>
      </div>
    </div>
    <div class="st-badges" id="st-badges"></div>
    <div style="margin-top:6px;background:#1e293b;height:7px;border-radius:99px;overflow:hidden;"><div id="st-disk-bar" style="height:100%;background:#22c55e;width:10%;"></div></div>
  </div>

  <div id="st-prog-box" class="st-card st-hide" style="background:#0c4a6e;border-color:#38bdf8;"></div>
  <div id="st-notice-box" class="st-card st-hide"></div>

  <!-- แถบเร่งสปีด -->
  <details class="st-card" open>
    <summary style="cursor:pointer;font-weight:700;color:#e9d5ff;font-size:13.5px;">⚡ ตั้งค่าเร่งสปีด (EAGLE-3 / Draft / MTP / MoE-CPU)</summary>
    <div class="st-row" style="margin-top:8px;" id="st-spec-modes"></div>
    <div class="st-row">
      <label style="font-size:12.5px;"><input type="checkbox" id="st-kvq8" onchange="stSaveSpec(false)"> 💾 KV Q8_0</label>
      <label style="font-size:12.5px;margin-left:8px;"><input type="checkbox" id="st-moe" onchange="stSaveSpec(false)"> 🧠 MoE-CPU Auto</label>
      <span style="font-size:12.5px;margin-left:8px;">Max: <input type="number" id="st-dmax" min="0" max="24" value="0" style="width:62px;padding:4px;border-radius:6px;background:#1e293b;color:#fff;border:1px solid #475569;" onchange="stSaveSpec(false)"></span>
      <button class="st-btn pri" onclick="stSaveSpec(true)">🔄 รีสตาร์ทใช้โหมดนี้</button>
    </div>
    <div class="st-row">
      <select id="st-draft-dd" class="st-sel" style="flex:1 1 220px;"></select>
      <button class="st-btn pri" onclick="stDlDraft()">📥 โหลด & ใช้ Draft</button>
    </div>
  </details>

  <!-- 3 แท็บหลักขนาดใหญ่กดง่ายบนมือถือและคอม -->
  <div class="st-tabs">
    <button class="st-btn act" id="st-tbtn-0" onclick="stTab(0)">📦 1. โมเดล (__TOTAL_MODELS__)</button>
    <button class="st-btn" id="st-tbtn-1" onclick="stTab(1)">💬 2. แชท + RP + ภาพ</button>
    <button class="st-btn" id="st-tbtn-2" onclick="stTab(2)">📚 3. คลัง PDF RAG</button>
  </div>

  <!-- TAB 0: Models -->
  <div id="st-tab-0">
    <div class="st-row">
      <input id="st-search" class="st-inp" style="flex:1 1 180px;" placeholder="🔍 ค้นหา เช่น obliterated, skyfall, anubis, magnum, nevoria, 70b..." oninput="stRenderModels()">
      <select id="st-filter" class="st-sel" style="flex:0 1 215px;" onchange="stRenderModels()">
        <option value="all">📋 ทั้งหมด (__TOTAL_MODELS__ รุ่น)</option>
        <option value="installed">✅ ติดตั้งแล้ว</option>
        <option value="fit">🎮 พอดี VRAM การ์ดนี้</option>
        <option value="obliterated">💥 สาย OBLITERATED</option>
        <option value="unrestricted">🔓 สาย Unrestricted</option>
        <option value="rp">🎭 สาย Roleplay (RP / ERP)</option>
        <option value="heretic">🔥 สาย Heretic / Uncensored</option>
        <option value="lora">🔥🧩 โมเดลเสริม Uncensored LoRA (6)</option>
        <option value="image">🎨 สร้างภาพ GGUF (__TOTAL_IMG__ รุ่น)</option>
        <option value="code">💻 สาย Code / Security</option>
      </select>
    </div>
    <div class="st-row">
      <button class="st-btn wrn" onclick="stModelOp('stop_server','')">🛑 หยุด Server</button>
      <button class="st-btn" onclick="stModelOp('clean_cache','')">🧹 ล้างแคช</button>
      <button class="st-btn dng" onclick="if(confirm('ยืนยันลบโมเดลทั้งหมดในเครื่อง?')) stModelOp('delete_all','')">🗑️ ลบทั้งหมด</button>
    </div>
    <div class="st-mgrid" id="st-mlist"></div>
    <div class="st-log" id="st-log"></div>
  </div>

  <!-- TAB 1: Chat + RP + Image -->
  <div id="st-tab-1" class="st-hide">
    <div class="st-row">
      <select id="st-tpl" class="st-sel" style="flex:1 1 210px;" onchange="stApplyTpl()"></select>
      <select id="st-qsw" class="st-sel" style="flex:1 1 170px;"></select>
      <button class="st-btn wrn" onclick="stQuickSwitch()">⚡ สลับโมเดล</button>
    </div>
    <textarea id="st-sys" class="st-ta" rows="2" style="margin-bottom:8px;" placeholder="System Prompt / บทบาทตัวละคร..."></textarea>
    <details class="st-card">
      <summary style="cursor:pointer;font-size:13px;color:#93c5fd;">🎛️ ปรับค่า Sampler สาย RP (Temp / Min-P / DRY / MaxTok)</summary>
      <div class="st-row" style="margin-top:8px;font-size:12.5px;">
        <label><input type="checkbox" id="st-userag" checked> 📚 ใช้ RAG</label>
        <span>Temp: <input id="st-temp" type="number" step="0.05" min="0" max="2" value="0.7" style="width:65px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>Min-P: <input id="st-minp" type="number" step="0.01" min="0" max="0.5" value="0.05" style="width:65px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>RepPen: <input id="st-reppen" type="number" step="0.02" min="1" max="1.5" value="1.08" style="width:65px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>DRY: <input id="st-dry" type="number" step="0.1" min="0" max="2" value="0" style="width:60px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>MaxTok: <input id="st-maxtok" type="number" step="64" min="128" max="4096" value="1024" style="width:78px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
      </div>
    </details>
    <div class="st-feed" id="st-feed"><div style="color:#94a3b8;text-align:center;padding:12px;">💬 เลือกโหมดแชท/Roleplay หรือสั่งวาดภาพด้านล่างได้เลย</div></div>
    <textarea id="st-uin" class="st-ta" rows="3" placeholder="พิมพ์ข้อความแชท / บทบาท RP หรือคำบรรยายภาพ..."></textarea>
    <div class="st-row" style="margin-top:6px;">
      <button class="st-btn suc" style="flex:2 1 130px;" id="st-send-btn" onclick="stSendChat(false)">📨 ส่งข้อความ</button>
      <button class="st-btn pri" style="flex:1 1 110px;" onclick="stSendChat(true)">🔁 สุ่มตอบใหม่</button>
      <button class="st-btn dng" onclick="stClearChat()">🗑️ ล้าง</button>
    </div>

    <!-- กล่องสร้างภาพสำหรับมือถือโดยตรง (ไม่ง้อ Iframe/คุกกี้) -->
    <div class="st-card" style="margin-top:10px;border-color:#f59e0b;">
      <div style="font-weight:700;color:#fbbf24;margin-bottom:6px;font-size:13.5px;">🎨 สตูดิโอสร้างภาพ GGUF (ใช้งานบนมือถือได้ทันที ไม่ต้องผ่าน Iframe)</div>
      <div class="st-row">
        <select id="st-img-key" class="st-sel" style="flex:1 1 200px;"></select>
        <select id="st-img-sz" class="st-sel" style="flex:1 1 165px;" onchange="stPresetWH()">
          <option value="1024x1024">📐 1:1 (1024×1024)</option>
          <option value="832x1216">📱 2:3 แนวตั้ง (832×1216)</option>
          <option value="1216x832">🖼️ 3:2 แนวนอน (1216×832)</option>
          <option value="768x1344">📲 9:16 มือถือ (768×1344)</option>
          <option value="1344x768">🎬 16:9 จอกว้าง (1344×768)</option>
          <option value="768x768">⚡ 1:1 เร็ว (768×768)</option>
        </select>
      </div>
      <div class="st-row" style="font-size:12.5px;">
        <span>กว้าง: <input id="st-iw" type="number" step="64" min="512" max="1536" value="1024" style="width:70px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>สูง: <input id="st-ih" type="number" step="64" min="512" max="1536" value="1024" style="width:70px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>Steps(0=Auto): <input id="st-isteps" type="number" min="0" max="50" value="0" style="width:58px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <span>Seed(-1=สุ่ม): <input id="st-iseed" type="number" value="-1" style="width:75px;padding:4px;background:#1e293b;color:#fff;border:1px solid #475569;border-radius:6px;"></span>
        <label><input type="checkbox" id="st-ifast" checked> ⚡ FastCache</label>
      </div>
      <input id="st-ineg" class="st-inp" style="margin-bottom:6px;" placeholder="Negative Prompt (เช่น blurry, low quality, deformed, watermark)">
      <div class="st-row">
        <button class="st-btn wrn" style="flex:2 1 160px;" onclick="stGenImg()">🖼️ สร้างภาพจากข้อความช่องบน</button>
        <button class="st-btn" onclick="stLoadGallery()">🖼️ รูปที่สร้างไว้</button>
      </div>
      <div id="st-img-stat" style="font-size:12.5px;color:#38bdf8;margin-top:4px;"></div>
    </div>
  </div>

  <!-- TAB 2: PDF RAG -->
  <div id="st-tab-2" class="st-hide">
    <div class="st-card">
      <div style="font-weight:700;color:#38bdf8;margin-bottom:6px;">📤 อัปโหลดเอกสารจากมือถือ/คอม (.pdf, .txt, .md, .csv, .py)</div>
      <div class="st-row">
        <input type="file" id="st-files" multiple accept=".pdf,.txt,.md,.csv,.json,.py" class="st-inp" style="flex:1 1 200px;">
        <button class="st-btn suc" onclick="stUploadRag()">📥 นำไฟล์เข้า RAG</button>
        <button class="st-btn dng" onclick="stRagOp('clear',{})">🗑️ ล้างคลัง</button>
      </div>
      <textarea id="st-rag-txt" class="st-ta" rows="3" placeholder="หรือวางเนื้อหา/ข้อมูลตัวละคร/บทความที่นี่..."></textarea>
      <div class="st-row" style="margin-top:6px;">
        <input id="st-rag-src" class="st-inp" style="flex:0 1 150px;" value="note" placeholder="ชื่อแหล่งที่มา">
        <button class="st-btn pri" onclick="stAddRagText()">➕ เพิ่มข้อความเข้า RAG</button>
      </div>
      <div class="st-row">
        <input id="st-rag-q" class="st-inp" style="flex:1 1 180px;" placeholder="พิมพ์คำค้นทดสอบในคลัง RAG...">
        <button class="st-btn" onclick="stSearchRag()">🔍 ทดสอบค้นหา</button>
      </div>
      <div class="st-row">
        <input id="st-rag-path" class="st-inp" style="flex:1 1 200px;">
        <button class="st-btn" onclick="stRagOp('save',{path:document.getElementById('st-rag-path').value})">💾 บันทึก</button>
        <button class="st-btn" onclick="stRagOp('load',{path:document.getElementById('st-rag-path').value})">📂 โหลดเดิม</button>
      </div>
      <div id="st-rag-out" style="font-size:13px;margin-top:6px;color:#cbd5e1;"></div>
    </div>
  </div>
</div>

<script>
(function(){
  const INIT = __INIT_JSON__;
  let S = INIT.state;
  const TPLS = INIT.templates;
  let lastImgShown = '';

  function esc(s){ const d=document.createElement('div'); d.textContent=String(s||''); return d.innerHTML; }

  async function rpc(action, payload){
    if(typeof google !== 'undefined' && google.colab && google.colab.kernel){
      const raw = await google.colab.kernel.invokeFunction('strata.rpc', [action, payload || {}], {});
      return (raw && raw.data && raw.data['application/json']) ? raw.data['application/json'] : raw;
    }
    return null;
  }

  window.stCall = async function(act, p){ const r = await rpc(act, p); if(r && r.models) { S = r; renderTop(); stRenderModels(); } return r; };

  window.stTab = function(idx){
    for(let i=0;i<3;i++){
      document.getElementById('st-tab-'+i).classList.toggle('st-hide', i!==idx);
      document.getElementById('st-tbtn-'+i).classList.toggle('act', i===idx);
    }
  };

  function renderSpecButtons(){
    const specMode = (S.spec && S.spec.mode) || 'auto';
    const modes = [['off','⏹️ ปิด'],['auto','✨ Auto'],['eagle3','🦅 EAGLE-3'],['mtp','🚀 MTP'],['draft','⚡ Draft'],['ngram','📚 N-gram+']];
    document.getElementById('st-spec-modes').innerHTML = modes.map(([k,lbl]) =>
      `<button class="st-btn ${specMode===k?'act':''}" onclick="stPickSpec('${k}')">${lbl}</button>`
    ).join('');
    document.getElementById('st-kvq8').checked = !!(S.spec && S.spec.kv_q8);
    document.getElementById('st-moe').checked = !!(S.spec && S.spec.moe_cpu_auto);
    document.getElementById('st-dmax').value = (S.spec && S.spec.draft_max) || 0;
    const dd = document.getElementById('st-draft-dd');
    const prev = dd.value || 'llama32-1b-uncensored';
    dd.innerHTML = (S.drafts || []).map(d => `<option value="${esc(d.key)}">${esc(d.label)}</option>`).join('');
    if([...dd.options].some(o=>o.value===prev)) dd.value = prev;
  }

  window.stPickSpec = function(m){ if(!S.spec) S.spec = {}; S.spec.mode = m; renderSpecButtons(); stSaveSpec(false); };
  window.stSaveSpec = function(restart){
    stCall('set_spec', {
      mode: (S.spec && S.spec.mode) || 'auto',
      draft_max: parseInt(document.getElementById('st-dmax').value||'0',10),
      kv_q8: document.getElementById('st-kvq8').checked,
      moe_cpu_auto: document.getElementById('st-moe').checked,
      restart: !!restart
    });
  };
  window.stDlDraft = function(){ stCall('dl_draft', {key: document.getElementById('st-draft-dd').value}); };

  function renderTop(){
    document.getElementById('st-plat').textContent = S.platform + ' • :' + S.port;
    const onBdg = S.online
      ? `<span class="st-bdg" style="background:#14532d;color:#4ade80;">🟢 ${esc(S.active_model)}</span>`
      : `<span class="st-bdg" style="background:#450a0a;color:#fca5a5;">🔴 Offline</span>`;
    const spBdg = (S.online && S.spec_label) ? `<span class="st-bdg" style="background:#3b0764;color:#e9d5ff;">${esc(S.spec_label)}</span>` : '';
    document.getElementById('st-badges').innerHTML =
      `${onBdg} ${spBdg} <span class="st-bdg">📚 RAG: ${S.rag_chunks}</span> <span class="st-bdg">${esc(S.gpu)}</span> <span class="st-bdg">💾 ว่าง ${S.disk.free_gb}/${S.disk.total_gb} GB</span> <span class="st-bdg">${esc(S.net)}</span>`;
    document.getElementById('st-disk-bar').style.width = S.disk.pct + '%';

    const pb = document.getElementById('st-prog-box');
    if(S.progress){
      pb.classList.remove('st-hide');
      pb.innerHTML = `<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;font-size:12.5px;margin-bottom:6px;"><span>📥 <b>Fetching [${esc(S.progress.files)}]</b> • ${esc(S.progress.file || S.progress.repo)}</span><span><b>${S.progress.pct.toFixed(1)}%</b> (${esc(S.progress.done)} / ${esc(S.progress.total)}) • <b style="color:#fde047;">⬇ ${esc(S.progress.rate)}</b> • ETA ${esc(S.progress.eta)}</span></div><div style="background:#082f49;height:10px;border-radius:99px;overflow:hidden;"><div style="width:${Math.max(2,S.progress.pct)}%;height:100%;background:linear-gradient(90deg,#38bdf8,#34d399);"></div></div>`;
    } else if(S.job) {
      pb.classList.remove('st-hide');
      pb.innerHTML = `<div style="font-size:13px;">⏳ กำลังทำงาน: <b>${esc(S.job)}</b> (แตะ ⏹️ ยกเลิกงาน เพื่อหยุดได้ทันที)</div>`;
    } else {
      pb.classList.add('st-hide');
    }

    const nb = document.getElementById('st-notice-box');
    if(S.notice){
      nb.classList.remove('st-hide');
      const bg = S.notice_kind==='ok'?'#064e3b':(S.notice_kind==='err'?'#450a0a':(S.notice_kind==='warn'?'#422006':'#0f172a'));
      nb.style.background = bg;
      nb.innerHTML = `<div style="font-size:12.5px;">${esc(S.notice)}</div>`;
    } else {
      nb.classList.add('st-hide');
    }

    if(S.img_status) document.getElementById('st-img-stat').textContent = S.img_status;
    if(S.last_image && S.last_image.name !== lastImgShown){
      lastImgShown = S.last_image.name;
      appendImgCard(S.last_image);
    }
    document.getElementById('st-log').innerHTML = (S.logs||[]).map(l=>`<div>${esc(l)}</div>`).join('') || '<div>พร้อมรับคำสั่งบนมือถือและคอมพิวเตอร์</div>';
    renderSpecButtons();
  }

  window.stRenderModels = function(){
    const q = (document.getElementById('st-search').value||'').toLowerCase().trim();
    const f = document.getElementById('st-filter').value;
    const specMode = (S.spec && S.spec.mode) || 'auto';
    const list = (S.models || []).filter(m => {
      m.tags = Array.isArray(m.tags) ? m.tags : [];
      if(q && !(m.key.toLowerCase().includes(q) || m.desc.toLowerCase().includes(q) || m.repo.toLowerCase().includes(q))) return false;
      if(f==='installed') return m.installed;
      if(f==='fit') return m.fit==='ok' || m.fit==='warn';
      if(f==='obliterated') return m.tags.includes('obliterated');
      if(f==='unrestricted') return m.tags.includes('unrestricted');
      if(f==='rp') return m.tags.includes('rp');
      if(f==='heretic') return m.tags.includes('heretic');
      if(f==='lora') return m.kind==='lora';
      if(f==='image') return m.kind==='image';
      if(f==='code') return m.tags.includes('code') || m.tags.includes('security');
      return true;
    });
    document.getElementById('st-mlist').innerHTML = list.map(m => {
      const fitTag = m.fit==='ok' ? `<span class="st-tag" style="background:#052e16;color:#4ade80;">🎮 พอดี VRAM</span>`
        : (m.fit==='warn' ? `<span class="st-tag" style="background:#422006;color:#fbbf24;">⚠️ เกือบเต็ม</span>`
        : (m.fit==='over' ? `<span class="st-tag" style="background:#450a0a;color:#fca5a5;">🧠 Offload RAM</span>` : ''));
      const obTag = m.tags.includes('obliterated') ? `<span class="st-tag" style="background:#7c2d12;color:#fed7aa;">💥 OBLITERATED</span>` : '';
      const unTag = m.tags.includes('unrestricted') ? `<span class="st-tag" style="background:#064e3b;color:#6ee7b7;">🔓 UNRESTRICTED</span>` : '';
      const rpTag = m.tags.includes('rp') ? `<span class="st-tag" style="background:#3b0764;color:#f5d0fe;">🎭 RP</span>` : '';
      const hTag = m.tags.includes('heretic') ? `<span class="st-tag" style="background:#4c0519;color:#fda4af;">🔥 HERETIC</span>` : '';
      const lrTag = m.kind==='lora' ? `<span class="st-tag" style="background:#7f1d1d;color:#fde047;font-weight:700;">🔥🧩 UNCENSORED LORA</span>` : '';
      const imTag = m.kind==='image' ? `<span class="st-tag" style="background:#0c4a6e;color:#7dd3fc;">🎨 IMAGE</span>` : '';
      const stTxt = m.installed ? `<span style="color:#4ade80;font-size:12px;">✅ ติดตั้งแล้ว (${esc(m.size_txt)})</span>` : `<span style="color:#94a3b8;font-size:12px;">⬜ ${esc(m.size_txt)}</span>`;
      const runBtn = m.kind==='image'
        ? `<button class="st-btn suc" onclick="stJumpImg('${m.key}')">🎨 ไปวาดภาพ</button>`
        : (m.kind==='lora'
          ? `<button class="st-btn wrn" onclick="stJumpImg('${m.key}')">🔥 ใช้ LoRA นี้</button>`
          : `<button class="st-btn suc" onclick="stRunChatModel('${m.key}')">🚀 รัน & แชท</button>`);
      return `<div class="st-mrow">
        <div style="flex:1 1 auto;">
          <div><b style="color:#38bdf8;font-size:14px;">${esc(m.key)}</b> ${lrTag}${obTag}${unTag}${rpTag}${hTag}${imTag} ${stTxt} ${fitTag}</div>
          <div style="font-size:12px;color:#cbd5e1;margin-top:3px;">${esc(m.desc)} <span style="color:#64748b;">(${esc(m.repo)})</span></div>
        </div>
        <div class="st-macts">
          <button class="st-btn ${m.installed?'':'pri'}" onclick="stModelOp('download','${m.key}')">📥 ${m.installed?'โหลดซ้ำ':'โหลด'}</button>
          ${runBtn}
          ${m.can_del ? `<button class="st-btn dng" onclick="stModelOp('delete','${m.key}')">🗑️ ลบ</button>` : ''}
        </div>
      </div>`;
    }).join('') || '<div style="padding:12px;color:#94a3b8;text-align:center;">ไม่พบโมเดลตามตัวกรอง</div>';
  };

  window.stModelOp = function(op, key){ stCall('model_op', {op, key}); };
  window.stRunChatModel = async function(key){ await stCall('model_op', {op:'run', key}); stTab(1); };
  window.stJumpImg = function(key){ document.getElementById('st-img-key').value = key; stTab(1); document.getElementById('st-uin').focus(); };
  window.stQuickSwitch = function(){ const k=document.getElementById('st-qsw').value; if(k) stCall('model_op', {op:'run', key:k}); };

  window.stApplyTpl = function(){
    const p = TPLS[document.getElementById('st-tpl').value];
    if(!p) return;
    document.getElementById('st-sys').value = p.prompt;
    document.getElementById('st-temp').value = p.temp;
    document.getElementById('st-minp').value = p.min_p;
    document.getElementById('st-reppen').value = p.repeat_penalty;
    document.getElementById('st-dry').value = p.dry;
    document.getElementById('st-maxtok').value = p.max_tokens;
  };

  function appendImgCard(im){
    const feed = document.getElementById('st-feed');
    const div = document.createElement('div');
    div.className = 'st-msg-b';
    div.innerHTML = `<div style="font-size:11.5px;color:#fbbf24;margin-bottom:4px;">🖼️ ${esc(im.key||'')} • ${esc(im.size||'')} • ${esc(im.seconds||'-')}s • seed ${esc(im.seed||'-')}</div><img src="${im.uri}" style="max-width:100%;border-radius:8px;display:block;margin-bottom:6px;"/><a href="${im.uri}" download="${esc(im.name||'strata.png')}" class="st-btn pri" style="text-decoration:none;font-size:12px;min-height:34px;">💾 แตะบันทึกรูปลงมือถือ/เครื่อง (${esc(im.name)})</a>`;
    feed.appendChild(div);
    feed.scrollTop = feed.scrollHeight;
  }

  window.stSendChat = async function(regen){
    const uin = document.getElementById('st-uin');
    const prompt = uin.value.trim();
    if(!regen && prompt.toLowerCase().startsWith('/img ')){
      uin.value = prompt.slice(5).trim();
      return stGenImg();
    }
    const feed = document.getElementById('st-feed');
    if(!regen){
      if(!prompt) return;
      uin.value = '';
      const ud = document.createElement('div'); ud.className = 'st-msg-u'; ud.textContent = '👤 ' + prompt;
      feed.appendChild(ud);
    } else if(feed.lastElementChild && feed.lastElementChild.classList.contains('st-msg-b')) {
      feed.removeChild(feed.lastElementChild);
    }
    const bd = document.createElement('div'); bd.className = 'st-msg-b'; bd.textContent = '⏳ กำลังคิดคำตอบ...';
    feed.appendChild(bd); feed.scrollTop = feed.scrollHeight;
    document.getElementById('st-send-btn').disabled = true;
    try {
      const res = await rpc('chat', {
        prompt, regen: !!regen,
        system: document.getElementById('st-sys').value,
        use_rag: document.getElementById('st-userag').checked,
        temp: parseFloat(document.getElementById('st-temp').value||'0.7'),
        min_p: parseFloat(document.getElementById('st-minp').value||'0.05'),
        repeat_penalty: parseFloat(document.getElementById('st-reppen').value||'1.08'),
        dry: parseFloat(document.getElementById('st-dry').value||'0'),
        max_tokens: parseInt(document.getElementById('st-maxtok').value||'1024',10)
      });
      if(!res || !res.ok){ bd.style.background='#450a0a'; bd.textContent = '❌ ' + (res && res.error ? res.error : 'เชื่อมต่อไม่สำเร็จ'); }
      else {
        const ragInfo = (res.rag_sources && res.rag_sources.length) ? ` • 📚 ${res.rag_sources.join(', ')}` : '';
        bd.innerHTML = `<div style="font-size:11.5px;color:#38bdf8;margin-bottom:4px;">🤖 ${esc(res.model)} • ${esc(res.spec)} • <b>${res.tok_s} tok/s</b> (${res.seconds}s)${esc(ragInfo)}</div><div>${esc(res.reply)}</div>`;
      }
    } finally {
      document.getElementById('st-send-btn').disabled = false;
      feed.scrollTop = feed.scrollHeight;
    }
  };

  window.stClearChat = async function(){ await rpc('clear_chat', {}); document.getElementById('st-feed').innerHTML = '<div style="color:#94a3b8;text-align:center;">🧹 ล้างประวัติแชทแล้ว</div>'; };

  window.stPresetWH = function(){
    const [w,h] = document.getElementById('st-img-sz').value.split('x');
    document.getElementById('st-iw').value = w; document.getElementById('st-ih').value = h;
  };

  window.stGenImg = async function(){
    const uin = document.getElementById('st-uin');
    const p = uin.value.trim();
    if(!p){ document.getElementById('st-img-stat').textContent = '⚠️ พิมพ์คำบรรยายภาพในช่องข้อความด้านบนก่อนครับ'; return; }
    uin.value = '';
    const ud = document.createElement('div'); ud.className = 'st-msg-u'; ud.style.background='#b45309'; ud.textContent = '🎨 ' + p;
    document.getElementById('st-feed').appendChild(ud);
    const r = await rpc('gen_image', {
      key: document.getElementById('st-img-key').value,
      prompt: p,
      negative: document.getElementById('st-ineg').value,
      width: parseInt(document.getElementById('st-iw').value||'1024',10),
      height: parseInt(document.getElementById('st-ih').value||'1024',10),
      steps: parseInt(document.getElementById('st-isteps').value||'0',10),
      seed: parseInt(document.getElementById('st-iseed').value||'-1',10),
      fast_cache: document.getElementById('st-ifast').checked
    });
    if(r && !r.ok) document.getElementById('st-img-stat').textContent = '❌ ' + r.error;
    stPoll(true);
  };

  window.stLoadGallery = async function(){
    const r = await rpc('gallery', {});
    if(r && r.images && r.images.length) r.images.forEach(im => appendImgCard(im));
    else document.getElementById('st-img-stat').textContent = 'ℹ️ ยังไม่มีรูปที่สร้างไว้ในเครื่อง';
  };

  window.stUploadRag = async function(){
    const inp = document.getElementById('st-files');
    if(!inp.files.length){ document.getElementById('st-rag-out').textContent = '⚠️ แตะเลือกไฟล์ก่อนครับ'; return; }
    for(const f of inp.files){
      const buf = await f.arrayBuffer();
      let bin = ''; const bytes = new Uint8Array(buf);
      for(let i=0;i<bytes.length;i+=8192) bin += String.fromCharCode.apply(null, bytes.subarray(i,i+8192));
      const r = await rpc('rag_op', {op:'upload', name:f.name, b64:btoa(bin)});
      document.getElementById('st-rag-out').textContent = r.msg || '';
    }
    stPoll(true);
  };
  window.stAddRagText = async function(){
    const el = document.getElementById('st-rag-txt');
    const r = await rpc('rag_op', {op:'add_text', text:el.value, source:document.getElementById('st-rag-src').value});
    el.value = ''; document.getElementById('st-rag-out').textContent = r.msg || ''; stPoll(true);
  };
  window.stSearchRag = async function(){
    const r = await rpc('rag_op', {op:'search', query:document.getElementById('st-rag-q').value});
    document.getElementById('st-rag-out').innerHTML = (r.hits||[]).map(h => `<div class="st-card"><b>#${h.rank} ${esc(h.source)}</b> (${h.score})<br>${esc(h.text)}</div>`).join('') || 'ไม่พบเนื้อหาที่ตรงกัน';
  };
  window.stRagOp = async function(op, extra){ const r = await rpc('rag_op', Object.assign({op}, extra)); document.getElementById('st-rag-out').textContent = r.msg || ''; stPoll(true); };

  let polling = false;
  window.stPoll = async function(force){
    if(polling && !force) return;
    polling = true;
    try {
      const next = await rpc('poll', {});
      if(next && next.models){ S = next; renderTop(); if(force) stRenderModels(); }
    } catch(_e){} finally { polling = false; }
  };

  // Init dropdowns once
  document.getElementById('st-tpl').innerHTML = Object.entries(TPLS).map(([k,v])=>`<option value="${esc(k)}">${esc(v.label)}</option>`).join('');
  document.getElementById('st-qsw').innerHTML = `<option value="">— เลือกสลับโมเดล —</option>` + INIT.text_models.map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');
  document.getElementById('st-img-key').innerHTML = INIT.image_models.map(k=>`<option value="${esc(k)}">${esc(k)}</option>`).join('');
  document.getElementById('st-rag-path').value = INIT.rag_default;
  stApplyTpl();
  renderTop();
  stRenderModels();

  setInterval(() => {
    if(S.job || S.progress || (S.img_status && S.img_status.startsWith('⏳')) || (S.img_status && S.img_status.startsWith('📥'))) stPoll(false);
  }, 1100);
  setInterval(() => stPoll(false), 4500);
})();
</script>
</div>
"""


def show_colab_touch_ui(model_dir: str | None = None, port: int | None = None):
    """สร้างหน้าจอ HTML5 Touch Studio 3 แท็บ ที่แตะปุ่มได้ลื่น 100% ทั้งบนเบราว์เซอร์มือถือ (Safari/Chrome) และบนคอมพิวเตอร์"""
    from google.colab import output as colab_output  # type: ignore[import-not-found]

    from .config import (
        ALL_PRESET_KEYS,
        COMFY_PORT,
        DEFAULT_PORT,
        DEFAULT_RAG_PATH,
        DRAFT_CATALOG,
        DRAFT_DIR,
        IMAGE_PRESET_KEYS,
        LLAMA_LOG_FILE,
        MODEL_CATALOG,
        MODEL_DIR,
        SPEC_STATE,
        SYSTEM_PROMPT_TEMPLATES,
        TEXT_PRESET_KEYS,
    )
    from .download import (
        _find_draft_gguf,
        cancel_active_download,
        clean_download_cache,
        get_download_progress,
        install_draft_model,
        install_model,
        keep_only_models,
        remove_model,
    )
    from .imagegen import (
        ensure_comfyui,
        generate_image,
        image_data_uri,
        image_model_status,
        install_image_model,
        link_all_downloaded_image_models,
        link_image_model,
        list_generated,
        start_comfyui,
    )
    from .platforms import detect as detect_platform
    from .rag import GLOBAL_RAG
    from .server import (
        _check_server_online,
        _invalidate_server_cache,
        check_gpu_and_recommend,
        get_openai_client,
        plan_multi_gpu,
        start_strata_server,
        stop_strata_server,
        switch_strata_model,
    )
    from .utils import _dir_stats, _fmt_bytes, _fmt_rate, _invalidate_dir_cache, _remember, net_bandwidth, recent_log

    mdir = model_dir or MODEL_DIR
    srv_port = int(port or DEFAULT_PORT)
    plat = detect_platform()
    total_models = len(ALL_PRESET_KEYS)
    total_img = len(IMAGE_PRESET_KEYS)

    def _snapshot() -> dict:
        cp = Path(mdir) if Path(mdir).exists() else Path(plat["home"])
        du = shutil.disk_usage(cp)
        hw = check_gpu_and_recommend(verbose=False)
        on, mid = _check_server_online(srv_port)
        net = net_bandwidth()
        gpu_plan = plan_multi_gpu(0.0, hw.get("devices"))
        usable = float(gpu_plan.get("usable_gb", 0.0)) if gpu_plan.get("has_gpu") else 0.0

        base = Path(mdir)
        cnt_dirs = len([d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]) if base.exists() else 0
        models_out = []
        for k in ALL_PRESET_KEYS:
            cfg = MODEL_CATALOG[k]
            sub = base / k
            ex = sub.is_dir()
            st = _dir_stats(sub) if ex else {"bytes": 0, "gguf": 0}
            is_img = cfg.get("kind") == "image"
            is_lora = cfg.get("kind") == "lora"
            if is_img:
                ready = (ex and all((sub / r).is_file() for r in cfg.get("targets", ())))
            elif is_lora:
                ready = (ex and any(p.suffix.lower() in (".safetensors", ".pt") for p in sub.rglob("*") if p.is_file() and not p.name.endswith(".part")))
            else:
                ready = (int(st["gguf"]) > 0)
            agb = float(cfg["approx_gb"])
            fit = ""
            if is_lora:
                fit = "ok"
            elif not is_img and usable > 0:
                fit = "ok" if agb <= usable * 0.9 else ("warn" if agb <= usable * 1.25 else "over")
            models_out.append({
                "key": k,
                "kind": "lora" if is_lora else ("image" if is_img else "text"),
                "desc": cfg["desc"],
                "repo": cfg["repo_id"],
                "approx_gb": agb,
                "installed": ready,
                "size_txt": _fmt_bytes(int(st["bytes"])) if ready else f"~{agb} GB",
                "tags": list(cfg.get("tags") or []),
                "has_mtp": bool(cfg.get("has_mtp")),
                "fit": fit,
                "can_keep": bool(ex and cnt_dirs > 1),
                "can_del": bool(ex),
            })

        drafts_out = []
        for dk, dv in DRAFT_CATALOG.items():
            fp = _find_draft_gguf(Path(DRAFT_DIR) / dk)
            drafts_out.append({
                "key": dk,
                "label": f"{'✅ ' if fp else '⬜ '}{dv['desc']}",
                "installed": bool(fp),
            })

        prog = get_download_progress()
        prog_fmt = None
        if prog.get("active"):
            eta = int(prog.get("eta_s", 0))
            prog_fmt = {
                "pct": float(prog.get("pct", 0.0)),
                "done": _fmt_bytes(int(prog.get("done_bytes", 0))),
                "total": _fmt_bytes(int(prog.get("total_bytes", 1))),
                "rate": _fmt_rate(float(prog.get("bps", 0.0))),
                "eta": f"{eta // 60}m {eta % 60:02d}s" if eta >= 60 else (f"{eta}s" if eta > 0 else "กำลังเร่งสปีด..."),
                "files": f"{prog.get('files_done', 0)}/{prog.get('files_total', 1)}",
                "repo": str(prog.get("repo_id", "")),
                "file": str(prog.get("current_file", "")),
            }

        tail: list[str] = []
        with contextlib.suppress(OSError):
            if LLAMA_LOG_FILE.exists():
                tail = LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-4:]

        return {
            "online": on,
            "active_model": mid if on else "Offline",
            "port": srv_port,
            "platform": plat["label"],
            "spec_label": SPEC_STATE.get("active_label", ""),
            "spec": {
                "mode": SPEC_STATE.get("mode", "auto"),
                "draft_max": int(SPEC_STATE.get("draft_max", 0)),
                "kv_q8": bool(SPEC_STATE.get("kv_q8", True)),
                "moe_cpu_auto": bool(SPEC_STATE.get("moe_cpu_auto", True)),
                "draft_model": str(SPEC_STATE.get("draft_model", "")),
            },
            "rag_chunks": len(GLOBAL_RAG.chunks),
            "gpu": (
                f"🎮 {hw['gpu_name']} ({hw['vram_free_gb']:.1f}/{hw['vram_total_gb']:.1f}GB)"
                if hw["has_gpu"]
                else "🖥️ CPU (ยังไม่เปิด GPU)"
            ),
            "disk": {
                "free_gb": round(du.free / 1024**3, 2),
                "total_gb": round(du.total / 1024**3, 2),
                "pct": round(min(100.0, du.used / du.total * 100), 1) if du.total else 0.0,
            },
            "net": f"⬇ {_fmt_rate(net['down_bps'])} · ⬆ {_fmt_rate(net['up_bps'])}",
            "job": _TOUCH_STATE["job"],
            "notice": _TOUCH_STATE["notice"],
            "notice_kind": _TOUCH_STATE["notice_kind"],
            "progress": prog_fmt,
            "img_status": _TOUCH_STATE["img_status"],
            "last_image": _TOUCH_STATE["last_image"],
            "logs": [*recent_log(12), *tail][-14:],
            "models": models_out,
            "drafts": drafts_out,
        }

    def _run_bg(label: str, fn) -> None:
        if _TOUCH_STATE["job"]:
            _TOUCH_STATE["notice"] = f"⏳ กำลังทำ '{_TOUCH_STATE['job']}' อยู่ — กดปุ่ม ⏹️ ยกเลิกงาน หากต้องการหยุดก่อน"
            _TOUCH_STATE["notice_kind"] = "warn"
            return

        def _worker():
            _TOUCH_STATE["job"] = label
            _TOUCH_STATE["notice"] = f"⏳ เริ่มดำเนินการ: {label}..."
            _TOUCH_STATE["notice_kind"] = "info"
            try:
                fn()
                _TOUCH_STATE["notice"] = f"✅ ดำเนินการสำเร็จ: {label}"
                _TOUCH_STATE["notice_kind"] = "ok"
            except Exception as exc:  # noqa: BLE001
                _remember(f"❌ {exc}")
                _TOUCH_STATE["notice"] = f"❌ {exc}"
                _TOUCH_STATE["notice_kind"] = "err"
            finally:
                _TOUCH_STATE["job"] = ""
                _invalidate_dir_cache()
                _invalidate_server_cache(srv_port)

        threading.Thread(target=_worker, daemon=True, name="strata-touch-job").start()

    def _rpc(action: str, payload: dict | None = None):
        data = payload or {}
        try:
            if action == "poll":
                return JSON(_snapshot())

            if action == "cancel":
                cancel_active_download()
                _TOUCH_STATE["job"] = ""
                _TOUCH_STATE["img_status"] = ""
                _invalidate_dir_cache()
                _invalidate_server_cache(srv_port)
                _TOUCH_STATE["notice"] = "⏹️ ยกเลิกงานค้างเรียบร้อย (ไฟล์ .part ที่โหลดค้างไว้สามารถกดโหลดต่อได้ทันที)"
                _TOUCH_STATE["notice_kind"] = "warn"
                return JSON(_snapshot())

            if action == "set_spec":
                SPEC_STATE.update({
                    "mode": str(data.get("mode", SPEC_STATE["mode"])),
                    "draft_max": int(data.get("draft_max", SPEC_STATE["draft_max"])),
                    "kv_q8": bool(data.get("kv_q8", SPEC_STATE["kv_q8"])),
                    "moe_cpu_auto": bool(data.get("moe_cpu_auto", SPEC_STATE.get("moe_cpu_auto", True))),
                })
                if data.get("restart"):
                    on, mid = _check_server_online(srv_port, ttl=0.0)
                    cdir = SPEC_STATE.get("last_model_dir") or (str(Path(mdir) / mid) if on and (Path(mdir) / mid).is_dir() else None)
                    if cdir and Path(cdir).is_dir():
                        _run_bg(f"รีสตาร์ทใช้โหมด {SPEC_STATE['mode']}", lambda: start_strata_server(model_dir=cdir, port=srv_port, restart=True))
                    else:
                        _TOUCH_STATE["notice"] = f"✅ บันทึกโหมด {SPEC_STATE['mode']} แล้ว — กด 🚀 รัน & แชท ที่แถวโมเดลได้เลย"
                        _TOUCH_STATE["notice_kind"] = "ok"
                return JSON(_snapshot())

            if action == "dl_draft":
                dk = str(data.get("key", "")).strip()
                if dk:
                    def _job_d():
                        gp, is_m = install_draft_model(dk, draft_base_dir=DRAFT_DIR)
                        is_e3 = bool(DRAFT_CATALOG.get(dk, {}).get("is_eagle3")) or ("eagle3" in Path(gp).name.lower())
                        SPEC_STATE["draft_model"] = gp
                        SPEC_STATE["mode"] = "eagle3" if is_e3 else ("mtp" if is_m else "draft")

                    _run_bg(f"โหลด Draft {dk}", _job_d)
                return JSON(_snapshot())

            if action == "model_op":
                op = str(data.get("op", ""))
                key = str(data.get("key", ""))
                if op == "download" and key in MODEL_CATALOG:
                    _run_bg(f"โหลด {key}", lambda: install_model(key, model_dir=mdir, port=srv_port))
                elif op == "run" and key in TEXT_PRESET_KEYS:
                    has_f = (Path(mdir) / key).is_dir() and _dir_stats(Path(mdir) / key)["gguf"] > 0
                    _run_bg(
                        f"เปิดใช้งาน {key}",
                        lambda: switch_strata_model(key, model_dir=mdir, port=srv_port)
                        if has_f
                        else install_model(key, model_dir=mdir, port=srv_port, auto_start_server=True),
                    )
                elif op == "prep_img" and key in IMAGE_PRESET_KEYS:
                    def _prep():
                        ensure_comfyui(pull=False)
                        if not image_model_status(key, mdir)["downloaded"]:
                            install_image_model(key, model_dir=mdir)
                        link_image_model(key, model_dir=mdir)
                        link_all_downloaded_image_models(model_dir=mdir)
                        start_comfyui(port=COMFY_PORT, public="local")

                    _run_bg(f"เตรียม ComfyUI ({key})", _prep)
                elif op == "keep" and key in MODEL_CATALOG:
                    _run_bg(f"เก็บเฉพาะ {key}", lambda: keep_only_models(key, model_dir=mdir))
                elif op == "delete" and key in MODEL_CATALOG:
                    _run_bg(f"ลบ {key}", lambda: remove_model(key, model_dir=mdir))
                elif op == "stop_server":
                    stop_strata_server()
                    _invalidate_server_cache(srv_port)
                    _TOUCH_STATE["notice"] = "🛑 หยุด llama-server เรียบร้อย"
                    _TOUCH_STATE["notice_kind"] = "info"
                elif op == "clean_cache":
                    _run_bg("ล้างแคชดาวน์โหลด", lambda: clean_download_cache(model_dir=mdir))
                elif op == "delete_all":
                    _run_bg("ลบโมเดลทั้งหมด", lambda: remove_model("all", model_dir=mdir, stop_server=True))
                return JSON(_snapshot())

            if action == "chat":
                is_on, active_mid = _check_server_online(srv_port, ttl=0.3)
                if not is_on:
                    return JSON({"ok": False, "error": f"Server พอร์ต {srv_port} ยังไม่เปิด! แตะ 🚀 รัน & แชท ในแท็บ 1 ก่อนครับ"})
                regen = bool(data.get("regen"))
                hist: list[dict] = _TOUCH_STATE["history"]
                if regen:
                    if len(hist) < 2 or hist[-1]["role"] != "assistant":
                        return JSON({"ok": False, "error": "ยังไม่มีคำตอบล่าสุดให้สุ่มใหม่"})
                    hist.pop()
                    q = hist.pop()["content"]
                else:
                    q = str(data.get("prompt", "")).strip()
                    if not q:
                        return JSON({"ok": False, "error": "พิมพ์ข้อความก่อนส่งครับ"})
                sys_p = str(data.get("system", SYSTEM_PROMPT_TEMPLATES["assistant"]["prompt"]))
                use_rag = bool(data.get("use_rag", True))
                aug_q, docs = GLOBAL_RAG.build_augmented_prompt(q, top_k=int(data.get("top_k", 3))) if (use_rag and GLOBAL_RAG.chunks) else (q, [])
                _ctx = max(2048, int(SPEC_STATE.get("ctx_size", 4096)))
                _mtok = max(128, min(int(data.get("max_tokens", 1024)), _ctx - 1024))
                extra: dict = {
                    "min_p": round(float(data.get("min_p", 0.05)), 3),
                    "repeat_penalty": round(float(data.get("repeat_penalty", 1.08)), 3),
                }
                if float(data.get("dry", 0.0)) > 0:
                    extra["dry_multiplier"] = round(float(data.get("dry", 0.0)), 2)
                t0 = time.perf_counter()
                resp = get_openai_client(port=srv_port).chat.completions.create(
                    model=active_mid,
                    messages=[{"role": "system", "content": sys_p}, *hist, {"role": "user", "content": aug_q}],
                    temperature=float(data.get("temp", 0.7)),
                    top_p=float(data.get("top_p", 0.95)),
                    max_tokens=_mtok,
                    extra_body=extra,
                    stream=False,
                )
                el = max(0.01, time.perf_counter() - t0)
                reply = (resp.choices[0].message.content or "") if resp.choices else ""
                comp_toks = getattr(resp.usage, "completion_tokens", None) or max(1, len(reply) // 4)
                hist.extend([{"role": "user", "content": q}, {"role": "assistant", "content": reply}])
                while len(hist) > 20:
                    hist.pop(0)
                return JSON({
                    "ok": True,
                    "user": q,
                    "reply": reply,
                    "model": active_mid,
                    "spec": SPEC_STATE.get("active_label", ""),
                    "tok_s": round(comp_toks / el, 1),
                    "seconds": round(el, 2),
                    "rag_sources": [d["source"] for d in docs],
                    "regen": regen,
                })

            if action == "clear_chat":
                _TOUCH_STATE["history"].clear()
                return JSON({"ok": True})

            if action == "gen_image":
                key = str(data.get("key", IMAGE_PRESET_KEYS[0]))
                prompt = str(data.get("prompt", "")).strip()
                if not prompt:
                    return JSON({"ok": False, "error": "พิมพ์คำบรรยายภาพก่อนกดสร้างภาพครับ"})
                neg = str(data.get("negative", "")).strip()
                w = max(512, min(1536, (int(data.get("width", 1024)) // 16) * 16))
                h = max(512, min(1536, (int(data.get("height", 1024)) // 16) * 16))
                steps = int(data.get("steps", 0)) or None
                seed_in = int(data.get("seed", -1))
                seed = seed_in if seed_in >= 0 else None
                fast_c = bool(data.get("fast_cache", True))

                def _img_worker():
                    try:
                        _TOUCH_STATE["img_status"] = f"⏳ เริ่มเตรียมวาดภาพ {key} ({w}×{h})..."
                        if not image_model_status(key, mdir)["downloaded"]:
                            _TOUCH_STATE["img_status"] = f"📥 กำลังโหลดโมเดลภาพ {key} ครั้งแรก (ดูความเร็วสดด้านบน)..."
                            install_image_model(key, model_dir=mdir)
                        res = generate_image(
                            key,
                            prompt,
                            negative=neg,
                            width=w,
                            height=h,
                            steps=steps,
                            seed=seed,
                            port=COMFY_PORT,
                            fast_cache=fast_c,
                            on_progress=lambda m: _TOUCH_STATE.__setitem__("img_status", m),
                        )
                        _TOUCH_STATE["last_image"] = {
                            "uri": image_data_uri(res["path"]),
                            "name": Path(res["path"]).name,
                            "key": key,
                            "seconds": res["seconds"],
                            "seed": res.get("seed", "-"),
                            "size": f"{w}×{h}",
                            "prompt": prompt,
                        }
                        _TOUCH_STATE["img_status"] = f"✅ วาดภาพเสร็จใน {res['seconds']}s ({w}×{h} • seed {res.get('seed', '-')})"
                    except Exception as exc:  # noqa: BLE001
                        _TOUCH_STATE["img_status"] = f"❌ สร้างภาพไม่สำเร็จ: {exc}"
                    finally:
                        _TOUCH_STATE["job"] = ""

                _TOUCH_STATE["job"] = f"วาดภาพ {key}"
                threading.Thread(target=_img_worker, daemon=True, name="strata-touch-img").start()
                return JSON({"ok": True, "started": True})

            if action == "gallery":
                imgs = []
                for p in list_generated(6):
                    with contextlib.suppress(OSError):
                        imgs.append({"uri": image_data_uri(p), "name": Path(p).name})
                return JSON({"ok": True, "images": imgs})

            if action == "rag_op":
                rop = str(data.get("op", ""))
                if rop == "upload":
                    name = str(data.get("name", "doc.txt"))
                    raw = base64.b64decode(str(data.get("b64", "")))
                    csz = int(data.get("chunk_size", 500))
                    if name.lower().endswith(".pdf"):
                        GLOBAL_RAG.add_pdf_bytes(raw, filename=name, chunk_size=csz)
                    else:
                        GLOBAL_RAG.add_documents([{"source": name, "text": raw.decode("utf-8", errors="ignore")}], chunk_size=csz)
                    return JSON({"ok": True, "chunks": len(GLOBAL_RAG.chunks), "msg": f"✅ นำเข้า {name} สำเร็จ ({len(GLOBAL_RAG.chunks)} chunks)"})
                if rop == "add_text":
                    txt = str(data.get("text", "")).strip()
                    if txt:
                        GLOBAL_RAG.add_documents([{"source": str(data.get("source", "note")), "text": txt}], chunk_size=int(data.get("chunk_size", 500)))
                    return JSON({"ok": True, "chunks": len(GLOBAL_RAG.chunks), "msg": f"✅ เพิ่มข้อความแล้ว ({len(GLOBAL_RAG.chunks)} chunks)"})
                if rop == "search":
                    hits = GLOBAL_RAG.search(str(data.get("query", "")), top_k=4)
                    return JSON({"ok": True, "hits": hits})
                if rop == "clear":
                    GLOBAL_RAG.clear()
                    return JSON({"ok": True, "chunks": 0, "msg": "🧹 ล้างคลัง RAG แล้ว"})
                if rop == "save":
                    sp = GLOBAL_RAG.save(str(data.get("path", DEFAULT_RAG_PATH)) or DEFAULT_RAG_PATH)
                    return JSON({"ok": True, "chunks": len(GLOBAL_RAG.chunks), "msg": f"💾 บันทึกแล้วที่ {sp}"})
                if rop == "load":
                    n = GLOBAL_RAG.load(str(data.get("path", DEFAULT_RAG_PATH)) or DEFAULT_RAG_PATH)
                    return JSON({"ok": True, "chunks": n, "msg": f"📂 โหลดคลังเดิม {n} chunks เรียบร้อย"})

            return JSON({"ok": False, "error": f"ไม่รู้จักคำสั่ง {action}"})
        except Exception as exc:  # noqa: BLE001
            return JSON({"ok": False, "error": str(exc)})

    colab_output.register_callback("strata.rpc", _rpc)

    init_json = json.dumps({
        "state": _snapshot(),
        "templates": SYSTEM_PROMPT_TEMPLATES,
        "text_models": TEXT_PRESET_KEYS,
        "image_models": IMAGE_PRESET_KEYS,
        "rag_default": DEFAULT_RAG_PATH,
    }, ensure_ascii=False).replace("</", "<\\/")

    html_ui = (
        _TOUCH_HTML_TEMPLATE
        .replace("__INIT_JSON__", init_json)
        .replace("__TOTAL_MODELS__", str(total_models))
        .replace("__TOTAL_IMG__", str(total_img))
    )
    display(HTML(html_ui))
    return {"backend": "colab_touch_html5", "port": srv_port, "model_dir": mdir}


def launch(
    model_dir: str | None = None,
    port: int | None = None,
    install_llama_server: bool = True,
    fallback_text: bool = True,
    ui_backend: str = "auto",
):
    """เปิด Studio UI (3 แท็บ):
    - บน Google Colab (`ui_backend="auto"`): ใช้ **HTML5 Mobile & Desktop Touch Studio** (`google.colab.kernel.invokeFunction`)
      แตะปุ่มได้ลื่น 100% ทั้งบนมือถือ iOS/Android และบนคอมพิวเตอร์ ไม่มีปัญหาปุ่ม ipywidgets ค้าง
    - บน Kaggle / JupyterLab / Unit Test: ใช้ `show_model_manager_ui` (ipywidgets)
    """
    from .config import DEFAULT_PORT, MODEL_DIR
    from .platforms import detect
    from .server import install_llama_server as install_server
    from .ui import show_model_manager_ui

    if install_llama_server:
        install_server()

    plat = detect()["name"]
    if ui_backend in ("auto", "touch", "mobile") and plat == "colab":
        with contextlib.suppress(Exception):
            import google.colab.output  # noqa: F401  # type: ignore[import-not-found]

            return show_colab_touch_ui(model_dir=model_dir or MODEL_DIR, port=port or DEFAULT_PORT)

    try:
        ensure_widgets()
    except Exception as e:  # noqa: BLE001
        _rprint()(f"[yellow]⚠️ ตรวจวิดเจ็ตไม่สำเร็จ: {e}[/yellow]")
    try:
        return show_model_manager_ui(model_dir=model_dir or MODEL_DIR, port=port or DEFAULT_PORT)
    except Exception as e:  # noqa: BLE001
        _rprint()(f"[bold red]❌ สร้าง UI ไม่สำเร็จ: {e}[/bold red]")
        if fallback_text:
            _rprint()("[yellow]→ แสดงสถานะแบบข้อความแทน[/yellow]")
            diagnose()
            text_dashboard(model_dir or MODEL_DIR)
        return None
