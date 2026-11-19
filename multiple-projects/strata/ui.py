"""Studio UI (ipywidgets) 3 แท็บ — Thread-safe 100% บน Colab (ipywidgets 7.7) & Kaggle (ipywidgets 8):
- ไม่ใช้ `with widgets.Output():` ในเธรดเบื้องหลัง (ซึ่งทำให้ ZMQ/comm ของ Colab ค้างจนกดปุ่มไม่ได้)
- ไม่ล็อกปุ่มทั้งหน้าจอค้างไว้ (มีปุ่ม `⏹️ ยกเลิกงาน` สั่งหยุดดาวน์โหลดได้ใน 0.1 วินาที)
- รองรับป้ายกำกับและตัวกรองค้นหาหมวด OBLITERATED / Unrestricted / Heretic / RP-ERP ครบ 54 รุ่นบน Kaggle & Colab
"""
from __future__ import annotations

import glob
import html as html_lib
import re
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display as _ipy_display
from rich import print as rprint

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
    PROFILE_LABELS,
    RECOMMENDED_LABELS,
    RECOMMENDED_PROFILES,
    SPEC_STATE,
    SYSTEM_PROMPT_TEMPLATES,
    TEXT_PRESET_KEYS,
)
from .download import (
    _find_draft_gguf,
    cancel_active_download,
    clean_download_cache,
    get_download_progress,
    install_all_models,
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
from .rag import GLOBAL_RAG, StrataRAG
from .server import (
    _check_server_online,
    _invalidate_server_cache,
    check_gpu_and_recommend,
    get_openai_client,
    plan_multi_gpu,
    start_strata_server,
    stop_strata_server,
    strata_server_status,
    switch_strata_model,
)
from .utils import (
    NetworkMonitor,
    _dir_stats,
    _fmt_bytes,
    _fmt_rate,
    _invalidate_dir_cache,
    _remember,
    net_bandwidth,
    recent_log,
)

display = _ipy_display  # ให้ ui_smoke_test monkeypatch `strata_ui.display` ได้เหมือนเดิม


def _fmt_reply_html(raw: str, docs: list[dict] | None = None) -> str:
    out = ""
    if docs:
        items = "".join(f"<div><b style='color:#fbbf24;'>[{html_lib.escape(d['source'])}]</b> ({d['score']}): {html_lib.escape(d['text'][:180])}...</div>" for d in docs)
        out += f"<details style='background:#0f172a;border:1px solid #1e3a8a;border-radius:6px;padding:6px 10px;margin-bottom:6px;font-size:12px;color:#cbd5e1;'><summary style='cursor:pointer;color:#60a5fa;font-weight:600;'>📚 อ้างอิงจาก StrataRAG ({len(docs)} ส่วน)</summary>{items}</details>"
    thinks: list[str] = []
    vis = re.sub(r"<think>(.*?)</think>", lambda m: (thinks.append(m.group(1).strip()) or ""), raw, flags=re.DOTALL)
    if "<think>" in vis and "</think>" not in vis:
        vis, rest = vis.split("<think>", 1); thinks.append(rest.strip() + " ⏳...")
    if thinks:
        out += f"<details style='background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px 10px;margin-bottom:6px;font-size:12px;color:#94a3b8;'><summary style='cursor:pointer;color:#38bdf8;'>🧠 กระบวนการคิด</summary>{html_lib.escape(chr(10).join(thinks)).replace(chr(10), '<br>')}</details>"
    return out + f"<div style='line-height:1.6;'>{html_lib.escape(vis.strip()).replace(chr(10), '<br>') or '...'}</div>"


def _L(**kw) -> widgets.Layout:
    return widgets.Layout(**kw)


def _B(desc: str, style: str = "", w: str = "110px", **kw) -> widgets.Button:
    return widgets.Button(description=desc, button_style=style, layout=_L(width=w, **kw))


_PLAT = detect_platform()

LIVE_REFS: list = []


class LiveRefresher:
    """รีเฟรช UI แบบเรียลไทม์ใน thread แยก — เปิด/ปิดได้ และเรียก run_once() เพื่อทดสอบได้"""

    def __init__(self, tick: Callable[[int], None], interval: float = 1.0):
        self.tick = tick
        self.interval = max(0.5, float(interval))
        self.running = False
        self.ticks = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.running = True
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="strata-live")
        self._thread.start()

    def stop(self) -> None:
        self.running = False

    def run_once(self) -> None:
        with self._lock:
            self.ticks += 1
            self.tick(self.ticks)

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval)
            if not self.running:
                continue
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                continue


def _fit_badge(approx_gb: float, plan: dict) -> str:
    if not plan.get("has_gpu"):
        return ""
    usable = float(plan.get("usable_gb", 0.0))
    if usable <= 0:
        return ""
    box = "padding:1px 6px;border-radius:6px;font-size:10px;white-space:nowrap;"
    if plan.get("count", 1) > 1:
        per = " + ".join(f"{g:.0f}" for g in plan.get("per_gpu_gb", []))
        tail = f" · {per} GB/ใบ"
    else:
        tail = ""
    if approx_gb <= usable * 0.9:
        return f"<span style='background:#052e16;color:#4ade80;{box}'>🎮 พอดี VRAM{tail}</span>"
    if approx_gb <= usable * 1.25:
        return f"<span style='background:#422006;color:#fbbf24;{box}'>⚠️ เกือบเต็ม{tail}</span>"
    return f"<span style='background:#450a0a;color:#fca5a5;{box}'>❌ เกิน VRAM (offload RAM)</span>"


def _render_fetch_banner(prog: dict, job_label: str = "") -> str:
    """วาดแถบ Fetching / งานที่กำลังทำแบบเรียลไทม์ (ไม่ล็อกปุ่มอื่น กดยกเลิกได้ตลอด)"""
    if prog.get("active"):
        pct = float(prog.get("pct", 0.0))
        done_b = int(prog.get("done_bytes", 0))
        tot_b = int(prog.get("total_bytes", 1))
        bps = float(prog.get("bps", 0.0))
        eta = int(prog.get("eta_s", 0))
        f_ok = int(prog.get("files_done", 0))
        f_tot = int(prog.get("files_total", 1))
        repo = html_lib.escape(str(prog.get("repo_id", "")))
        cur_f = html_lib.escape(str(prog.get("current_file", "")))
        eta_txt = f"{eta // 60}m {eta % 60:02d}s" if eta >= 60 else (f"{eta}s" if eta > 0 else "กำลังเร่งสปีด...")
        return (
            "<div style='background:#0c4a6e;color:#f8fafc;padding:10px 14px;border-radius:10px;"
            "border:1px solid #38bdf8;font-family:sans-serif;margin:6px 0;'>"
            "<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;font-size:12.5px;margin-bottom:6px;'>"
            f"<span>📥 <b>Fetching เรียลไทม์ [{f_ok}/{f_tot}]</b> • <span style='color:#bae6fd;'>{repo}</span>"
            f"{' • <code>' + cur_f + '</code>' if cur_f else ''}</span>"
            f"<span><b>{pct:.1f}%</b> ({_fmt_bytes(done_b)} / {_fmt_bytes(tot_b)}) • "
            f"<b style='color:#fde047;'>⬇ {_fmt_rate(bps)}</b> • ETA {eta_txt} "
            "<span style='color:#fca5a5;'>(กด ⏹️ ยกเลิกงาน เพื่อหยุดได้ทันที)</span></span>"
            "</div>"
            "<div style='width:100%;background:#082f49;height:10px;border-radius:99px;overflow:hidden;'>"
            f"<div style='width:{min(100.0, max(1.5, pct)):.1f}%;background:linear-gradient(90deg,#38bdf8,#34d399);height:100%;transition:width 0.3s;'></div>"
            "</div></div>"
        )
    if job_label:
        return (
            "<div style='background:#1e293b;color:#e2e8f0;padding:8px 12px;border-radius:8px;"
            "border:1px solid #38bdf8;font-family:sans-serif;font-size:12px;margin:6px 0;'>"
            f"⏳ กำลังดำเนินการ: <b style='color:#38bdf8;'>{html_lib.escape(job_label)}</b> "
            "(ปุ่มอื่นยังกดใช้งานได้ปกติ • กด <b>⏹️ ยกเลิกงาน</b> หากต้องการหยุด)</div>"
        )
    return ""


def show_model_manager_ui(model_dir: str = MODEL_DIR, port: int = DEFAULT_PORT, rag_engine: StrataRAG = GLOBAL_RAG, live: bool = True):
    state = {"active": "ยังไม่ระบุ", "job": ""}
    history: list[dict] = []
    header, rag_stat, spec_hint, c_stat, rag_out = (widgets.HTML() for _ in range(5))
    dl_prog_out = widgets.HTML()
    action_notice = widgets.HTML()
    guide_card = widgets.HTML()

    def _set_notice(msg: str, kind: str = "info") -> None:
        if not msg:
            action_notice.value = ""
            return
        palette = {
            "ok": ("#064e3b", "#10b981", "#ecfdf5"),
            "warn": ("#422006", "#f59e0b", "#fef3c7"),
            "err": ("#450a0a", "#ef4444", "#fecaca"),
            "info": ("#0f172a", "#38bdf8", "#f8fafc"),
        }.get(kind, ("#0f172a", "#38bdf8", "#f8fafc"))
        action_notice.value = (
            f"<div style='background:{palette[0]};border:1px solid {palette[1]};color:{palette[2]};"
            "padding:8px 12px;border-radius:8px;font-family:sans-serif;font-size:12px;margin:4px 0;'>"
            f"{msg}</div>"
        )

    def _header_html() -> str:
        cp = Path(model_dir) if Path(model_dir).exists() else Path(_PLAT["home"])
        u = shutil.disk_usage(cp); tg, ug, fg = u.total / (1024**3), u.used / (1024**3), u.free / (1024**3)
        pct = min(100.0, ug / tg * 100) if tg else 0.0
        bc = "#22c55e" if pct < 70 else ("#f59e0b" if pct < 88 else "#ef4444")
        hw = check_gpu_and_recommend(verbose=False); on, mid = _check_server_online(port)
        if on:
            state["active"] = mid
        sb = f"<span style='background:#14532d;color:#4ade80;padding:3px 9px;border-radius:99px;font-size:12px;'>🟢 Online (:{port} • {mid})</span>" if on else f"<span style='background:#450a0a;color:#fca5a5;padding:3px 9px;border-radius:99px;font-size:12px;'>🔴 Offline (:{port})</span>"
        spb = f"<span style='background:#3b0764;color:#e9d5ff;padding:3px 9px;border-radius:99px;font-size:12px;'>{html_lib.escape(SPEC_STATE['active_label'])}</span>" if on and not SPEC_STATE["active_label"].startswith("⏹️") else ""
        rb = f"<span style='background:#1e3a8a;color:#93c5fd;padding:3px 9px;border-radius:99px;font-size:12px;'>📚 RAG: {len(rag_engine.chunks)}</span>"
        if hw["has_gpu"]:
            spl = plan_multi_gpu(0.0, hw.get("devices"))
            names = " + ".join(d["name"] for d in hw.get("devices", [])) if hw.get("gpu_count", 1) > 1 else hw["gpu_name"]
            per_dev = " · ".join(
                f"GPU{d['index']} {d['vram_free_gb']:.1f}/{d['vram_total_gb']:.0f}GB" for d in hw.get("devices", [])
            )
            gb = f"<span title='{per_dev}' style='cursor:help;'>🎮 {names} ({hw['vram_free_gb']:.1f}/{hw['vram_total_gb']:.1f}GB)</span>"
            if spl.get("count", 0) > 1:
                gb += f" · split {'/'.join(f'{r:.2f}' for r in spl['tensor_split'])} · main {spl['main_gpu']}"
        else:
            gb = "🖥️ CPU (ยังไม่เปิด GPU)"
        net = net_bandwidth()
        nb = (
            "<span style='background:#0c4a6e;color:#7dd3fc;padding:3px 9px;border-radius:99px;font-size:12px;'"
            f" title='รวมทั้งหมด — ดาวน์โหลด {_fmt_bytes(net['rx_bytes'])} • อัปโหลด {_fmt_bytes(net['tx_bytes'])}'>"
            f"⬇ {_fmt_rate(net['down_bps'])} · ⬆ {_fmt_rate(net['up_bps'])}"
            "</span>"
        )
        return f"<div style='background:#0f172a;color:#f8fafc;padding:12px 16px;border-radius:10px;border:1px solid #334155;font-family:sans-serif;margin-bottom:6px;'><div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;margin-bottom:6px;'><b style='font-size:15px;'>🎛️ Strata Studio ({_PLAT['label']} • Port {port} • {len(ALL_PRESET_KEYS)} Models + {len(DRAFT_CATALOG)} Draft/EAGLE-3 + RAG)</b><div>{sb} {spb} {rb} {nb} <span style='font-size:12px;color:#38bdf8;margin-left:4px;'>{gb}</span></div></div><div style='font-size:12px;color:#cbd5e1;margin-bottom:4px;'>💾 ดิสก์ว่าง <b style='color:{bc};'>{fg:.2f} GB</b> / {tg:.2f} GB ({pct:.1f}%)</div><div style='width:100%;background:#1e293b;height:8px;border-radius:99px;overflow:hidden;'><div style='width:{pct:.1f}%;background:{bc};height:100%;'></div></div></div>"

    def _draft_opts() -> list[tuple[str, str]]:
        opts = []
        for dk, dv in DRAFT_CATALOG.items():
            fp = _find_draft_gguf(Path(DRAFT_DIR) / dk)
            opts.append((f"✅ {dv['desc']} ({_fmt_bytes(fp.stat().st_size)})" if fp else f"⬜ {dv['desc']}", dk))
        opts.append(("⚙️ กำหนดเอง (พิมพ์ org/repo:*Q4_K_M*.gguf ช่องขวา)", "custom"))
        return opts

    def _img_dd_opts() -> list[tuple[str, str]]:
        out = []
        for k in IMAGE_PRESET_KEYS:
            ready = image_model_status(k, model_dir)["downloaded"]
            icon = "✅" if ready else ("🔥" if "heretic" in (MODEL_CATALOG[k].get("tags") or []) else "🎨")
            out.append((f"{icon} {k} (~{MODEL_CATALOG[k]['approx_gb']}GB)", k))
        return out

    spec_tb = widgets.ToggleButtons(
        options=[
            ("⏹️ ปิด", "off"),
            ("✨ Auto", "auto"),
            ("🦅 EAGLE-3", "eagle3"),
            ("🚀 MTP", "mtp"),
            ("⚡ Draft", "draft"),
            ("📚 N-gram+", "ngram"),
        ],
        value=SPEC_STATE["mode"],
        description="⚡ เร่งสปีด:",
        style={"button_width": "82px", "description_width": "72px"},
    )
    dmax_sl = widgets.IntSlider(value=int(SPEC_STATE["draft_max"]), min=0, max=24, description="Draft-Max:", continuous_update=False, layout=_L(width="195px"))
    chk_kvq8 = widgets.Checkbox(value=bool(SPEC_STATE["kv_q8"]), description="💾 KV Q8_0", indent=False, layout=_L(width="105px"))
    chk_moe = widgets.Checkbox(value=bool(SPEC_STATE.get("moe_cpu_auto", True)), description="🧠 MoE-CPU Auto", indent=False, layout=_L(width="135px"))
    b_apply_spec = _B("🔄 รีสตาร์ทใช้โหมดนี้", "info", "150px")
    draft_dd = widgets.Dropdown(options=_draft_opts(), value="llama32-1b-uncensored", description="📦 Draft:", layout=_L(width="380px"))
    b_dl_draft, b_clr_draft, b_rm_draft = _B("📥 โหลด & ใช้ Draft", "primary", "145px"), _B("✖ ล้าง", "", "68px"), _B("🗑️ ลบไฟล์", "danger", "88px")
    draft_in = widgets.Text(value=str(SPEC_STATE["draft_model"]), placeholder="เว้นว่าง = จับคู่ Draft ตามตระกูลคำศัพท์อัตโนมัติ หรือพิมพ์พาธ .gguf", description="ไฟล์ที่ใช้:", layout=_L(flex="1 1 230px"))

    def _upd_spec(_=None):
        SPEC_STATE.update({
            "mode": spec_tb.value,
            "draft_max": int(dmax_sl.value),
            "kv_q8": bool(chk_kvq8.value),
            "moe_cpu_auto": bool(chk_moe.value),
            "draft_model": draft_in.value.strip(),
        })
        dm_t = "Auto (EAGLE3=4, MTP=3, Draft=4, N-gram+=24)" if dmax_sl.value == 0 else str(dmax_sl.value)
        df_t = Path(SPEC_STATE["draft_model"]).name if SPEC_STATE["draft_model"] and Path(SPEC_STATE["draft_model"]).is_file() else "จับคู่ตามตระกูลอัตโนมัติ"
        mdesc = {
            "off": "⏹️ ปิด Speculative",
            "auto": "✨ Auto: EAGLE-3 > Draft (จับคู่ชุดคำศัพท์อัตโนมัติ) > MTP > N-gram+",
            "eagle3": f"🦅 EAGLE-3 (--spec-type draft-eagle3 • {html_lib.escape(df_t)})",
            "mtp": "🚀 MTP (--spec-type draft-mtp)",
            "draft": f"⚡ External Draft ({html_lib.escape(df_t)})",
            "ngram": "📚 N-gram+ (ngram-map-k / ngram-mod ฟรี VRAM)",
        }[spec_tb.value]
        kv_badge = ' | <b style="color:#4ade80;">KV:Q8_0</b>' if chk_kvq8.value else ""
        moe_badge = ' | <b style="color:#c084fc;">MoE-CPU:Auto</b>' if chk_moe.value else ""
        spec_hint.value = f"<span style='font-size:11.5px;color:#cbd5e1;'>{mdesc} | <b style='color:#38bdf8;'>Max:</b> {dm_t}{kv_badge}{moe_badge}</span>"

    def _on_dd_chg(ch):
        fp = _find_draft_gguf(Path(DRAFT_DIR) / ch["new"]) if ch["new"] in DRAFT_CATALOG else None
        if fp:
            draft_in.value = str(fp)
            cat_d = DRAFT_CATALOG[ch["new"]]
            spec_tb.value = "eagle3" if cat_d.get("is_eagle3") else ("mtp" if cat_d.get("is_mtp") else "draft")

    draft_dd.observe(_on_dd_chg, names="value")
    for w in (spec_tb, dmax_sl, chk_kvq8, chk_moe, draft_in):
        w.observe(_upd_spec, names="value")
    _upd_spec()

    flt_in = widgets.Text(
        value="",
        placeholder="🔍 พิมพ์ค้นหาชื่อโมเดล เช่น obliterated, skyfall, anubis, magnum, 70b...",
        description="ค้นหา:",
        layout=_L(flex="1 1 260px"),
    )
    flt_dd = widgets.Dropdown(
        options=[
            (f"📋 ทั้งหมด ({len(ALL_PRESET_KEYS)} รุ่น)", "all"),
            ("✅ ติดตั้งแล้ว", "installed"),
            ("🎮 พอดี VRAM การ์ดนี้", "fit"),
            ("💥 สาย OBLITERATED", "obliterated"),
            ("🔓 สาย Unrestricted", "unrestricted"),
            ("🎭 สาย Roleplay (RP / ERP)", "rp"),
            ("🔥 สาย Heretic / Uncensored", "heretic"),
            ("🎨 สร้างภาพ GGUF (6 รุ่น)", "image"),
            ("💻 สาย Code / Security", "code"),
        ],
        value="all",
        description="หมวดหมู่:",
        layout=_L(width="265px"),
    )

    rows_box = widgets.VBox(layout=_L(gap="6px", margin="6px 0"))

    net_out = widgets.HTML("<div style='font-size:12px;color:#64748b;'>📶 กดปุ่มเพื่อเริ่มวัดความเร็วเน็ตแบบเรียลไทม์ (อัปเดตทุก 1 วิ)</div>")
    b_net = _B("📶 วัดเน็ตเรียลไทม์", "", "165px")
    net_state: dict = {"run": False, "mon": NetworkMonitor()}

    def _net_card(mon: NetworkMonitor) -> str:
        sample = mon.sample()
        bar = min(100.0, sample["down_bps"] / (20 * 1024**2) * 100) if sample["down_bps"] else 0.0
        return (
            "<div style='font-family:sans-serif;background:#0f172a;color:#f8fafc;border:1px solid #334155;"
            "border-radius:8px;padding:8px 10px;font-size:12px;'>"
            f"<b style='color:#7dd3fc;'>⬇ {_fmt_rate(sample['down_bps'])}</b> · "
            f"<b style='color:#fcd34d;'>⬆ {_fmt_rate(sample['up_bps'])}</b>"
            f"<span style='color:#64748b;'> | รวม ⬇ {_fmt_bytes(sample['rx_bytes'])} • ⬆ {_fmt_bytes(sample['tx_bytes'])}</span>"
            "<div style='background:#1e293b;height:6px;border-radius:99px;margin-top:6px;overflow:hidden;'>"
            f"<div style='width:{bar:.1f}%;background:#38bdf8;height:100%;'></div></div></div>"
        )

    def _net_reset_button():
        try:
            b_net.description = "📶 วัดเน็ตเรียลไทม์"
            b_net.button_style = ""
        except Exception:
            pass

    def _net_loop():
        mon = net_state["mon"]
        mon.reset()
        while net_state["run"]:
            try:
                net_out.value = _net_card(mon)
            except Exception:
                break
            time.sleep(1.0)
        net_state["run"] = False
        _net_reset_button()

    def _toggle_net(_):
        if net_state["run"]:
            net_state["run"] = False
            _net_reset_button()
            return
        net_state["run"] = True
        b_net.description = "⏸ หยุดวัดเน็ต"
        b_net.button_style = "warning"
        threading.Thread(target=_net_loop, daemon=True).start()

    b_net.on_click(_toggle_net)

    b_all = _B(f"📦 โหลดครบ {len(ALL_PRESET_KEYS)} ตัว", "primary", "150px")
    b_cache = _B("🧹 ล้างแคช", "info", "100px")
    b_stat = _B("📡 สถานะ", "", "90px")
    b_stop = _B("🛑 หยุด Server", "warning", "115px")
    b_cancel = _B("⏹️ ยกเลิกงาน", "danger", "110px")
    b_delall = _B("🗑️ ลบทั้งหมด", "danger", "105px")
    b_ref = _B("🔄 รีเฟรช", "", "88px")
    chk_del = widgets.Checkbox(value=False, description="ยืนยันลบทั้งหมด", indent=False, layout=_L(width="125px"))

    def _on_cancel(_):
        cancel_active_download()
        state["job"] = ""
        dl_prog_out.value = ""
        _invalidate_dir_cache(); _invalidate_server_cache(port)
        _refresh()
        _set_notice("⏹️ สั่งยกเลิกงานค้างและปลดล็อกพร้อมใช้งานเรียบร้อย (ไฟล์ .part ที่โหลดค้างไว้สามารถกดโหลดต่อได้ทันที)", "warn")

    b_cancel.on_click(_on_cancel)

    def _use_sync_mode() -> bool:
        return (not live) or (display is not _ipy_display)

    def _act(fn: Callable, *a, active: str | None = None, tab: int | None = None, label: str = "", **kw):
        job_lbl = label or (str(a[0]) if a else getattr(fn, "__name__", "งานระบบ"))
        if _use_sync_mode():
            try:
                fn(*a, **kw)
                if active:
                    state["active"] = active
                if tab is not None:
                    tabs.selected_index = tab
            except Exception as e:
                _remember(f"❌ {e}")
                _set_notice(f"❌ {html_lib.escape(str(e))}", "err")
            finally:
                _invalidate_dir_cache(); _invalidate_server_cache(port)
                _refresh()
            return

        if state["job"]:
            _set_notice(
                f"⏳ กำลังทำ <b>{html_lib.escape(state['job'])}</b> อยู่ — กดปุ่ม <b>⏹️ ยกเลิกงาน</b> ด้านบนก่อนหากต้องการเริ่มงานใหม่",
                "warn",
            )
            return

        def _bg():
            state["job"] = job_lbl
            dl_prog_out.value = _render_fetch_banner(get_download_progress(), job_lbl)
            stop_pump = threading.Event()

            def _pump():
                while not stop_pump.wait(0.4):
                    try:
                        dl_prog_out.value = _render_fetch_banner(get_download_progress(), state["job"])
                    except Exception:
                        pass

            threading.Thread(target=_pump, daemon=True, name="strata-dl-bar").start()
            try:
                fn(*a, **kw)
                if active:
                    state["active"] = active
                if tab is not None:
                    tabs.selected_index = tab
                _set_notice(f"✅ ดำเนินการ <b>{html_lib.escape(job_lbl)}</b> เสร็จเรียบร้อย!", "ok")
            except Exception as e:
                _remember(f"❌ {e}")
                _set_notice(f"❌ {html_lib.escape(str(e))}", "err")
            finally:
                stop_pump.set()
                state["job"] = ""
                dl_prog_out.value = ""
                _invalidate_dir_cache(); _invalidate_server_cache(port)
                _refresh()

        threading.Thread(target=_bg, daemon=True, name="strata-worker").start()

    def _on_dl_draft(_):
        spec = draft_in.value.strip() if draft_dd.value == "custom" else draft_dd.value
        if not spec:
            spec_hint.value = "<span style='color:#fbbf24;'>⚠️ พิมพ์ org/repo:*Q4_K_M*.gguf ในช่อง 'ไฟล์ที่ใช้' ก่อนครับ</span>"; return

        def _job():
            gp, is_m = install_draft_model(spec, draft_base_dir=DRAFT_DIR)
            is_e3 = bool(DRAFT_CATALOG.get(spec, {}).get("is_eagle3")) or ("eagle3" in Path(gp).name.lower())
            draft_in.value = gp
            spec_tb.value = "eagle3" if is_e3 else ("mtp" if is_m else "draft")
            _upd_spec()

        tabs.selected_index = 0
        _act(_job, label=f"โหลด Draft {spec}")

    def _on_rm_draft(_):
        def _job():
            if draft_dd.value in DRAFT_CATALOG and (Path(DRAFT_DIR) / draft_dd.value).is_dir():
                shutil.rmtree(Path(DRAFT_DIR) / draft_dd.value, ignore_errors=True)
                _remember(f"🗑️ ลบ Draft '{draft_dd.value}' แล้ว")
            elif draft_in.value.strip() and Path(draft_in.value.strip()).is_file():
                Path(draft_in.value.strip()).unlink(missing_ok=True)
            draft_in.value = ""
            if spec_tb.value in ("draft", "eagle3"):
                spec_tb.value = "off"
            _upd_spec()

        _act(_job, label=f"ลบ Draft {draft_dd.value}")

    def _on_apply_spec(_):
        _upd_spec(); on, mid = _check_server_online(port, ttl=0.0)
        cdir = SPEC_STATE.get("last_model_dir") or (str(Path(model_dir) / mid) if on and (Path(model_dir) / mid).is_dir() else None)
        if cdir and Path(cdir).is_dir():
            _act(start_strata_server, model_dir=cdir, port=port, restart=True, label=f"รีสตาร์ทโหมดเร่งสปีด ({SPEC_STATE['mode']})")
        else:
            spec_hint.value = "<span style='color:#fbbf24;'>✅ บันทึกโหมดแล้ว! กด 🚀 รัน & แชท ของโมเดลหลักได้เลย</span>"

    def _on_del_all(_):
        if not chk_del.value:
            _set_notice("⚠️ กรุณาติ๊กช่อง 'ยืนยันลบทั้งหมด' ก่อนกดลบครับ", "warn")
            return
        chk_del.value = False
        _act(remove_model, "all", model_dir=model_dir, stop_server=True, active="หยุดแล้ว", label="ลบโมเดลทั้งหมด")

    b_dl_draft.on_click(_on_dl_draft)
    b_clr_draft.on_click(lambda _: (setattr(draft_in, "value", ""), setattr(spec_tb, "value", "off" if spec_tb.value in ("draft", "eagle3") else spec_tb.value), _upd_spec()))
    b_rm_draft.on_click(_on_rm_draft); b_apply_spec.on_click(_on_apply_spec)
    b_all.on_click(lambda _: _act(install_all_models, model_dir=model_dir, port=port, label="ดาวน์โหลดครบทุกโมเดล"))
    b_cache.on_click(lambda _: _act(clean_download_cache, model_dir=model_dir, label="ล้างแคชดาวน์โหลด"))
    b_stat.on_click(lambda _: _act(strata_server_status, port=port, label="ตรวจสถานะ Server"))
    b_stop.on_click(lambda _: _act(stop_strata_server, active="หยุดแล้ว", label="หยุด llama-server"))
    b_delall.on_click(_on_del_all)

    def _on_refresh(_):
        _invalidate_dir_cache(); _invalidate_server_cache(port)
        _refresh()
        _set_notice("🔄 รีเฟรชสถานะโมเดล ดิสก์ และ GPU ล่าสุดแล้ว", "info")

    b_ref.on_click(_on_refresh)

    rows: dict[str, dict] = {}

    def _prepare_image_for_cell3(opt: str) -> None:
        ensure_comfyui(pull=False)
        if not image_model_status(opt, model_dir)["downloaded"]:
            install_image_model(opt, model_dir=model_dir)
        link_image_model(opt, model_dir=model_dir)
        all_ready = link_all_downloaded_image_models(model_dir=model_dir)
        start_comfyui(port=COMFY_PORT, public="local")
        _set_notice(
            f"🎨 เตรียม ComfyUI + ผูกโมเดล <code>{html_lib.escape(opt)}</code> พร้อมแล้ว (:{COMFY_PORT} • ไม่ใช้ Tunnel) "
            f"— เลื่อนลงไปกดรัน <b>เซลล์ที่ 3</b> ด้านล่างเพื่อใช้งานหน้าจอ ComfyUI ได้เลยครับ "
            f"(หรือสลับไปแท็บ 2 เพื่อกดสร้างภาพด่วน • พร้อมใช้ {len(all_ready)} ตัว)",
            "ok",
        )

    def _row_widgets(k: str) -> dict:
        is_img = MODEL_CATALOG[k].get("kind") == "image"
        w = {
            "info": widgets.HTML(layout=_L(flex="1 1 250px")),
            "bd": _B("📥 โหลด", "primary", "95px"),
            "br": _B("🎨 สร้างภาพ" if is_img else "🚀 รัน & แชท", "success", "112px"),
            "bk": _B("📌 เก็บตัวเดียว", "", "105px"),
            "bx": _B("🗑️ ลบ", "danger", "70px"),
            "installed": False,
            "kind": "image" if is_img else "text",
        }
        w["bd"].on_click(lambda _, opt=k: _act(install_model, opt, model_dir=model_dir, port=port, label=f"โหลด {opt}"))
        if is_img:
            w["br"].on_click(lambda _, opt=k: _act(_prepare_image_for_cell3, opt, label=f"เตรียม ComfyUI ({opt})"))
        else:
            w["br"].on_click(
                lambda _, opt=k: _act(
                    switch_strata_model if rows[opt]["installed"] else install_model,
                    opt,
                    model_dir=model_dir,
                    port=port,
                    active=opt,
                    tab=1,
                    label=f"เปิดใช้งาน {opt}",
                    **({} if rows[opt]["installed"] else {"auto_start_server": True}),
                )
            )
        w["bk"].on_click(lambda _, opt=k: _act(keep_only_models, opt, model_dir=model_dir, label=f"เก็บเฉพาะ {opt}"))
        w["bx"].on_click(lambda _, opt=k: _act(remove_model, opt, model_dir=model_dir, label=f"ลบ {opt}"))
        w["hbox"] = widgets.HBox(
            [w["info"], widgets.HBox([w["bd"], w["br"], w["bk"], w["bx"]], layout=_L(gap="5px"))],
            layout=_L(justify_content="space-between", align_items="center", padding="8px 10px", border="1px solid #334155", border_radius="8px", flex_flow="row wrap"),
        )
        return w

    def _build_rows() -> tuple:
        for k in ALL_PRESET_KEYS:
            rows.setdefault(k, _row_widgets(k))
        return tuple(rows[k]["hbox"] for k in ALL_PRESET_KEYS)

    def _apply_row_filter(*_) -> None:
        q = flt_in.value.strip().lower()
        cat = flt_dd.value
        hw_now = check_gpu_and_recommend(verbose=False)
        usable = float(plan_multi_gpu(0.0, hw_now.get("devices")).get("usable_gb", 0.0))
        for k in ALL_PRESET_KEYS:
            if k not in rows:
                continue
            cfg = MODEL_CATALOG[k]
            w = rows[k]
            tags = cfg.get("tags") or []
            show = True
            if q and not (q in k.lower() or q in str(cfg.get("desc", "")).lower() or q in str(cfg.get("repo_id", "")).lower()):
                show = False
            elif cat == "installed":
                show = bool(w["installed"])
            elif cat == "fit":
                show = (cfg.get("kind") != "image") and (usable > 0) and (float(cfg["approx_gb"]) <= usable * 1.15)
            elif cat == "obliterated":
                show = "obliterated" in tags
            elif cat == "unrestricted":
                show = "unrestricted" in tags
            elif cat == "rp":
                show = "rp" in tags
            elif cat == "heretic":
                show = "heretic" in tags
            elif cat == "image":
                show = cfg.get("kind") == "image"
            elif cat == "code":
                show = ("code" in tags) or ("security" in tags)
            w["hbox"].layout.display = "" if show else "none"

    flt_in.observe(_apply_row_filter, names="value")
    flt_dd.observe(_apply_row_filter, names="value")

    ui_lock = threading.RLock()

    _init_img_key = next((k for k in IMAGE_PRESET_KEYS if image_model_status(k, model_dir)["downloaded"]), IMAGE_PRESET_KEYS[0])
    img_dd = widgets.Dropdown(options=_img_dd_opts(), value=_init_img_key, description="โมเดลภาพ:", layout=_L(width="330px"))

    def _refresh():
        with ui_lock:
            _refresh_inner()

    def _refresh_inner():
        header.value = _header_html()
        cur_d = draft_dd.value; n_opts = _draft_opts(); draft_dd.options = n_opts
        if any(v == cur_d for _, v in n_opts):
            draft_dd.value = cur_d
        cur_im = img_dd.value; im_opts = _img_dd_opts(); img_dd.options = im_opts
        if any(v == cur_im for _, v in im_opts):
            img_dd.value = cur_im
        rag_stat.value = f"<b style='color:#38bdf8;'>📊 คลังความรู้ StrataRAG:</b> <b>{len(rag_engine.chunks)}</b> chunks (PDF, TXT, MD)"
        if not rows:
            rows_box.children = _build_rows()
        base = Path(model_dir)
        cnt_dirs = len([d for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]) if base.exists() else 0
        hw_now = check_gpu_and_recommend(verbose=False)
        gpu_plan_rows = plan_multi_gpu(0.0, hw_now.get("devices"))
        for k in ALL_PRESET_KEYS:
            cfg = MODEL_CATALOG[k]; w = rows[k]; sub = base / k; ex = sub.is_dir()
            stats = _dir_stats(sub) if ex else {"bytes": 0, "gguf": 0}
            gc, sz = int(stats["gguf"]), int(stats["bytes"])
            if cfg.get("kind") == "image":
                img_ready = ex and all((sub / rel).is_file() for rel in cfg.get("targets", ()))
                w["installed"] = img_ready
            else:
                w["installed"] = gc > 0
            mtag = " <span style='background:#3b0764;color:#e9d5ff;padding:1px 6px;border-radius:6px;font-size:10px;'>🚀 MTP</span>" if cfg.get("has_mtp") else ""
            _labels: list[str] = []
            for _prof, _slots in RECOMMENDED_PROFILES.items():
                for _slot, _key in _slots.items():
                    if _key == k:
                        _lbl = RECOMMENDED_LABELS.get(_slot, _slot)
                        if _lbl not in _labels:
                            _labels.append(_lbl)
            for _lbl in _labels:
                mtag += (
                    " <span style='background:#422006;color:#fcd34d;padding:1px 6px;border-radius:6px;"
                    f"font-size:10px;' title=' preset ที่แนะนำสำหรับ: {_lbl}'>✨ {_lbl}</span>"
                )
            tags = cfg.get("tags") or []
            if "obliterated" in tags:
                mtag += " <span style='background:#7c2d12;color:#fed7aa;padding:1px 6px;border-radius:6px;font-size:10px;'>💥 OBLITERATED</span>"
            if "unrestricted" in tags:
                mtag += " <span style='background:#064e3b;color:#6ee7b7;padding:1px 6px;border-radius:6px;font-size:10px;'>🔓 UNRESTRICTED</span>"
            if "security" in tags:
                mtag += " <span style='background:#450a0a;color:#fca5a5;padding:1px 6px;border-radius:6px;font-size:10px;'>🛡️ SEC</span>"
            if "code" in tags:
                mtag += " <span style='background:#0c4a6e;color:#7dd3fc;padding:1px 6px;border-radius:6px;font-size:10px;'>💻 CODE</span>"
            if "agent" in tags:
                mtag += " <span style='background:#14532d;color:#86efac;padding:1px 6px;border-radius:6px;font-size:10px;'>🤖 AGENT</span>"
            if "rp" in tags:
                mtag += " <span style='background:#3b0764;color:#f5d0fe;padding:1px 6px;border-radius:6px;font-size:10px;'>🎭 RP</span>"
            if "heretic" in tags:
                mtag += " <span style='background:#4c0519;color:#fda4af;padding:1px 6px;border-radius:6px;font-size:10px;'>🔥 HERETIC</span>"
            if cfg.get("kind") == "image":
                mtag += " <span style='background:#0c4a6e;color:#7dd3fc;padding:1px 6px;border-radius:6px;font-size:10px;'>🎨 IMAGE</span>"
            ready_now = w["installed"]
            bdg = f"<span style='color:#4ade80;font-size:11px;'>✅ ติดตั้งแล้ว ({_fmt_bytes(sz)})</span>" if ready_now else f"<span style='color:#94a3b8;font-size:11px;'>⬜ ยังไม่โหลด (~{cfg['approx_gb']}GB)</span>"
            fbd = "" if cfg.get("kind") == "image" else _fit_badge(float(cfg["approx_gb"]), gpu_plan_rows)
            extra_repos = sorted(set((cfg.get("sources") or {}).values()) - {cfg["repo_id"]})
            repo_note = f"<code>{cfg['repo_id']}</code>" + (f" + <code>{', '.join(extra_repos)}</code>" if extra_repos else "")
            w["info"].value = f"<div style='font-family:sans-serif;'><b style='color:#38bdf8;'>{k}</b>{mtag} {bdg} {fbd}<div style='font-size:11px;color:#cbd5e1;'>{cfg['desc']} ({repo_note})</div></div>"
            w["bd"].description = "📥 โหลดซ้ำ" if ready_now else "📥 โหลด"
            w["bd"].button_style = "" if ready_now else "primary"
            w["bd"].disabled = False
            w["br"].disabled = False
            w["bk"].disabled = not (ex and cnt_dirs > 1)
            w["bx"].disabled = not ex
        _apply_row_filter()

    spec_bar = widgets.VBox([
        widgets.HBox([spec_tb, dmax_sl, chk_kvq8, chk_moe, b_apply_spec], layout=_L(flex_flow="row wrap", align_items="center", gap="6px")),
        widgets.HBox([draft_dd, b_dl_draft, b_clr_draft, b_rm_draft, draft_in], layout=_L(flex_flow="row wrap", align_items="center", gap="6px", width="100%")),
        spec_hint,
    ], layout=_L(padding="8px 12px", border="1px solid #334155", border_radius="10px", margin="0 0 6px 0", gap="5px"))

    b_pick = _B("✨ ไม่รู้จะเลือกอันไหน", "", "190px")

    def _on_pick(_):
        if guide_card.value:
            guide_card.value = ""
            return
        lines = ["<div style='font-family:sans-serif;background:#0f172a;color:#f8fafc;border:1px solid #334155;"
                 "border-radius:8px;padding:10px;font-size:12px;line-height:1.6;margin-top:6px;'>"
                 "<b style='color:#fcd34d;'>✨ แนะนำให้เริ่มแบบนี้ (คลิกปุ่มซ้ำเพื่อพับเก็บ)</b><br>"]
        for prof, slots in RECOMMENDED_PROFILES.items():
            lines.append(f"<div style='margin-top:6px;'><b>{PROFILE_LABELS.get(prof, prof)}</b></div>")
            for slot, key in slots.items():
                c = MODEL_CATALOG.get(key, {})
                lines.append(
                    f"<b style='color:#38bdf8;'>{c.get('desc', key)}</b> — <code>{key}</code> "
                    f"(~{c.get('approx_gb', 0)} GB) → <span style='color:#fcd34d;'>{RECOMMENDED_LABELS.get(slot, slot)}</span><br>"
                )
        lines.append(
            "<div style='margin-top:8px;padding-top:6px;border-top:1px solid #334155;'>"
            "<b style='color:#fda4af;'>⚡ คู่มือจับคู่ Draft Model เร่งสปีด (Uncensored / Heretic — สลับให้อัตโนมัติตามตระกูลคำศัพท์)</b><br>"
            "• <b>Euryale / Anubis / Nevoria / Lumimaid-v0.2 (8B & 70B) / Lexi / Stheno</b> → โหลด <code>llama32-1b-uncensored</code> (~0.8 GB)<br>"
            "• <b>Skyfall 31B / Magnum 22B / Rocinante 12B / NemoMix / Cydonia / Venice / Dolphin-R1</b> → โหลด <code>ministral-3b-abliterated</code> (~1.9 GB)<br>"
            "• <b>Noromaid 20B & 8x7B / WizardLM 13B–30B</b> → โหลด <code>tinyllama-1.1b-heretic</code> (~0.7 GB)<br>"
            "• <b>Qwen3.8 OBLITERATED / Qwen3-8B Hivemind / Qwen3-14B / QwQ-32B</b> → โหลด <code>qwen3-14b-eagle3</code> หรือ <code>qwen3-0.6b-heretic</code> (~0.4 GB)<br>"
            "• <b>Gemma-4 31B Heretic / Gemma-4 E4B OBLITERATED / Muse-Glimmer 30B</b> → โหลด <code>gemma4-26b-eagle3</code> หรือ <code>gemma3-1b-heretic</code> (~0.8 GB)"
            "</div>"
        )
        lines.append("<span style='color:#94a3b8;'>เริ่มด้วยตัวแรกก่อน 1 ตัว แล้วค่อยโหลดเพิ่ม "
                     "(บน Kaggle เก็บลง /tmp ได้เกิน 100 GB โดยไม่กินโควตา /kaggle/working 20 GB)</span></div>")
        guide_card.value = "".join(lines)

    b_pick.on_click(_on_pick)

    b_live = _B("⏱ เรียลไทม์: เปิด", "success", "150px")
    live_out = widgets.HTML()
    tab1 = widgets.VBox([
        widgets.HBox([b_all, b_cache, b_stat, b_stop, b_cancel, chk_del, b_delall, b_ref], layout=_L(flex_flow="row wrap", gap="5px", align_items="center")),
        widgets.HBox([flt_in, flt_dd], layout=_L(flex_flow="row wrap", gap="6px", align_items="center", margin="4px 0")),
        dl_prog_out,
        action_notice,
        rows_box,
        widgets.HBox([b_pick, b_net, b_live, net_out], layout=_L(flex_flow="row wrap", gap="8px", align_items="center")),
        guide_card,
        live_out,
    ])

    # --- TAB 2: Chat + Roleplay + StrataRAG + Image Studio ---
    tpl_opts = [(v["label"], k) for k, v in SYSTEM_PROMPT_TEMPLATES.items()] + [
        ("✍️ กำหนดเอง (แก้ไขข้อความ System ได้อิสระ)", "custom")
    ]
    tpl_dd = widgets.Dropdown(options=tpl_opts, value="assistant", description="โหมดแชท:", layout=_L(width="365px"))
    q_dd = widgets.Dropdown(options=[("— สลับโมเดลด่วน —", "")] + [(k, k) for k in TEXT_PRESET_KEYS], value="", layout=_L(width="235px"))
    b_qsw = _B("⚡ สลับ", "warning", "90px")
    sys_in = widgets.Textarea(
        value=SYSTEM_PROMPT_TEMPLATES["assistant"]["prompt"],
        placeholder="กำหนดบทบาท บุคลิกตัวละคร ฉากหลัง หรือคำสั่ง System Prompt ได้หลายบรรทัด...",
        description="System:",
        rows=2,
        layout=_L(width="100%"),
    )
    chk_rag = widgets.Checkbox(value=True, description="📚 ใช้ StrataRAG", indent=False, layout=_L(width="140px"))
    topk_sl = widgets.IntSlider(value=3, min=1, max=8, description="Top-K:", layout=_L(width="175px"))
    temp_sl = widgets.FloatSlider(value=0.70, min=0.0, max=2.0, step=0.05, description="Temp:", layout=_L(width="175px"))
    minp_sl = widgets.FloatSlider(value=0.05, min=0.0, max=0.50, step=0.01, description="Min-P:", layout=_L(width="175px"))
    topp_sl = widgets.FloatSlider(value=0.95, min=0.10, max=1.0, step=0.05, description="Top-P:", layout=_L(width="175px"))
    reppen_sl = widgets.FloatSlider(value=1.08, min=1.00, max=1.50, step=0.02, description="RepPen:", layout=_L(width="180px"))
    dry_sl = widgets.FloatSlider(value=0.00, min=0.00, max=2.00, step=0.10, description="DRY:", layout=_L(width="170px"))
    max_sl = widgets.IntSlider(value=1024, min=128, max=4096, step=64, description="MaxTok:", layout=_L(width="205px"))

    _tpl_lock = {"busy": False}

    def _on_tpl_select(ch):
        if _tpl_lock["busy"] or ch["new"] == "custom":
            return
        preset = SYSTEM_PROMPT_TEMPLATES.get(ch["new"])
        if not preset:
            return
        _tpl_lock["busy"] = True
        try:
            sys_in.value = preset["prompt"]
            temp_sl.value = float(preset["temp"])
            minp_sl.value = float(preset["min_p"])
            topp_sl.value = float(preset["top_p"])
            reppen_sl.value = float(preset["repeat_penalty"])
            dry_sl.value = float(preset["dry"])
            max_sl.value = int(preset["max_tokens"])
        finally:
            _tpl_lock["busy"] = False

    def _on_sys_manual_edit(_):
        if _tpl_lock["busy"]:
            return
        cur = sys_in.value.strip()
        matched = next((k for k, v in SYSTEM_PROMPT_TEMPLATES.items() if v["prompt"].strip() == cur), "custom")
        if tpl_dd.value != matched:
            _tpl_lock["busy"] = True
            try:
                tpl_dd.value = matched
            finally:
                _tpl_lock["busy"] = False

    tpl_dd.observe(_on_tpl_select, names="value")
    sys_in.observe(_on_sys_manual_edit, names="value")

    def _sampler_extra_body(min_p: float, rep_pen: float, dry_mult: float) -> dict:
        extra: dict = {"min_p": round(float(min_p), 3), "repeat_penalty": round(float(rep_pen), 3)}
        if float(dry_mult) > 0.0:
            extra["dry_multiplier"] = round(float(dry_mult), 2)
        return extra

    feed = widgets.VBox([widgets.HTML("<div style='color:#94a3b8;text-align:center;padding:10px;'>💬 เลือกโหมดแชท/Roleplay ด้านบน แล้วพิมพ์ข้อความด้านล่างได้เลย</div>")], layout=_L(border="1px solid #334155", border_radius="8px", padding="10px", min_height="240px", max_height="420px", overflow_y="auto", gap="8px"))
    u_in = widgets.Textarea(placeholder="พิมพ์ข้อความแชท / บทบาทตัวละคร... (หรือ /img คำบรรยาย --neg blurry เพื่อสร้างภาพ)", rows=3, layout=_L(flex="1 1 auto"))
    b_send = _B("📨 ส่งข้อความ", "success", "120px", height="66px")
    b_regen = _B("🔁 สุ่มตอบใหม่", "info", "110px", height="66px")
    b_clr = _B("🗑️ ล้างแชท", "danger", "95px", height="66px")

    img_sizes = [
        ("📐 1:1 จัตุรัส (1024×1024)", "1024x1024"),
        ("📱 2:3 แนวตั้ง (832×1216)", "832x1216"),
        ("🖼️ 3:2 แนวนอน (1216×832)", "1216x832"),
        ("📲 9:16 มือถือ (768×1344)", "768x1344"),
        ("🎬 16:9 จอกว้าง (1344×768)", "1344x768"),
        ("⚡ 1:1 เร็ว/ประหยัด (768×768)", "768x768"),
        ("⚙️ กำหนดเอง (กว้าง × สูง)", "custom"),
    ]
    img_size_dd = widgets.Dropdown(options=img_sizes, value="1024x1024", description="ขนาดภาพ:", layout=_L(width="245px"))
    img_w_in = widgets.BoundedIntText(value=1024, min=512, max=1536, step=64, description="กว้าง:", style={"description_width": "38px"}, layout=_L(width="120px"))
    img_h_in = widgets.BoundedIntText(value=1024, min=512, max=1536, step=64, description="สูง:", style={"description_width": "28px"}, layout=_L(width="112px"))
    img_steps_in = widgets.BoundedIntText(value=0, min=0, max=50, step=1, description="Steps(0=Auto):", style={"description_width": "85px"}, layout=_L(width="150px"))
    img_seed_in = widgets.IntText(value=-1, description="Seed(-1=สุ่ม):", style={"description_width": "82px"}, layout=_L(width="165px"))
    chk_img_fast = widgets.Checkbox(value=True, description="⚡ FastCache (เร่งรุ่น 20+ steps)", indent=False, layout=_L(width="205px"))
    neg_in = widgets.Text(
        value="",
        placeholder="สิ่งที่ไม่ต้องการในภาพ เช่น blurry, low quality, deformed, watermark (เว้นว่างได้)",
        description="Negative:",
        layout=_L(flex="1 1 260px"),
    )
    b_img = _B("🖼️ สร้างภาพจากข้อความ", "warning", "195px")
    b_gal = _B("🖼️ รูปที่สร้างไว้", "", "140px")

    _sz_lock = {"busy": False}

    def _on_img_size_preset(ch):
        if _sz_lock["busy"] or ch["new"] == "custom":
            return
        ws, hs = str(ch["new"]).split("x", 1)
        _sz_lock["busy"] = True
        try:
            img_w_in.value = int(ws)
            img_h_in.value = int(hs)
        finally:
            _sz_lock["busy"] = False

    def _on_img_wh_edit(_):
        if _sz_lock["busy"]:
            return
        cur_wh = f"{int(img_w_in.value)}x{int(img_h_in.value)}"
        preset_vals = {v for _, v in img_sizes if v != "custom"}
        _sz_lock["busy"] = True
        try:
            img_size_dd.value = cur_wh if cur_wh in preset_vals else "custom"
        finally:
            _sz_lock["busy"] = False

    img_size_dd.observe(_on_img_size_preset, names="value")
    img_w_in.observe(_on_img_wh_edit, names="value")
    img_h_in.observe(_on_img_wh_edit, names="value")

    b_qsw.on_click(lambda _: q_dd.value and _act(switch_strata_model if (Path(model_dir) / q_dd.value).exists() else install_model, q_dd.value, model_dir=model_dir, port=port, active=q_dd.value, label=f"สลับเป็น {q_dd.value}", **({} if (Path(model_dir) / q_dd.value).exists() else {"auto_start_server": True})))
    b_clr.on_click(lambda _: (history.clear(), setattr(feed, "children", (widgets.HTML("<div style='color:#94a3b8;text-align:center;'>🧹 ล้างแชทแล้ว</div>"),))))

    def _img_card(res: dict, prompt: str, negative: str = "", width: int | None = None, height: int | None = None) -> widgets.HTML:
        try:
            uri = image_data_uri(res["path"])
        except OSError as e:
            return widgets.HTML(f"<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;'>❌ อ่านไฟล์ภาพไม่ได้: {html_lib.escape(str(e))}</div>")
        name = Path(res["path"]).name
        dim_badge = f" • 📐 {width}×{height}" if (width and height) else ""
        fc_badge = " • ⚡FastCache" if res.get("fast_cache") else ""
        neg_line = f"<br>🚫 Negative: {html_lib.escape(negative[:160])}" if negative else ""
        return widgets.HTML(
            "<div style='background:#1e293b;color:#f8fafc;padding:8px;border-radius:10px;border:1px solid #334155;'>"
            f"<div style='font-size:11px;color:#fbbf24;margin-bottom:4px;'>🖼️ {html_lib.escape(res.get('key', ''))}"
            f"{dim_badge}{fc_badge} • {res.get('seconds', 0)}s • seed {res.get('seed', '-')}</div>"
            f"<img src='{uri}' style='max-width:100%;border-radius:8px;display:block;' alt='{html_lib.escape(prompt[:80])}'/>"
            f"<div style='font-size:11px;color:#94a3b8;margin-top:4px;'>📝 {html_lib.escape(prompt[:200])}{neg_line}<br>💾 {html_lib.escape(name)}</div>"
            "</div>"
        )

    def _parse_img_prompt(raw_prompt: str) -> tuple[str, str]:
        base_neg = neg_in.value.strip()
        if " --neg " in raw_prompt:
            pos_part, inline_neg = raw_prompt.split(" --neg ", 1)
            merged_neg = ", ".join(x for x in (base_neg, inline_neg.strip()) if x)
            return pos_part.strip(), merged_neg
        return raw_prompt.strip(), base_neg

    def _start_image_job(prompt: str):
        key = img_dd.value
        clean_p, neg_txt = _parse_img_prompt(prompt)
        w = max(512, min(1536, (int(img_w_in.value) // 16) * 16))
        h = max(512, min(1536, (int(img_h_in.value) // 16) * 16))
        ov_steps = int(img_steps_in.value) if int(img_steps_in.value) > 0 else None
        ov_seed = int(img_seed_in.value) if int(img_seed_in.value) >= 0 else None
        use_fc = bool(chk_img_fast.value)
        neg_tag = f"<div style='font-size:11px;opacity:0.85;'>🚫 {html_lib.escape(neg_txt[:120])}</div>" if neg_txt else ""
        ub = widgets.HTML(
            "<div style='text-align:right;'><span style='display:inline-block;background:#b45309;color:#fff;"
            f"padding:8px 12px;border-radius:10px;max-width:80%;text-align:left;'>🎨 {html_lib.escape(clean_p)}"
            f" <span style='font-size:11px;opacity:0.85;'>({w}×{h})</span>{neg_tag}</span></div>"
        )
        bb = widgets.HTML(
            "<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #334155;'>"
            f"⏳ เริ่มเตรียมงานวาดภาพ <b>{html_lib.escape(key)}</b> ({w}×{h})…</div>"
        )
        feed.children = tuple([*feed.children, ub, bb])
        b_img.disabled = True

        def _upd_prog(msg: str) -> None:
            bb.value = (
                "<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #38bdf8;'>"
                f"{html_lib.escape(msg)}</div>"
            )
            c_stat.value = f"<span style='color:#38bdf8;font-size:12px;'>{html_lib.escape(msg)}</span>"

        def _worker():
            try:
                if not image_model_status(key, model_dir)["downloaded"]:
                    approx = MODEL_CATALOG.get(key, {}).get("approx_gb", 14.6)
                    _upd_prog(f"📥 กำลังดาวน์โหลดโมเดลภาพ {key} ครั้งแรก (~{approx} GB) — ดูความเร็วสดได้ที่แท็บ 1…")
                    install_image_model(key, model_dir=model_dir)
                    _invalidate_dir_cache()
                    _refresh()
                res = generate_image(
                    key, clean_p, negative=neg_txt, width=w, height=h,
                    steps=ov_steps, seed=ov_seed, port=COMFY_PORT, fast_cache=use_fc,
                    on_progress=_upd_prog,
                )
                bb.value = ""
                feed.children = tuple([*(wd for wd in feed.children if wd is not bb), _img_card(res, clean_p, neg_txt, w, h)])
                fc_lbl = " • ⚡FastCache" if res.get("fast_cache") else ""
                c_stat.value = f"<span style='color:#4ade80;font-size:12px;'>✅ สร้างภาพ {w}×{h} เสร็จใน {res['seconds']}s{fc_lbl} • seed {res.get('seed', '-')} • {Path(res['path']).name}</span>"
            except Exception as e:
                bb.value = f"<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;white-space:pre-wrap;font-size:12px;'>❌ {html_lib.escape(str(e))}</div>"
                c_stat.value = f"<span style='color:#f87171;font-size:12px;'>❌ สร้างภาพไม่สำเร็จ: {html_lib.escape(str(e)[:160])}</span>"
            finally:
                b_img.disabled = False
                header.value = _header_html()

        if _use_sync_mode():
            _worker()
        else:
            threading.Thread(target=_worker, daemon=True, name="strata-imggen").start()

    def _on_img(_):
        q = u_in.value.strip()
        if not q:
            c_stat.value = "<span style='color:#fbbf24;font-size:12px;'>ℹ️ พิมพ์คำอธิบายภาพในช่องข้อความก่อน</span>"
            return
        u_in.value = ""
        _start_image_job(q)

    def _on_gallery(_):
        files = list_generated(8)
        if not files:
            c_stat.value = "<span style='color:#94a3b8;font-size:12px;'>ℹ️ ยังไม่มีรูปที่สร้างไว้</span>"
            return
        feed.children = tuple([*feed.children, *(_img_card({"path": f, "key": "gallery", "seconds": 0, "seed": "-"}, Path(f).name) for f in files)])

    def _stream_chat_turn(q: str, append_user_bubble: bool = True):
        is_on, active_mid = _check_server_online(port, ttl=0.5)
        if not is_on:
            c_stat.value = f"<span style='color:#f87171;'>❌ Server พอร์ต {port} ยังไม่เปิด! กด 🚀 รัน & แชท ที่แท็บแรกก่อนครับ</span>"
            return
        aug_q, docs = rag_engine.build_augmented_prompt(q, top_k=int(topk_sl.value)) if (chk_rag.value and rag_engine.chunks) else (q, [])
        bb = widgets.HTML("<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;'>⏳ กำลังคิด...</div>")
        if append_user_bubble:
            ub = widgets.HTML(f"<div style='text-align:right;'><span style='display:inline-block;background:#1d4ed8;color:#fff;padding:8px 12px;border-radius:10px;max-width:80%;text-align:left;'>👤 {html_lib.escape(q)}</span></div>")
            feed.children = tuple([*(feed.children if history else []), ub, bb])
        else:
            feed.children = tuple([*feed.children, bb])

        def _chat_worker():
            b_send.disabled = True; b_regen.disabled = True
            t0, buf, n = time.perf_counter(), [], 0; stag = html_lib.escape(SPEC_STATE["active_label"])
            try:
                _ctx = max(2048, int(SPEC_STATE.get("ctx_size", 4096)))
                _mtok = max(128, min(int(max_sl.value), _ctx - 1024))
                st = get_openai_client(port=port).chat.completions.create(
                    model=active_mid,
                    messages=[{"role": "system", "content": sys_in.value}, *history, {"role": "user", "content": aug_q}],
                    temperature=float(temp_sl.value),
                    top_p=float(topp_sl.value),
                    max_tokens=_mtok,
                    extra_body=_sampler_extra_body(minp_sl.value, reppen_sl.value, dry_sl.value),
                    stream=True,
                )  # type: ignore[arg-type]
                last_draw = 0.0
                for ch in st:
                    c0 = next(iter(ch.choices), None); d = c0.delta.content if c0 else None
                    if not d:
                        continue
                    buf.append(d); n += 1
                    now = time.perf_counter()
                    if (now - last_draw) < 0.1 and not d.endswith(("\n", "。")):
                        continue
                    last_draw = now
                    spd = n / max(0.01, now - t0)
                    bb.value = f"<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #334155;'><div style='font-size:11px;color:#38bdf8;'>🤖 {active_mid} • <span style='color:#e9d5ff;'>{stag}</span> • <b>{spd:.1f} tok/s</b></div>{_fmt_reply_html(''.join(buf), docs)}</div>"
                el = max(0.01, time.perf_counter() - t0)
                bb.value = f"<div style='background:#1e293b;color:#f8fafc;padding:10px;border-radius:10px;border:1px solid #334155;'><div style='font-size:11px;color:#38bdf8;'>🤖 {active_mid} • <span style='color:#e9d5ff;'>{stag}</span> • <b>{n / el:.1f} tok/s</b></div>{_fmt_reply_html(''.join(buf), docs)}</div>"
                history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": "".join(buf)}])
                _budget_chars = max(1024, _ctx - _mtok - 512) * 3
                while len(history) > 2 and sum(len(m["content"]) for m in history) > _budget_chars:
                    history.pop(0)
                    if history and history[0]["role"] == "assistant":
                        history.pop(0)
                c_stat.value = f"<span style='color:#4ade80;font-size:12px;'>✅ เสร็จใน {el:.2f}s ({n / el:.1f} tok/s • {stag})</span>"
            except Exception as e:
                bb.value = f"<div style='background:#450a0a;color:#fecaca;padding:10px;border-radius:8px;'>❌ {html_lib.escape(str(e))}</div>"
            finally:
                b_send.disabled = False; b_regen.disabled = False; header.value = _header_html()

        if _use_sync_mode():
            _chat_worker()
        else:
            threading.Thread(target=_chat_worker, daemon=True, name="strata-chat").start()

    def _send(_):
        q = u_in.value.strip()
        if not q:
            return
        if q.lower().startswith("/img "):
            u_in.value = ""
            _start_image_job(q[5:].strip())
            return
        u_in.value = ""
        _stream_chat_turn(q, append_user_bubble=True)

    def _on_regen(_):
        if len(history) < 2 or history[-1]["role"] != "assistant":
            c_stat.value = "<span style='color:#fbbf24;font-size:12px;'>ℹ️ ยังไม่มีคำตอบล่าสุดให้สุ่มใหม่</span>"
            return
        history.pop()
        last_q = history.pop()["content"]
        if feed.children:
            feed.children = tuple(feed.children[:-1])
        _stream_chat_turn(last_q, append_user_bubble=False)

    b_send.on_click(_send)
    b_regen.on_click(_on_regen)
    b_img.on_click(_on_img)
    b_gal.on_click(_on_gallery)
    tab2 = widgets.VBox([
        widgets.HBox([tpl_dd, q_dd, b_qsw], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")),
        sys_in,
        widgets.HBox([chk_rag, topk_sl, temp_sl, minp_sl, topp_sl, reppen_sl, dry_sl, max_sl], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")),
        feed,
        widgets.HBox([u_in, b_send, b_regen, b_clr], layout=_L(gap="6px")),
        widgets.HBox([img_dd, img_size_dd, img_w_in, img_h_in, img_steps_in, img_seed_in], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")),
        widgets.HBox([chk_img_fast, neg_in, b_img, b_gal], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")),
        c_stat,
    ])

    # --- TAB 3: PDF & Text StrataRAG ---
    uploader = widgets.FileUpload(accept=".pdf,.txt,.md,.csv,.json,.py", multiple=True, description="📤 อัปโหลดไฟล์")
    b_up, b_clrrag = _B("📥 นำไฟล์เข้า RAG", "success", "165px"), _B("🗑️ ล้างคลัง RAG", "danger", "130px")
    chk_sz = widgets.IntSlider(value=500, min=200, max=1500, step=50, description="Chunk:", layout=_L(width="205px"))
    fp_in = widgets.Text(placeholder="/content/doc.pdf หรือ /content/*.pdf", description="พาธไฟล์:", layout=_L(flex="1 1 260px"))
    b_fp, b_txt, b_tq = _B("📂 โหลดจากพาธ", "info", "130px"), _B("➕ เพิ่มข้อความ", "primary", "130px"), _B("🔍 ทดสอบค้นหา", "", "125px")
    b_save_rag, b_load_rag = _B("💾 บันทึกคลัง", "info", "130px"), _B("📂 โหลดคลังเดิม", "info", "140px")
    rag_path_in = widgets.Text(value=DEFAULT_RAG_PATH, description="ไฟล์คลัง:", layout=_L(flex="1 1 260px"))
    src_in = widgets.Text(value="note", description="แหล่งที่มา:", layout=_L(width="210px"))
    txt_in = widgets.Textarea(placeholder="วางข้อความ/บทความที่นี่...", rows=4, layout=_L(width="100%"))
    tq_in = widgets.Text(placeholder="คำค้นหาทดสอบ...", description="ค้นหา:", layout=_L(flex="1 1 240px"))

    def _ingest_up(_):
        items = list(uploader.value.values()) if isinstance(uploader.value, dict) else list(uploader.value)
        if not items:
            rag_out.value = "<span style='color:#fbbf24;'>⚠️ เลือกไฟล์ก่อนครับ</span>"; return
        for it in items:
            nm = it.get("name") or it.get("metadata", {}).get("name", "upload.pdf")
            raw = it.get("content", b""); rb = raw.tobytes() if hasattr(raw, "tobytes") else bytes(raw)
            rag_engine.add_pdf_bytes(rb, filename=nm, chunk_size=int(chk_sz.value)) if nm.lower().endswith(".pdf") else rag_engine.add_documents([{"source": nm, "text": rb.decode("utf-8", errors="ignore")}], chunk_size=int(chk_sz.value))
        _refresh(); rag_out.value = f"<span style='color:#4ade80;'>✅ นำเข้า {len(items)} ไฟล์สำเร็จ! ({len(rag_engine.chunks)} chunks)</span>"

    def _ingest_fp(_):
        pat = fp_in.value.strip()
        files = [Path(p) for p in (glob.glob(pat) if any(c in pat for c in "*?[]") else [pat]) if Path(p).is_file()] if pat else []
        if not files:
            rag_out.value = "<span style='color:#f87171;'>❌ ไม่พบไฟล์</span>"; return
        rag_engine.add_files(*files, chunk_size=int(chk_sz.value)); _refresh()
        rag_out.value = f"<span style='color:#4ade80;'>✅ โหลด {len(files)} ไฟล์เรียบร้อย!</span>"

    def _add_note(_):
        if txt_in.value.strip():
            rag_engine.add_documents([{"source": src_in.value or "note", "text": txt_in.value.strip()}], chunk_size=int(chk_sz.value))
            txt_in.value = ""; _refresh()

    def _save_rag(_):
        try:
            p = rag_engine.save(rag_path_in.value.strip() or DEFAULT_RAG_PATH)
            rag_out.value = f"<span style='color:#4ade80;'>💾 บันทึกแล้ว: {html_lib.escape(p)} ({len(rag_engine.chunks)} chunks)</span>"
        except Exception as e:
            rag_out.value = f"<span style='color:#f87171;'>❌ {html_lib.escape(str(e))}</span>"

    def _load_rag(_):
        try:
            n = rag_engine.load(rag_path_in.value.strip() or DEFAULT_RAG_PATH)
            _refresh()
            rag_out.value = f"<span style='color:#4ade80;'>📂 โหลดคลังเดิม {n} chunks เรียบร้อย</span>"
        except Exception as e:
            rag_out.value = f"<span style='color:#f87171;'>❌ {html_lib.escape(str(e))}</span>"

    b_up.on_click(_ingest_up); b_fp.on_click(_ingest_fp); b_txt.on_click(_add_note)
    b_save_rag.on_click(_save_rag); b_load_rag.on_click(_load_rag)
    b_clrrag.on_click(lambda _: (rag_engine.clear(), _refresh(), setattr(rag_out, "value", "<span style='color:#fbbf24;'>🧹 ล้างคลัง RAG แล้ว</span>")))
    b_tq.on_click(lambda _: setattr(rag_out, "value", "".join(f"<div style='background:#1e293b;color:#f8fafc;padding:6px 10px;border-radius:6px;margin-top:4px;font-size:12px;'><b style='color:#38bdf8;'>#{h['rank']} {html_lib.escape(h['source'])}</b> ({h['score']})<br>{html_lib.escape(h['text'])}</div>" for h in rag_engine.search(tq_in.value.strip(), top_k=int(topk_sl.value))) or "<span style='color:#fbbf24;'>ไม่พบข้อมูลที่เกี่ยวข้อง (ลองคำค้นอื่น)</span>"))

    tab3 = widgets.VBox([rag_stat, widgets.HBox([uploader, b_up, chk_sz, b_clrrag], layout=_L(flex_flow="row wrap", gap="8px", align_items="center")), widgets.HBox([fp_in, b_fp], layout=_L(gap="6px")), widgets.HBox([src_in, b_txt], layout=_L(gap="6px")), txt_in, widgets.HBox([tq_in, b_tq], layout=_L(gap="6px")), widgets.HBox([rag_path_in, b_save_rag, b_load_rag], layout=_L(flex_flow="row wrap", gap="6px", align_items="center")), rag_out], layout=_L(gap="6px"))

    tabs = widgets.Tab(children=[tab1, tab2, tab3])
    for idx, ttl in enumerate([
        f"📦 1. โมเดล & ดิสก์ ({len(ALL_PRESET_KEYS)} ตัว)",
        "💬 2. แชท + RP + ภาพ",
        "📚 3. คลัง PDF RAG",
    ]):
        tabs.set_title(idx, ttl)
    live_ref = None

    def _tick(n: int) -> None:
        header.value = _header_html()
        dl_prog_out.value = _render_fetch_banner(get_download_progress(), state["job"])
        lines = recent_log(12)
        tail: list[str] = []
        try:
            if LLAMA_LOG_FILE.exists():
                tail = LLAMA_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()[-4:]
        except OSError:
            tail = []
        rows_html = "".join(f"<div>{html_lib.escape(x)}</div>" for x in [*lines, *tail][-14:])
        live_out.value = (
            "<div style='background:#0b1220;color:#94a3b8;font-family:ui-monospace,SFMono-Regular,monospace;"
            "font-size:11px;line-height:1.5;padding:8px 10px;border-radius:8px;border:1px solid #334155;"
            "max-height:180px;overflow:auto;'>"
            f"<b style='color:#38bdf8;'>⏱ เรียลไทม์ (ทุก {live_ref.interval:.1f}s • tick {n})</b>"
            f"{rows_html or '<div>(พร้อมรับคำสั่ง — กดปุ่มใดก็ได้ทันที)</div>'}</div>"
        )
        if n % 4 == 0:
            _refresh()

    live_ref = LiveRefresher(_tick, interval=1.5)

    def _toggle_live(_):
        if live_ref.running:
            live_ref.stop()
            b_live.description = "⏸ เรียลไทม์: ปิด"
            b_live.button_style = ""
        else:
            live_ref.start()
            _tick(live_ref.ticks)
            b_live.description = "⏱ เรียลไทม์: เปิด"
            b_live.button_style = "success"

    b_live.on_click(_toggle_live)

    _refresh()
    _tick(1)
    root = widgets.VBox([header, spec_bar, tabs])
    display(root)
    LIVE_REFS.append(live_ref)
    if live:
        live_ref.start()
    return root
