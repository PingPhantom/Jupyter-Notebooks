"""แผงควบคุม (ipywidgets) สำหรับแพลตฟอร์มที่ไม่มี Colab RPC — Kaggle / JupyterLab

หน้าจอหลักของโปรเจกต์เป็น HTML5 (เหมือน Colab ทุกพิกเซล) แต่บน Kaggle ไม่มีช่องทางให้ JS
ในหน้าเว็บเรียกกลับเข้า kernel ได้ (``google.colab.kernel.invokeFunction`` มีแค่บน Colab)
แผงนี้จึงทำหน้าที่เป็น "รีโมต" ที่วางไว้ใต้หน้าจอ — เรียก action ชุดเดียวกับที่ปุ่มใน HTML เรียก
(``_handle_action`` ของเซลล์ 2 และ ``_cf_rpc`` ของเซลล์ 3) เพื่อให้ใช้งานได้จริงโดยไม่ต้องใช้ tunnel

หลักการ: ไม่พึ่งไลบรารีใหม่ ใช้ ``ipywidgets`` ที่ notebook มีอยู่แล้ว และทุก action ถูกเรียกผ่าน
dispatcher ที่ส่งเข้ามา → แผงนี้ไม่รู้จักธุรกิจภายใน จึงใช้ซ้ำได้ทั้งสองเซลล์
"""

from __future__ import annotations

import base64
import contextlib
import html as _html
import threading
from typing import Any, Callable

def _widgets():
    """import ipywidgets แบบ lazy (คืน None ถ้าไม่มี เพื่อให้ผู้เรียก fallback ได้)"""
    try:
        import ipywidgets as widgets  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    return widgets


def _esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""))


def _lab(model: dict) -> str:
    bits = [str(model.get("key", "?"))]
    if model.get("installed"):
        bits.append("✅")
    if model.get("size_txt"):
        bits.append(str(model["size_txt"]))
    if model.get("fit_label"):
        bits.append(str(model["fit_label"]))
    return " · ".join(bits)


def _kind(models: list[dict], kinds: tuple[str, ...]) -> list[dict]:
    return [m for m in models if str(m.get("kind") or "text") in kinds]


class _Bridge:
    """ห่อ dispatcher ให้ปลอดภัย + เก็บ state ล่าสุดไว้ให้วิดเจ็ตอ่าน"""

    def __init__(self, dispatch: Callable[[str, dict], Any]) -> None:
        self._dispatch = dispatch
        self.state: dict = {}
        self.last_error = ""
        self._lock = threading.Lock()

    def call(self, act: str, payload: dict | None = None) -> dict:
        try:
            out = self._dispatch(act, payload or {})
            if isinstance(out, dict) and act == "poll":
                with self._lock:
                    self.state = out
            return out if isinstance(out, dict) else {"ok": True, "raw": out}
        except Exception as exc:  # noqa: BLE001 — ต้องไม่ทำให้เซลล์พัง
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {"ok": False, "error": self.last_error}

    def refresh(self) -> dict:
        return self.call("poll", {})

    def models(self, kinds: tuple[str, ...] = ("text", "image", "lora")) -> list[dict]:
        if not self.state:
            self.refresh()
        return _kind(list(self.state.get("models") or []), kinds)


def _pal() -> dict:
    """สีชุดเดียวกับหน้าจอ HTML ของ Colab (strata studio)"""
    return {
        "bg": "#0b1120", "panel": "#0f172a", "line": "#334155", "chip": "#1e293b",
        "text": "#f8fafc", "dim": "#94a3b8", "inp_line": "#475569",
        "pri": "#2563eb", "suc": "#059669", "wrn": "#d97706", "dng": "#dc2626", "act": "#7c3aed",
        "me": "#1e3a8a", "me_line": "#3b82f6",
    }


def _f(font: str = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;") -> str:
    return font


def _chip(text: str, color: str = "#e2e8f0", bg: str = "#1e293b", line: str = "#334155") -> str:
    return (f"<span style=\"display:inline-block;padding:3px 10px;border-radius:99px;background:{bg};"
            f"border:1px solid {line};color:{color};font-size:12.5px\">{text}</span>")


def _card(inner: str, pal: dict, pad: str = "11px 13px", extra: str = "") -> str:
    return (f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:10px;"
            f"padding:{pad};{extra}\">{inner}</div>")


def _models_html(models: list[dict], query: str = "", flt: str = "all", pal: dict | None = None) -> str:
    """รายการโมเดลแบบการ์ด — เลียนแบบ .s-mrow ของหน้าจอ Colab"""
    pal = pal or _pal()
    q = (query or "").strip().lower()
    rows = []
    for m in models:
        kind = str(m.get("kind") or "text")
        tags = [str(t).lower() for t in (m.get("tags") or [])]
        if flt == "installed" and not m.get("installed"):
            continue
        if flt == "fit" and str(m.get("fit")) not in ("ok", "tight", "good"):
            continue
        if flt in ("lora", "image", "text") and kind != flt:
            continue
        if flt == "obliterated" and not any("obliter" in t for t in tags):
            continue
        if flt == "unrestricted" and not any(("unrestrict" in t or "uncensor" in t) for t in tags):
            continue
        hay = " ".join([str(m.get("key", "")), str(m.get("desc", "")), " ".join(tags)]).lower()
        if q and q not in hay:
            continue
        ok = bool(m.get("installed"))
        badge = "✅" if ok else "⬜"
        kind_bg = {"text": "#1e3a8a", "image": "#065f46", "lora": "#7c2d12"}.get(kind, "#1e293b")
        tag_html = f"<span style=\"display:inline-block;padding:1px 6px;border-radius:6px;background:{kind_bg};font-size:10.5px;margin-left:4px\">{kind}</span>"
        for t in tags[:3]:
            tag_html += f"<span style=\"display:inline-block;padding:1px 6px;border-radius:6px;background:#1e293b;border:1px solid #334155;font-size:10.5px;margin-left:4px\">{_esc(t)}</span>"
        rows.append(
            f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:9px;"
            f"padding:9px 12px;margin-bottom:6px\">"
            f"<div style=\"display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap\">"
            f"<div style=\"min-width:200px;flex:1 1 60%\"><b style=\"color:{pal['text']};font-size:13px\">{badge} {_esc(str(m.get('key', '')))}</b>{tag_html}"
            f"<div style=\"color:{pal['dim']};font-size:12px;margin-top:3px\">{_esc(str(m.get('desc', ''))[:120])}</div></div>"
            f"<div style=\"color:{pal['dim']};font-size:12px;text-align:right;white-space:nowrap\">"
            f"{_esc(str(m.get('size_txt', '')))} · {_esc(str(m.get('fit_label', '')))}</div></div></div>"
        )
    if not rows:
        rows.append(f"<div style=\"color:{pal['dim']};font-size:12.5px;padding:6px 2px\">ไม่พบโมเดลที่ตรงกับเงื่อนไข</div>")
    return (f"<div style=\"max-height:330px;overflow-y:auto;padding-right:4px\">{''.join(rows)}</div>")


def _bubble(text: str, who: str, meta: str = "", pal: dict | None = None) -> str:
    pal = pal or _pal()
    if who == "me":
        style = f"align-self:flex-end;background:{pal['me']};border:1px solid {pal['me_line']};color:#fff;"
    else:
        style = f"align-self:flex-start;background:{pal['chip']};border:1px solid {pal['line']};color:{pal['text']};"
    meta_html = f"<div style=\"color:{pal['dim']};font-size:11.5px;margin-top:6px\">{_esc(meta)}</div>" if meta else ""
    return (f"<div style=\"{style}padding:9px 13px;border-radius:11px;max-width:88%;white-space:pre-wrap;"
            f"font-size:13.5px;line-height:1.55;margin:5px 0\">{_esc(text)}{meta_html}</div>")


def _set_options(widget: Any, opts: list[tuple[str, str]]) -> None:
    """กำหนด options ให้ Dropdown/Select แล้วรับประกันว่ามีค่าที่ถูกเลือกอยู่เสมอ

    (ipywidgets ไม่เลือกค่าแรกให้เมื่อ assign ``options`` ทีหลัง → ปุ่มจะอ่านค่าได้ None)
    """
    widget.options = opts
    keys = [o[1] for o in opts]
    with contextlib.suppress(Exception):
        if widget.value not in keys:
            widget.value = keys[0] if keys else ""


def _model_matches(m: dict, query: str = "", flt: str = "all") -> bool:
    """เงื่อนไขกรองเดียวกับ _models_html (ใช้ให้ Select ตรงกับรายการที่เห็น)"""
    kind = str(m.get("kind") or "text")
    tags = [str(t).lower() for t in (m.get("tags") or [])]
    if flt == "installed" and not m.get("installed"):
        return False
    if flt == "fit" and str(m.get("fit")) not in ("ok", "tight", "good"):
        return False
    if flt in ("lora", "image", "text") and kind != flt:
        return False
    if flt == "obliterated" and not any("obliter" in t for t in tags):
        return False
    if flt == "unrestricted" and not any(("unrestrict" in t or "uncensor" in t) for t in tags):
        return False
    q = (query or "").strip().lower()
    hay = " ".join([str(m.get("key", "")), str(m.get("desc", "")), " ".join(tags)]).lower()
    return not q or q in hay


def studio_console(
    dispatch: Callable[[str, dict], Any],
    *,
    title: str = "🎛️ Strata Studio",
    model_dir: str | None = None,
    templates: dict | None = None,
) -> Any:
    """แผงควบคุมหน้าตาเดียวกับหน้าจอ HTML ของ Colab (3 แท็บ: โมเดล / แชท / RAG)

    บนแพลตฟอร์มที่บล็อกสคริปต์ใน output (Kaggle) หน้าจอ HTML สั่ง kernel ไม่ได้
    แผงนี้จึงทำหน้าที่แทน โดยคงลำดับแท็บ สี และป้ายกำกับชุดเดียวกับหน้าจอ Colab
    """
    widgets = _widgets()
    if widgets is None:
        return None

    pal = _pal()
    bridge = _Bridge(dispatch)
    state = bridge.refresh()
    all_models = bridge.models()
    tpls = dict(templates or {})
    base_font = _f()

    def _btn(caption: str, kind: str = "", tip: str = "") -> Any:
        style: dict = {"font_weight": "600", "font_size": "12.5px"}
        if kind in ("pri", "suc", "wrn", "dng", "act"):
            style.update({"button_color": pal[kind], "text_color": "#f8fafc"})
        return widgets.Button(
            description=caption, tooltip=tip or caption,
            layout=widgets.Layout(min_width="118px", height="34px"), style=style,
        )

    # ---------- header (การ์ดเดียวกับ Colab) ----------
    header = widgets.HTML("")

    def _render_header() -> None:
        st = dict(bridge.state) or state
        models_now = list(st.get("models") or [])
        n_models = len(models_now) or len(all_models)
        plat = str(st.get("platform") or "notebook")
        disk = st.get("disk") or {}
        chunks = [
            _chip(f"🧭 {_esc(plat)}"),
            _chip("🟢 ออนไลน์" if st.get("online") else "🔴 ออฟไลน์",
                  "#4ade80" if st.get("online") else "#f87171"),
            _chip(f"🚀 {_esc(str(st.get('active_model') or 'ยังไม่รัน'))}"),
            _chip(_esc(str(st.get("gpu") or "🖥️ CPU"))),
        ]
        if disk:
            disk_line = f"💾 {disk.get('free_gb', '?')} / {disk.get('total_gb', '?')} GB"
            chunks.append(_chip(f"{disk_line}{' ⚠️' if float(disk.get('pct') or 0) > 92 else ''}"))
        if st.get("port"):
            chunks.append(_chip(f"🔌 พอร์ต {st['port']}"))
        if st.get("rag_chunks"):
            chunks.append(_chip(f"📚 RAG {st['rag_chunks']} chunks"))
        spec_mode = str((st.get("spec") or {}).get("mode") or "auto")
        chunks.append(_chip(f"⚙️ spec {_esc(spec_mode)}"))
        pct = int(float(st.get("progress") or 8))
        pct = max(2, min(100, pct))
        notice = str(st.get("notice") or "")
        notice_color = {"ok": "#4ade80", "warn": "#fbbf24", "err": "#f87171"}.get(str(st.get("kind") or ""), pal["dim"])
        job = str(st.get("job") or "")
        job_html = f"<div style=\"color:#93c5fd;font-size:12px;margin-top:4px\">⏳ {_esc(job)} {_esc(str(st.get('live_step') or ''))}</div>" if job else ""
        header.value = (
            f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:10px;"
            f"padding:12px 16px;{base_font}color:{pal['text']}\">"
            f"<b style=\"font-size:16px\">{_esc(title)} ({_esc(plat)} • {n_models} Models • "
            "ปุ่มรีพาย Pro + Auto โหนด 4–7)</b>"
            f"<div style=\"display:flex;flex-wrap:wrap;gap:6px;margin-top:7px\">{''.join(chunks)}</div>"
            f"<div style=\"margin-top:7px;background:{pal['chip']};height:7px;border-radius:99px;overflow:hidden\">"
            f"<div style=\"height:100%;background:#22c55e;width:{pct}%;transition:width .4s\"></div></div>"
            f"{job_html}"
            + (f"<div style=\"color:{notice_color};font-size:12px;margin-top:6px\">{_esc(notice)}</div>" if notice else "")
            + "</div>"
        )

    _render_header()
    refresh_btn = _btn("🔄 รีเฟรช", "")
    cancel_btn = _btn("⏹️ ยกเลิกงาน", "dng")

    # ---------- แท็บ 1: โมเดล ----------
    search = widgets.Text(placeholder="🔍 ค้นหาโมเดล หรือพิมพ์ lora, uncensored, z-image, qwen-image, glm4…",
                          layout=widgets.Layout(width="100%"))
    flt_dd = widgets.Dropdown(
        options=(
            (f"📋 ทั้งหมด ({len(all_models)} รุ่น)", "all"),
            ("✅ ติดตั้งแล้ว", "installed"),
            ("🎮 พอดี VRAM จริง", "fit"),
            ("🔥🧩 Uncensored LoRA", "lora"),
            ("🎨 โมเดลสร้างภาพ GGUF", "image"),
            ("💥 สาย OBLITERATED", "obliterated"),
            ("🔓 สาย Unrestricted", "unrestricted"),
        ),
        value="all", layout=widgets.Layout(width="100%"),
    )
    model_list = widgets.HTML("")
    model_pick = widgets.Select(rows=8, layout=widgets.Layout(width="100%"))
    spec_dd = widgets.Dropdown(options=[("auto", "auto"), ("none", "none"), ("draft", "draft"), ("ngram", "ngram")],
                               description="โหมด spec", layout=widgets.Layout(width="240px"))
    kv_cb = widgets.Checkbox(value=True, description="KV cache Q8")
    draft_dd = widgets.Dropdown(description="Draft", layout=widgets.Layout(width="100%"))
    dl_btn = _btn("📥 โหลด", "pri")
    run_btn = _btn("🚀 รัน & แชท", "suc")
    del_btn = _btn("🗑 ลบ", "dng")
    link_btn = _btn("🔗 ซิงก์เข้าโหนดภาพ", "act")
    draft_btn = _btn("📥 โหลด Draft", "")
    spec_btn = _btn("💾 บันทึกโหมด + รีสตาร์ท", "wrn")
    model_out = widgets.HTML("")
    spec_seen: dict = {}

    def _draw_models(*_a) -> None:
        items = list(bridge.state.get("models") or []) or all_models
        flt = str(flt_dd.value)
        model_list.value = _models_html(items, str(search.value or ""), flt, pal)
        shown = [m for m in items if _model_matches(m, str(search.value or ""), flt)]
        prev = str(model_pick.value or "")
        opts = [(_lab(m), m["key"]) for m in shown] or [("(ไม่พบ)", "")]
        model_pick.options = opts
        # Select ไม่เลือกค่าแรกให้อัตโนมัติ (ต่างจาก Dropdown) → ต้องตั้งเอง ไม่งั้นปุ่มจะเห็นเป็น "ยังไม่เลือก"
        target = prev if prev in [o[1] for o in opts] else opts[0][1]
        with contextlib.suppress(Exception):
            model_pick.value = target

    def _draw_drafts() -> None:
        drafts = list(bridge.state.get("drafts") or [])
        _set_options(draft_dd, [(str(d.get("label") or d.get("key")), d["key"]) for d in drafts]
                     or [("(ยังไม่มี Draft)", "")])

    search.observe(lambda _c: _draw_models(), names="value")
    flt_dd.observe(lambda _c: _draw_models(), names="value")
    _draw_models()
    _draw_drafts()

    # ---------- แท็บ 2: แชท + รีพาย Pro ----------
    tpl_opts = [("(เลือกเทมเพลตพรอมป์)", "")] + [(str(v.get("label") or k), k) for k, v in tpls.items()]
    tpl_dd = widgets.Dropdown(options=tpl_opts, value=tpl_opts[0][1], layout=widgets.Layout(width="100%"))
    quick_dd = widgets.Dropdown(options=[(_lab(m), m["key"]) for m in bridge.models(("text",))] or [("(ไม่มี)", "")],
                                layout=widgets.Layout(width="100%"))
    quick_btn = _btn("⚡ สลับ", "wrn")
    system_ta = widgets.Textarea(placeholder="คำสั่งระบบ (system prompt)", layout=widgets.Layout(width="100%", height="62px"))
    rag_cb = widgets.Checkbox(value=True, description="📚 ใช้ RAG")
    temp_sl = widgets.FloatSlider(value=0.7, min=0.0, max=1.5, step=0.05, description="Temp",
                                  layout=widgets.Layout(width="230px"))
    autowf_cb = widgets.Checkbox(value=False, description="Auto โหนด 4–7")
    feed = widgets.HTML("")
    chat_in = widgets.Textarea(placeholder="พิมพ์ข้อความแล้วกดส่ง (Ctrl+Enter ไม่ทำงาน — กดปุ่ม 📨 ส่ง)…",
                               layout=widgets.Layout(width="100%", height="72px"))
    send_btn = _btn("📨 ส่ง", "pri")
    regen_btn = _btn("↩ สุ่มใหม่", "")
    clear_btn = _btn("🧹 ล้างแชท", "")
    rp_opts = [("ขยายความ", "expand"), ("อธิบายละเอียดขึ้น", "detail"), ("สรุปเป็นข้อ ๆ", "bullets"),
               ("เขียนใหม่ให้กระชับ", "shorten"), ("ยกตัวอย่างเพิ่ม", "example")]
    rp_dd = widgets.Dropdown(options=rp_opts, value=rp_opts[0][1], layout=widgets.Layout(width="100%"))
    rp_btn = _btn("🔁 รีพาย Pro", "act")
    chat_log: list[tuple[str, str, str]] = []
    last_bot: list[str] = [""]

    def _draw_feed() -> None:
        if not chat_log:
            feed.value = (f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:10px;"
                          f"padding:14px;min-height:240px;display:flex;flex-direction:column;{base_font}\">"
                          f"<div style=\"color:{pal['dim']};font-size:12.5px\">ยังไม่มีข้อความ — พิมพ์ด้านล่างแล้วกด 📨 ส่ง</div></div>")
            return
        bubbles = "".join(_bubble(t, w, meta, pal) for w, t, meta in chat_log[-40:])
        feed.value = (f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:10px;"
                      f"padding:12px;min-height:240px;max-height:520px;overflow-y:auto;display:flex;"
                      f"flex-direction:column;{base_font}\">{bubbles}</div>")

    _draw_feed()

    # ---------- ซิงก์โหนด 4–7 (อยู่ในแท็บแชทเหมือน Colab) ----------
    img_dd = widgets.Dropdown(options=[(_lab(m), m["key"]) for m in bridge.models(("image",))] or [("(ยังไม่มีโมเดลภาพ)", "")],
                              description="โมเดลภาพ", layout=widgets.Layout(width="100%"))
    lora_dd = widgets.Dropdown(options=[("(ไม่ใช้ LoRA)", "")], description="LoRA", layout=widgets.Layout(width="100%"))
    lora_str = widgets.FloatSlider(value=1.0, min=0.0, max=2.0, step=0.05, description="LoRA ×",
                                   layout=widgets.Layout(width="260px"))
    wf_hint = widgets.Textarea(placeholder="ข้อความตั้งต้นสำหรับแต่งโหนด 4–7 (เว้นว่าง = ใช้แชทล่าสุด)",
                               layout=widgets.Layout(width="100%", height="54px"))
    sync_btn = _btn("💾 ซิงก์ที่เลือก", "")
    wf_btn = _btn("🔗 แต่งโหนด 4–7", "act")
    wf_llm_btn = _btn("🤖 แต่งด้วย LLM", "act")
    wf_out = widgets.HTML("")

    # ---------- แท็บ 3: คลัง RAG ----------
    rag_up = widgets.FileUpload(accept=".pdf,.txt,.md,.csv,.json,.py", multiple=False, description="📄 เลือกไฟล์")
    rag_imp = _btn("📥 นำไฟล์เข้า RAG", "suc")
    rag_save = _btn("💾 บันทึกคลัง", "pri")
    rag_load = _btn("📂 โหลดคลัง", "")
    rag_clear = _btn("🗑️ ล้างคลัง", "dng")
    rag_text = widgets.Textarea(placeholder="วางข้อความ/โน้ตเพื่อเพิ่มเข้าคลังเวกเตอร์…",
                                layout=widgets.Layout(width="100%", height="90px"))
    rag_add = _btn("➕ เพิ่มเข้าคลัง", "pri")
    rag_q = widgets.Text(placeholder="🔍 ค้นในคลัง RAG…", layout=widgets.Layout(width="100%"))
    rag_find = _btn("🔎 ค้น", "suc")
    rag_out = widgets.HTML(f"<div style=\"color:{pal['dim']};font-size:12.5px;{base_font}\">ยังไม่มีข้อมูลในคลัง</div>")

    # ---------- handlers ----------
    def _note(widget: Any, text: str, cls: str = "ok") -> None:
        color = {"ok": "#4ade80", "warn": "#fbbf24", "err": "#f87171"}.get(cls, pal["dim"])
        widget.value = f"<div style=\"{base_font}color:{color};font-size:12.5px;padding:4px 2px\">{_esc(text)}</div>"

    def _refresh_all(*_a) -> None:
        bridge.refresh()
        st = dict(bridge.state)
        text_models = [m for m in (st.get("models") or []) if m.get("kind") == "text"]
        image_models = [m for m in (st.get("models") or []) if m.get("kind") == "image"]
        prev_txt, prev_img = str(quick_dd.value or ""), str(img_dd.value or "")
        _set_options(quick_dd, [(_lab(m), m["key"]) for m in text_models] or [("(ไม่มี)", "")])
        _set_options(img_dd, [(_lab(m), m["key"]) for m in image_models] or [("(ยังไม่มีโมเดลภาพ)", "")])
        for widget, prev in ((quick_dd, prev_txt), (img_dd, prev_img)):
            if prev and prev in [o[1] for o in widget.options]:
                widget.value = prev
        loras = [str(x) for x in (st.get("ready_loras") or [])]
        prev_l = str(lora_dd.value or "")
        _set_options(lora_dd, [("(ไม่ใช้ LoRA)", "")] + [(n, n) for n in loras])
        if prev_l in loras:
            lora_dd.value = prev_l
        spec = st.get("spec") or {}
        with contextlib.suppress(Exception):
            spec_dd.value = str(spec.get("mode") or "auto")
            kv_cb.value = bool(spec.get("kv_q8", True))
        _draw_models()
        _draw_drafts()
        _render_header()

    def _op(op: str) -> None:
        key = str(model_pick.value or "")
        if not key:
            _note(model_out, "เลือกโมเดลจากรายการก่อนครับ", "warn")
            return
        res = bridge.call("model_op", {"op": op, "key": key})
        ok = not (res.get("error") or res.get("ok") is False)
        _note(model_out, str(res.get("notice") or res.get("error") or f"{op} {key}"), "ok" if ok else "err")
        _refresh_all()

    def _on_link(_b=None) -> None:
        key = str(model_pick.value or "")
        if not key:
            _note(model_out, "เลือกโมเดล/LoRA ก่อนครับ", "warn")
            return
        res = bridge.call("link_image", {"key": key})
        ok = not (res.get("error") or res.get("ok") is False)
        _note(model_out, str(res.get("notice") or res.get("error") or f"เชื่อม {key} แล้ว"), "ok" if ok else "err")
        _refresh_all()

    def _on_draft(_b=None) -> None:
        key = str(draft_dd.value or "")
        if not key:
            _note(model_out, "ยังไม่มี Draft ให้โหลด", "warn")
            return
        bridge.call("dl_draft", {"key": key})
        _note(model_out, f"⏳ กำลังโหลด Draft {key} — กด 🔄 รีเฟรชอีกครั้ง", "warn")

    def _on_spec(_b=None) -> None:
        res = bridge.call("set_spec", {"mode": str(spec_dd.value), "kv_q8": bool(kv_cb.value), "restart": True})
        _note(model_out, str(res.get("notice") or "บันทึกโหมดแล้ว (รีสตาร์ทเซิร์ฟเวอร์ถ้าเปิดอยู่)"), "ok")
        _refresh_all()

    def _on_quick(_b=None) -> None:
        key = str(quick_dd.value or "")
        if not key:
            return
        res = bridge.call("model_op", {"op": "run", "key": key})
        _note(model_out, str(res.get("notice") or f"🚀 สลับไป {key}"), "ok")
        _refresh_all()

    def _on_tpl(_change=None) -> None:
        info = tpls.get(str(tpl_dd.value) or "")
        if not isinstance(info, dict):
            return
        system_ta.value = str(info.get("prompt") or "")
        with contextlib.suppress(Exception):
            temp_sl.value = float(info.get("temp") or 0.7)

    def _send(prompt: str = "", *, regen: bool = False, extra: dict | None = None) -> None:
        payload = {
            "prompt": prompt, "temp": float(temp_sl.value), "use_rag": bool(rag_cb.value),
            "system": str(system_ta.value or ""), "auto_wf": bool(autowf_cb.value),
        }
        if regen:
            payload["regen"] = True
        if extra:
            payload.update(extra)
        if not regen:
            chat_log.append(("me", prompt or "(ขยายความข้อความที่รีพาย)", ""))
            _draw_feed()
        res = bridge.call("chat", payload)
        if res.get("error") or res.get("ok") is False:
            chat_log.append(("bot", f"⚠️ {res.get('error')}", ""))
            _draw_feed()
            return
        reply = str(res.get("reply") or "")
        meta = " · ".join(str(x) for x in (res.get("model") or "", f"{res.get('tok_s')} tok/s" if res.get("tok_s") else "",
                                           f"{res.get('seconds')}s" if res.get("seconds") else "") if x)
        chat_log.append(("bot", reply, meta))
        last_bot[0] = reply
        _draw_feed()
        chat_in.value = ""
        _refresh_all()

    def _on_send(_b=None) -> None:
        text = str(chat_in.value or "").strip()
        if not text:
            return
        _send(text)

    def _on_regen(_b=None) -> None:
        _send("", regen=True)

    def _on_repro(_b=None) -> None:
        mode = str(rp_dd.value or "")
        if not last_bot[0]:
            _note(wf_out, "ยังไม่มีข้อความของโมเดลให้รีพาย", "warn")
            return
        ask = {"expand": "ขยายความให้ละเอียดขึ้น", "detail": "อธิบายให้ละเอียดและลึกขึ้น",
               "bullets": "สรุปเป็นข้อ ๆ", "shorten": "เขียนใหม่ให้กระชับขึ้น",
               "example": "ยกตัวอย่างเพิ่ม"}.get(mode, "ขยายความ")
        _send(ask, regen=True, extra={"reply_to": {"text": last_bot[0], "who": "bot"}})

    def _on_clear(_b=None) -> None:
        bridge.call("clear_chat", {})
        chat_log.clear()
        last_bot[0] = ""
        _draw_feed()
        _refresh_all()

    def _on_sync(_b=None) -> None:
        payload: dict[str, Any] = {"lora_strength": float(lora_str.value), "active_lora": str(lora_dd.value or "")}
        if img_dd.value:
            payload["active_model"] = str(img_dd.value)
        if str(wf_hint.value or "").strip():
            payload["prompt"] = str(wf_hint.value)
        bridge.call("sync_comfy_sel", payload)
        _note(wf_out, "💾 ซิงก์ตัวเลือกเข้าเซลล์ 3 แล้ว", "ok")
        _refresh_all()

    def _on_wf(use_llm: bool = False):
        def _h(_b=None):
            payload: dict[str, Any] = {
                "prompt": str(wf_hint.value or ""), "key": str(img_dd.value or ""),
                "lora_name": str(lora_dd.value or ""), "lora_strength": float(lora_str.value),
                "use_llm": bool(use_llm), "from_last_chat": not str(wf_hint.value or "").strip(),
            }
            if last_bot[0]:
                payload["reply_to"] = {"text": last_bot[0], "who": "bot"}
            _note(wf_out, "⏳ กำลังแต่งโหนด 4–7…", "warn")
            res = bridge.call("auto_workflow", payload)
            if res.get("error") or res.get("ok") is False:
                _note(wf_out, f"⚠️ {res.get('error')}", "err")
                return
            wf = res.get("workflow") or {}
            wf_out.value = (
                f"<div style=\"{base_font}background:{pal['panel']};border:1px solid {pal['line']};"
                f"border-radius:10px;padding:11px 13px;font-size:12.5px;color:{pal['text']}\">"
                f"<b style=\"color:#86efac\">🔗 ซิงก์เข้าโหนด 4–7 เซลล์ 3 แล้ว:</b> {_esc(str(wf.get('summary') or ''))}"
                f"<div style=\"margin-top:6px;color:#86efac\">➕ <b>Node 4 (Pos):</b> {_esc(str(wf.get('pos') or '')[:380])}</div>"
                f"<div style=\"margin-top:4px;color:#fca5a5\">➖ <b>Node 5 (Neg):</b> {_esc(str(wf.get('neg') or '')[:260])}</div></div>"
            )
            _refresh_all()
        return _h

    def _on_rag_import(_b=None) -> None:
        ups = rag_up.value
        if not ups:
            _note(rag_out, "ยังไม่ได้เลือกไฟล์ (กด 📄 เลือกไฟล์ ก่อน)", "warn")
            return
        if isinstance(ups, dict):
            name = str(next(iter(ups)))
            raw = (ups[name] or {}).get("content") if isinstance(ups[name], dict) else ups[name]
        else:
            first = next(iter(ups), {}) or {}
            name, raw = str(first.get("name") or "upload.bin"), first.get("content")
        res = bridge.call("rag_op", {"op": "upload", "name": name,
                                     "b64": base64.b64encode(bytes(raw or b"")).decode("ascii")})
        _note(rag_out, str(res.get("msg") or res.get("error") or f"นำเข้า {name} แล้ว"),
              "err" if res.get("ok") is False else "ok")
        _refresh_all()

    def _rag_simple(op: str):
        def _h(_b=None):
            res = bridge.call("rag_op", {"op": op})
            _note(rag_out, str(res.get("msg") or res.get("error") or op), "err" if res.get("ok") is False else "ok")
            _refresh_all()
        return _h

    def _on_rag_add(_b=None) -> None:
        text = str(rag_text.value or "").strip()
        if not text:
            _note(rag_out, "ยังไม่มีข้อความให้เพิ่ม", "warn")
            return
        res = bridge.call("rag_op", {"op": "add_text", "text": text})
        _note(rag_out, str(res.get("msg") or res.get("error") or "เพิ่มแล้ว"), "ok")
        rag_text.value = ""
        _refresh_all()

    def _on_rag_find(_b=None) -> None:
        res = bridge.call("rag_op", {"op": "search", "query": str(rag_q.value or "")})
        hits = res.get("hits") or []
        if not hits:
            _note(rag_out, "ไม่พบข้อมูลที่ตรงกับคำค้น", "warn")
            return
        rows = "".join(
            f"<div style=\"background:{pal['panel']};border:1px solid {pal['line']};border-radius:8px;"
            f"padding:8px 10px;margin:6px 0;font-size:12.5px;white-space:pre-wrap;color:{pal['text']}\">"
            f"{_esc(str(h.get('text') if isinstance(h, dict) else h)[:320])}</div>"
            for h in hits[:5]
        )
        rag_out.value = (f"<div style=\"{base_font}\"><div style=\"color:#4ade80;font-size:12.5px\">"
                         f"🔎 พบ {len(hits)} ชิ้น</div>{rows}</div>")

    def _on_cancel(_b=None) -> None:
        res = bridge.call("cancel", {})
        _note(model_out, str(res.get("notice") or "ยกเลิกแล้ว"), "warn")
        _refresh_all()

    def _on_refresh(_b=None) -> None:
        _refresh_all()

    tpl_dd.observe(lambda c: _on_tpl(c), names="value")
    quick_btn.on_click(_on_quick)
    dl_btn.on_click(lambda _b: _op("download"))
    run_btn.on_click(lambda _b: _op("run"))
    del_btn.on_click(lambda _b: _op("delete"))
    link_btn.on_click(_on_link)
    draft_btn.on_click(_on_draft)
    spec_btn.on_click(_on_spec)
    send_btn.on_click(_on_send)
    regen_btn.on_click(_on_regen)
    clear_btn.on_click(_on_clear)
    rp_btn.on_click(_on_repro)
    sync_btn.on_click(_on_sync)
    wf_btn.on_click(_on_wf(False))
    wf_llm_btn.on_click(_on_wf(True))
    rag_imp.on_click(_on_rag_import)
    rag_add.on_click(_on_rag_add)
    rag_find.on_click(_on_rag_find)
    rag_save.on_click(_rag_simple("save"))
    rag_load.on_click(_rag_simple("load"))
    rag_clear.on_click(_rag_simple("clear"))
    refresh_btn.on_click(_on_refresh)
    cancel_btn.on_click(_on_cancel)

    def _sec(text: str) -> Any:
        return widgets.HTML(f"<div style=\"{base_font}color:{pal['dim']};font-size:12.5px;margin:8px 0 4px 0;"
                            f"font-weight:600\">{_esc(text)}</div>")

    models_tab = widgets.VBox([
        widgets.HBox([search]), widgets.HBox([flt_dd]),
        model_list,
        _sec("เลือกโมเดลจากรายการ แล้วสั่งงาน"),
        model_pick,
        widgets.HBox([dl_btn, run_btn, del_btn, link_btn]),
        widgets.HBox([draft_dd]), widgets.HBox([draft_btn]),
        model_out,
        _sec("⚙️ โหมด spec / KV cache"),
        widgets.HBox([spec_dd, kv_cb]), widgets.HBox([spec_btn]),
    ])
    chat_tab = widgets.VBox([
        widgets.HBox([tpl_dd]),
        widgets.HBox([quick_dd, quick_btn]),
        system_ta,
        widgets.HBox([rag_cb, temp_sl, autowf_cb]),
        feed,
        chat_in,
        widgets.HBox([send_btn, regen_btn, clear_btn]),
        _sec("🔁 ปุ่มรีพาย Pro (ต่อจากข้อความล่าสุดของโมเดล)"),
        widgets.HBox([rp_dd, rp_btn]),
        _sec("🔗 ซิงก์โหนด 4–7 (เซลล์ 3)"),
        img_dd, lora_dd, widgets.HBox([lora_str]),
        widgets.HBox([sync_btn, wf_btn, wf_llm_btn]),
        wf_hint, wf_out,
    ])
    rag_tab = widgets.VBox([
        widgets.HBox([rag_up]), widgets.HBox([rag_imp, rag_save, rag_load, rag_clear]),
        rag_text, widgets.HBox([rag_add]),
        rag_q, widgets.HBox([rag_find]), rag_out,
    ])
    acc = widgets.Accordion(children=[models_tab, chat_tab, rag_tab])
    acc.set_title(0, f"📦 1. โมเดล & Uncensored LoRA ({len(all_models)})")
    acc.set_title(1, "💬 2. แชท + ปุ่มรีพาย Pro")
    acc.set_title(2, "📚 3. คลัง PDF RAG")
    acc.selected_index = 0  # Colab เปิดแท็บ 1 (โมเดล) เป็นค่าเริ่มต้น
    return widgets.VBox([header, widgets.HBox([cancel_btn, refresh_btn]), acc])


def comfy_console(dispatch: Callable[[str, dict], Any]) -> Any:
    """แผง ComfyUI สำหรับ Kaggle — เลียนแบบหน้าจอ ComfyUI Studio ของ Colab (การ์ดโหนด 1–10)

    ปุ่ม/ช่องกรอกชุดเดียวกับหน้าจอ HTML: โหนด 4–5 (pos/neg), โหนด 6–7 (ขนาด/steps/cfg/seed/
    sampler/scheduler), LoRA และปุ่มคิวงาน → ``dispatch("queue", {...})``
    """
    widgets = _widgets()
    if widgets is None:
        return None

    pal = _pal()
    grn, grn_l, node_bg = "#10b981", "#34d399", "#111827"
    font = _f()
    bridge = _Bridge(dispatch)
    state = bridge.refresh()
    active = dict(state.get("active") or {})

    def _btn(caption: str, color: str = "#059669", tip: str = "") -> Any:
        return widgets.Button(description=caption, tooltip=tip or caption,
                              layout=widgets.Layout(min_width="140px", height="36px"),
                              style={"button_color": color, "text_color": "#fff", "font_weight": "700"})

    def _card(title: str, tag: str, children: list[Any], accent: str = grn) -> Any:
        head = widgets.HTML(
            f"<div style=\"{font}display:flex;justify-content:space-between;align-items:center;gap:6px\">"
            f"<b style=\"color:{grn_l};font-size:13px\">{_esc(title)}</b>"
            f"<span style=\"color:{pal['dim']};font-size:11px\">{_esc(tag)}</span></div>"
        )
        return widgets.VBox([head, *children], layout=widgets.Layout(
            border=f"1px solid {pal['line']}", border_top=f"3px solid {accent}",
            padding="10px 12px", margin="0 0 10px 0", width="100%",
        ))

    header = widgets.HTML("")
    loaders = widgets.HTML("")
    status_line = widgets.HTML("")
    pos_ta = widgets.Textarea(
        placeholder="คำบรรยายภาพ (positive) — จะซิงก์จากโหนด 4 อัตโนมัติถ้าเว้นว่าง",
        layout=widgets.Layout(width="100%", height="76px"))
    neg_ta = widgets.Textarea(value="blurry, low quality, deformed, watermark",
                              placeholder="negative prompt", layout=widgets.Layout(width="100%", height="54px"))
    size_dd = widgets.Dropdown(options=("1024x1024", "832x1216", "1216x832", "768x1344", "1344x768"),
                               value="1024x1024", description="ขนาด",
                               layout=widgets.Layout(width="210px"))
    steps_sl = widgets.IntSlider(value=int(active.get("steps") or 8), min=1, max=40, description="steps",
                                 layout=widgets.Layout(width="260px"))
    cfg_fl = widgets.FloatSlider(value=float(active.get("cfg") or 1.0), min=0.5, max=12.0, step=0.1,
                                 description="CFG", layout=widgets.Layout(width="260px"))
    seed_it = widgets.IntText(value=-1, description="seed (-1=สุ่ม)", layout=widgets.Layout(width="210px"))
    sampler_dd = widgets.Dropdown(options=("euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "heun"),
                                  value=str(active.get("sampler") or "euler"), description="sampler",
                                  layout=widgets.Layout(width="240px"))
    sched_dd = widgets.Dropdown(options=("simple", "normal", "karras", "exponential", "sgm_uniform"),
                                value=str(active.get("scheduler") or "simple"), description="scheduler",
                                layout=widgets.Layout(width="240px"))
    lora_txt = widgets.Text(value=str(state.get("synced_lora") or ""), description="LoRA",
                            layout=widgets.Layout(width="100%"))
    lora_str = widgets.FloatSlider(value=float(state.get("synced_lora_strength") or 1.0), min=0.0, max=1.5,
                                   step=0.05, description="LoRA ×", layout=widgets.Layout(width="260px"))
    auto_cb = widgets.Checkbox(value=True, description="🤖 Auto โหนด 4–7 (รับค่าสดจากเซลล์ 2)",
                               layout=widgets.Layout(width="100%"))
    queue_btn = _btn("🎨 สร้างภาพ", "#059669")
    refresh_btn = _btn("🔄 รีเฟรชสถานะ", "#1e293b")
    out = widgets.HTML("")

    def _render_header() -> None:
        st = dict(bridge.state) or state
        act = dict(st.get("active") or {})
        run = bool(st.get("running"))
        pill = _chip("🟢 ComfyUI :8188 พร้อมทำงาน" if st.get("status") else "⚪ ไม่ทราบสถานะ",
                     "#4ade80" if st.get("status") else pal["dim"], "#1e293b", grn)
        sync_pill = _chip(f"🔗 {_esc(str(act.get('key') or 'ยังไม่เลือกโมเดลภาพ'))}", pal["text"],
                          "#2e1065", "#a855f7")
        run_pill = _chip("⏳ กำลังเรนเดอร์" if run else "💤 ว่าง", "#fbbf24" if run else pal["dim"],
                         "#1e293b", pal["line"])
        header.value = (
            f"<div style=\"background:#0b1120;border:1px solid {grn};border-radius:10px;padding:12px 16px;{font}color:{pal['text']}\">"
            f"<b style=\"font-size:15.5px;color:{grn_l}\">🎨 ComfyUI GGUF Node Workflow Studio "
            "(Automation โหนด 4–7 ตามช่องแชทและการรีพายเซลล์ที่ 2)</b>"
            f"<div style=\"display:flex;flex-wrap:wrap;gap:6px;margin-top:8px\">{pill}{sync_pill}{run_pill}</div>"
            f"<div style=\"color:#fde047;font-size:12.5px;margin-top:7px\">{_esc(str(st.get('status') or ''))}</div>"
            "</div>"
        )
        loaders.value = (
            f"<div style=\"background:{node_bg};border:1px solid {pal['line']};border-radius:9px;"
            f"padding:10px 12px;{font}color:{pal['text']};font-size:12.5px\">"
            f"<b style=\"color:{grn_l}\">📦 Node 1–3 & 10: Loaders + LoRA</b> "
            f"<span style=\"color:{pal['dim']};font-size:11px\">AUTO-SYNC</span><br>"
            f"<span style=\"color:{pal['dim']}\">UNet:</span> {_esc(str(act.get('unet') or '—'))}<br>"
            f"<span style=\"color:{pal['dim']}\">CLIP:</span> {_esc(str(act.get('clip') or '—'))} "
            f"({_esc(str(act.get('clip_type') or 'qwen_image'))})<br>"
            f"<span style=\"color:{pal['dim']}\">VAE:</span> {_esc(str(act.get('vae') or '—'))}"
            + ("" if act.get("ready") else f"<br><span style=\"color:#fbbf24\">🔒 ยังไม่พร้อม — กด 📥 โหลดโมเดลภาพในเซลล์ 2 ก่อน</span>")
            + f"<br><span style=\"color:{pal['dim']}\">LoRA ที่ซิงก์:</span> {_esc(str(st.get('synced_lora') or '(ไม่ใช้)'))}"
            "</div>"
        )

    def _render_out() -> None:
        st = dict(bridge.state) or state
        img = st.get("last_img") or {}
        parts = []
        if img.get("uri"):
            parts.append(f"<img src=\"{img['uri']}\" style=\"max-width:100%;border-radius:8px;border:1px solid {pal['line']};margin-top:8px\">")
            meta = " · ".join(str(x) for x in (img.get("name"), f"seed {img.get('seed')}",
                                               f"{img.get('seconds')}s" if img.get("seconds") else "",
                                               img.get("lora_info")) if x)
            parts.append(f"<div style=\"color:{pal['dim']};font-size:12px;margin-top:4px\">{_esc(meta)}</div>")
        disk = st.get("disk") or {}
        if disk:
            parts.append(f"<div style=\"color:{pal['dim']};font-size:12px;margin-top:4px\">💾 ว่าง "
                         f"{disk.get('free_gb', '?')} / {disk.get('total_gb', '?')} GB</div>")
        out.value = (f"<div style=\"background:{node_bg};border:1px solid {pal['line']};border-radius:9px;"
                     f"padding:10px 12px;{font}color:{pal['text']};font-size:12.5px\">"
                     f"<b style=\"color:{grn_l}\">🖼️ Node 8–9: VAEDecode → SaveImage</b>"
                     f"<span style=\"color:{pal['dim']};font-size:11px\"> OUTPUT</span>"
                     + "".join(parts) + "</div>")
        status_line.value = (f"<div style=\"{font}color:{pal['dim']};font-size:12px\">"
                             f"รูปแบบภาพ {size_dd.value} • steps {steps_sl.value} • CFG {cfg_fl.value:.2f} • "
                             f"seed {seed_it.value} • {sampler_dd.value}/{sched_dd.value}</div>")

    def _refresh(*_a) -> None:
        bridge.refresh()
        _render_header()
        _render_out()

    def _on_queue(_b=None) -> None:
        w_txt, h_txt = str(size_dd.value).split("x")
        payload = {
            "pos": str(pos_ta.value or ""), "neg": str(neg_ta.value or ""),
            "width": int(w_txt), "height": int(h_txt), "steps": int(steps_sl.value),
            "cfg": float(cfg_fl.value), "seed": int(seed_it.value),
            "sampler": str(sampler_dd.value), "scheduler": str(sched_dd.value),
            "lora_name": str(lora_txt.value or ""), "lora_strength": float(lora_str.value),
        }
        res = bridge.call("queue", payload)
        _render_header()
        if res.get("error"):
            out.value = f"<div style=\"{font}color:#f87171;font-size:12.5px\">⚠️ {_esc(str(res['error']))}</div>"
        _poll_later(6.0)

    def _poll_later(seconds: float) -> None:
        def _job() -> None:
            _sleep(seconds)
            with contextlib.suppress(Exception):
                _refresh()
        threading.Thread(target=_job, daemon=True).start()

    queue_btn.on_click(_on_queue)
    refresh_btn.on_click(_refresh)
    for widget in (size_dd, steps_sl, cfg_fl, seed_it, sampler_dd, sched_dd):
        with contextlib.suppress(Exception):
            widget.observe(lambda _c: _render_out(), names="value")

    _render_header()
    _render_out()

    return widgets.VBox([
        header,
        loaders,
        _card("✍️ Node 4–5: CLIPTextEncode", "PROMPT AUTO", [pos_ta, neg_ta], accent=grn),
        _card("⚙️ Node 6–7: Latent & KSampler", "SAMPLER AUTO",
              [widgets.HBox([size_dd, steps_sl]), widgets.HBox([cfg_fl, seed_it]),
               widgets.HBox([sampler_dd, sched_dd]), auto_cb], accent=grn),
        _card("📦 LoRA (โหนด 10)", "AUTO-SYNC", [lora_txt, lora_str], accent="#a855f7"),
        _card("🖼️ Node 8–9: VAEDecode → SaveImage", "OUTPUT", [out, status_line, widgets.HBox([queue_btn, refresh_btn])], accent=grn),
    ])


def _sleep(seconds: float) -> None:
    import time as _t  # noqa: PLC0415
    _t.sleep(seconds)


def show_console(widget: Any, title: str = "") -> bool:
    """show แผงที่สร้างไว้ (คืน False ถ้าไม่มี ipywidgets)"""
    if widget is None:
        if title:
            print(f"ℹ️ {title}: ไม่มี ipywidgets — ใช้คำสั่งผ่านโค้ดแทน")
        return False
    from IPython.display import display  # noqa: PLC0415

    display(widget)
    return True


__all__ = ["studio_console", "comfy_console", "show_console"]
