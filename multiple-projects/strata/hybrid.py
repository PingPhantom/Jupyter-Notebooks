"""แผงควบคุม (ipywidgets) สำหรับแพลตฟอร์มที่ไม่มี Colab RPC — Kaggle / JupyterLab

หน้าจอหลักของโปรเจกต์เป็น HTML5 (เหมือน Colab ทุกพิกเซล) แต่บน Kaggle ไม่มีช่องทางให้ JS
ในหน้าเว็บเรียกกลับเข้า kernel ได้ (``google.colab.kernel.invokeFunction`` มีแค่บน Colab)
แผงนี้จึงทำหน้าที่เป็น "รีโมต" ที่วางไว้ใต้หน้าจอ — เรียก action ชุดเดียวกับที่ปุ่มใน HTML เรียก
(``_handle_action`` ของเซลล์ 2 และ ``_cf_rpc`` ของเซลล์ 3) เพื่อให้ใช้งานได้จริงโดยไม่ต้องใช้ tunnel

หลักการ: ไม่พึ่งไลบรารีใหม่ ใช้ ``ipywidgets`` ที่ notebook มีอยู่แล้ว และทุก action ถูกเรียกผ่าน
dispatcher ที่ส่งเข้ามา → แผงนี้ไม่รู้จักธุรกิจภายใน จึงใช้ซ้ำได้ทั้งสองเซลล์
"""

from __future__ import annotations

import contextlib
import html as _html
import threading
from typing import Any, Callable

_CSS = """<style>
.hy-root { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background:#0b1120; color:#e2e8f0; border:1px solid #334155; border-radius:10px;
           padding:12px 14px; max-width:100%; }
.hy-root .hy-title { font-size:14px; font-weight:700; margin:0 0 6px 0; color:#f8fafc; }
.hy-root .hy-sub { font-size:12px; color:#94a3b8; margin:0 0 8px 0; line-height:1.5; }
.hy-root .hy-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:6px 0; }
.hy-badge { display:inline-block; padding:2px 9px; border-radius:99px; background:#1e293b;
            border:1px solid #334155; font-size:11.5px; margin:0 6px 6px 0; }
.hy-ok { color:#4ade80; } .hy-warn { color:#fbbf24; } .hy-err { color:#f87171; } .hy-dim { color:#94a3b8; }
.hy-reply { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:9px 11px;
            margin:6px 0; font-size:13px; line-height:1.55; white-space:pre-wrap; }
.hy-reply.me { border-color:#2563eb; } .hy-reply.bot { border-color:#059669; }
.hy-pre { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:8px 10px;
          font-size:12px; white-space:pre-wrap; margin:6px 0; }
</style>"""

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


def _status_html(bridge: _Bridge) -> str:
    s = bridge.state
    if not s:
        return "<span class='hy-dim'>ยังไม่ได้เชื่อมต่อ kernel</span>"
    bits = [f"<span class='hy-badge'>{_esc(s.get('platform', ''))}</span>"]
    on = bool(s.get("online"))
    bits.append(f"<span class='hy-badge'>{'🟢' if on else '🔴'} {_esc(s.get('active_model', 'Offline'))}</span>")
    if s.get("gpu"):
        bits.append(f"<span class='hy-badge'>{_esc(s['gpu'])}</span>")
    if s.get("disk"):
        d = s["disk"]
        bits.append(f"<span class='hy-badge'>💾 ว่าง {d.get('free_gb', '?')} GB</span>")
    if s.get("rag_chunks"):
        bits.append(f"<span class='hy-badge'>📚 RAG {s['rag_chunks']} chunks</span>")
    if s.get("job"):
        bits.append(f"<span class='hy-badge hy-warn'>⏳ {_esc(s['job'])}</span>")
    if s.get("notice"):
        cls = {"ok": "hy-ok", "warn": "hy-warn", "err": "hy-err"}.get(str(s.get("kind")), "hy-dim")
        bits.append(f"<div class='{cls}' style='font-size:12.5px;margin-top:4px'>{_esc(s['notice'])}</div>")
    return "".join(bits)


def _img_html(bridge: _Bridge, alt: str = "") -> str:
    s = bridge.state
    uri = str(((s.get("last_img") or {}).get("uri") or ""))
    if not uri:
        stat = str(s.get("img_stat") or "")
        return f"<div class='hy-pre hy-dim'>{_esc(stat or 'ยังไม่มีภาพ — กดสร้างภาพด้านบน')}</div>" if stat else ""
    meta = s.get("last_img") or {}
    cap = _esc(meta.get("size") or meta.get("name") or "")
    secs = meta.get("seconds")
    cap = f"{cap} • {secs}s" if secs else cap
    return (f"<div style='margin-top:8px'><img src='{uri}' alt='{_esc(alt)}' "
            f"style='max-width:100%;border:1px solid #334155;border-radius:8px'>"
            f"<div class='hy-dim' style='font-size:12px;margin-top:4px'>{cap}</div></div>")


def _btn(widgets, caption: str, style: str = "", tooltip: str = ""):
    btn = widgets.Button(description=caption, tooltip=tooltip or caption)
    if style:
        btn.add_class(f"hy-btn-{style}")
    return btn


def studio_console(
    dispatch: Callable[[str, dict], Any],
    *,
    title: str = "🎛️ แผงควบคุมด่วน (ใช้ได้จริงบนแพลตฟอร์มนี้)",
    model_dir: str | None = None,
) -> Any:
    """สร้างแผงควบคุมคู่กับหน้าจอ Studio — คืน ``widgets.VBox`` (หรือ None ถ้าไม่มี ipywidgets)"""
    widgets = _widgets()
    if widgets is None:
        return None

    bridge = _Bridge(dispatch)
    state = bridge.refresh()

    head = widgets.HTML(
        f"{_CSS}<div class='hy-root'><div class='hy-title'>{_esc(title)}</div>"
        "<div class='hy-sub'>หน้าจอ HTML ด้านบนเป็นชุดเดียวกับ Colab แต่บนแพลตฟอร์มนี้ JS สั่ง kernel ตรง ๆ ไม่ได้ "
        "— ใช้แผงนี้สั่งงานแทน (ปุ่มในหน้าจอด้านบนจะกดไม่ติด)</div></div>"
    )
    status = widgets.HTML(f"<div class='hy-root'>{_status_html(bridge)}</div>")
    refresh_btn = widgets.Button(description="🔄 รีเฟรชสถานะ", layout=widgets.Layout(width="150px"))

    # ---------- โมเดล ----------
    text_models = bridge.models(("text",))
    image_models = bridge.models(("image",))
    text_dd = widgets.Dropdown(
        options=[(_lab(m), m["key"]) for m in text_models] or [("(ไม่มี)", "")],
        description="โมเดลข้อความ", layout=widgets.Layout(width="100%"),
    )
    image_dd = widgets.Dropdown(
        options=[(_lab(m), m["key"]) for m in image_models] or [("(ไม่มี)", "")],
        description="โมเดลภาพ", layout=widgets.Layout(width="100%"),
    )
    spec_dd = widgets.Dropdown(
        options=[("auto", "auto"), ("none", "none"), ("draft", "draft"), ("ngram", "ngram")],
        description="โหมด spec", layout=widgets.Layout(width="260px"),
    )
    kv_cb = widgets.Checkbox(value=True, description="KV cache Q8")
    if state.get("spec"):
        with contextlib.suppress(Exception):
            spec_dd.value = str(state["spec"].get("mode") or "auto")
            kv_cb.value = bool(state["spec"].get("kv_q8", True))

    dl_btn = _btn(widgets, "📥 โหลด", "pri")
    run_btn = _btn(widgets, "🚀 รัน & แชท", "suc")
    del_btn = _btn(widgets, "🗑 ลบ", "dng")
    img_dl_btn = _btn(widgets, "📥 โหลดโมเดลภาพ")
    spec_btn = _btn(widgets, "💾 บันทึกโหมด + รีสตาร์ท", "wrn")
    model_out = widgets.HTML("")

    # ---------- แชท ----------
    chat_in = widgets.Textarea(placeholder="พิมพ์ข้อความแล้วกดส่ง…", layout=widgets.Layout(width="100%", height="76px"))
    rag_cb = widgets.Checkbox(value=False, description="ใช้ RAG")
    temp_sl = widgets.FloatSlider(value=0.7, min=0.0, max=1.5, step=0.05, description="temp",
                                  layout=widgets.Layout(width="260px"))
    send_btn = _btn(widgets, "📨 ส่ง", "pri")
    regen_btn = _btn(widgets, "↩ สุ่มใหม่")
    clear_btn = _btn(widgets, "🧹 ล้างแชท")
    chat_out = widgets.HTML("<div class='hy-root'><div class='hy-dim'>ยังไม่มีข้อความ</div></div>")
    chat_log: list[str] = []

    # ---------- ภาพ ----------
    img_in = widgets.Textarea(placeholder="คำบรรยายภาพ…", layout=widgets.Layout(width="100%", height="66px"))
    gen_btn = _btn(widgets, "🖼️ สร้างภาพ", "suc")
    img_out = widgets.HTML(f"<div class='hy-root'>{_img_html(bridge)}</div>")

    # ---------- ซิงก์โหนด 4–7 ----------
    wf_in = widgets.Textarea(placeholder="ข้อความตั้งต้น (เว้นว่าง = ใช้แชทล่าสุด)",
                             layout=widgets.Layout(width="100%", height="56px"))
    wf_btn = _btn(widgets, "🔗 แต่งโหนด 4–7")
    wf_cancel = _btn(widgets, "⏹ ยกเลิกงาน", "dng")
    wf_out = widgets.HTML("")

    # ---------- handlers ----------
    def _sync_status(*_a):
        bridge.refresh()
        status.value = f"<div class='hy-root'>{_status_html(bridge)}</div>"

    def _refresh_models(*_a):
        bridge.refresh()
        tm = bridge.models(("text",))
        im = bridge.models(("image",))
        text_dd.options = [(_lab(m), m["key"]) for m in tm] or [("(ไม่มี)", "")]
        image_dd.options = [(_lab(m), m["key"]) for m in im] or [("(ไม่มี)", "")]
        _sync_status()

    def _op(kind: str, key: str, op: str):
        if not key:
            model_out.value = "<div class='hy-root hy-warn'>เลือกโมเดลก่อนครับ</div>"
            return
        res = bridge.call("model_op", {"op": op, "key": key})
        note = res.get("notice") or res.get("error") or f"{op} {key}"
        cls = "hy-err" if res.get("error") else "hy-ok"
        model_out.value = f"<div class='hy-root'><span class='{cls}'>{_esc(note)}</span></div>"
        _refresh_models()

    def _render_chat():
        if not chat_log:
            chat_out.value = "<div class='hy-root'><div class='hy-dim'>ยังไม่มีข้อความ</div></div>"
            return
        body = "".join(chat_log[-8:])
        chat_out.value = f"<div class='hy-root'>{body}</div>"

    def _on_send(_b=None, regen: bool = False):
        payload = {"prompt": chat_in.value, "temp": float(temp_sl.value), "use_rag": bool(rag_cb.value)}
        if regen:
            payload["regen"] = True
        res = bridge.call("chat", payload)
        if res.get("error") or res.get("ok") is False:
            chat_log.append(f"<div class='hy-reply me hy-err'>⚠️ {_esc(res.get('error'))}</div>")
            _render_chat()
            return
        if not regen:
            chat_log.append(f"<div class='hy-reply me'>🧑 {_esc(chat_in.value[:600])}</div>")
        reply = str(res.get("reply") or "")
        meta = []
        if res.get("model"):
            meta.append(str(res["model"]))
        for key, unit in (("tokens", "tok"), ("tok_s", "tok/s"), ("seconds", "s")):
            if res.get(key):
                meta.append(f"{res[key]} {unit}")
        chat_log.append(
            f"<div class='hy-reply bot'>🤖 {_esc(reply)}"
            + (f"<div class='hy-dim' style='font-size:11.5px;margin-top:6px'>{_esc(' · '.join(meta))}</div>" if meta else "")
            + "</div>"
        )
        chat_in.value = ""
        _render_chat()
        _refresh_models()

    def _on_clear(_b):
        bridge.call("clear_chat", {})
        chat_log.clear()
        _render_chat()
        _sync_status()

    def _on_gen(_b):
        res = bridge.call("gen_image", {"key": image_dd.value, "prompt": img_in.value})
        if res.get("error") or res.get("ok") is False:
            img_out.value = f"<div class='hy-root hy-err'>⚠️ {_esc(res.get('error'))}</div>"
            return
        img_out.value = "<div class='hy-root hy-warn'>⏳ กำลังวาดภาพ… (อีกสักครู่ กด 🔄 รีเฟรชสถานะ)</div>"
        _refresh_later(8)

    def _on_wf(_b):
        res = bridge.call("auto_workflow", {"prompt": wf_in.value, "from_last_chat": not wf_in.value.strip()})
        if res.get("error") or res.get("ok") is False:
            wf_out.value = f"<div class='hy-root hy-err'>⚠️ {_esc(res.get('error'))}</div>"
            return
        wf = res.get("workflow") or {}
        wf_out.value = (
            "<div class='hy-root'>"
            f"<div class='hy-ok'>✅ ซิงก์โหนด 4–7 แล้ว: {_esc(wf.get('summary'))}</div>"
            f"<div class='hy-pre'>➕ Pos: {_esc(str(wf.get('pos'))[:400])}</div>"
            f"<div class='hy-pre'>➖ Neg: {_esc(str(wf.get('neg'))[:260])}</div></div>"
        )
        _sync_status()

    def _on_cancel(_b):
        res = bridge.call("cancel", {})
        _sync_status()
        model_out.value = f"<div class='hy-root hy-warn'>{_esc(res.get('notice') or 'ยกเลิกแล้ว')}</div>"

    def _on_spec(_b):
        res = bridge.call("set_spec", {"mode": spec_dd.value, "kv_q8": bool(kv_cb.value), "restart": True})
        note = res.get("notice") or "บันทึกโหมดแล้ว (กำลังรีสตาร์ทเซิร์ฟเวอร์ถ้าเปิดอยู่)"
        model_out.value = f"<div class='hy-root'><span class='hy-ok'>{_esc(note)}</span></div>"
        _refresh_later(6)

    def _refresh_later(seconds: float) -> None:
        def _job():
            import time as _t  # noqa: PLC0415
            _t.sleep(seconds)
            try:
                _sync_status()
                img_out.value = f"<div class='hy-root'>{_img_html(bridge)}</div>"
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_job, daemon=True).start()

    dl_btn.on_click(lambda _b: _op("text", text_dd.value, "download"))
    run_btn.on_click(lambda _b: _op("text", text_dd.value, "run"))
    del_btn.on_click(lambda _b: _op("text", text_dd.value, "delete"))
    img_dl_btn.on_click(lambda _b: _op("image", image_dd.value, "download"))
    spec_btn.on_click(_on_spec)
    def _on_refresh(_b):
        _refresh_models()
        img_out.value = f"<div class='hy-root'>{_img_html(bridge)}</div>"

    refresh_btn.on_click(_on_refresh)
    send_btn.on_click(_on_send)
    regen_btn.on_click(lambda _b: _on_send(regen=True))
    clear_btn.on_click(_on_clear)
    gen_btn.on_click(_on_gen)
    wf_btn.on_click(_on_wf)
    wf_cancel.on_click(_on_cancel)

    models_box = widgets.VBox([
        widgets.HTML("<div class='hy-root'><b>🚀 โมเดลข้อความ</b></div>"),
        text_dd, widgets.HBox([dl_btn, run_btn, del_btn]),
        widgets.HTML("<div class='hy-root'><b>🎨 โมเดลภาพ</b></div>"),
        image_dd, widgets.HBox([img_dl_btn]),
        widgets.HTML("<div class='hy-root'><b>⚙️ โหมด spec</b></div>"),
        widgets.HBox([spec_dd, kv_cb]), spec_btn,
        model_out,
    ])
    chat_box = widgets.VBox([chat_in, widgets.HBox([rag_cb, temp_sl]), widgets.HBox([send_btn, regen_btn, clear_btn]), chat_out])
    image_box = widgets.VBox([img_in, widgets.HBox([gen_btn]), img_out])
    wf_box = widgets.VBox([wf_in, widgets.HBox([wf_btn, wf_cancel]), wf_out])
    acc = widgets.Accordion(children=[models_box, chat_box, image_box, wf_box])
    for i, t in enumerate(("🚀 โมเดล & โหมด", "💬 แชท", "🖼️ สร้างภาพ", "🔗 ซิงก์โหนด 4–7")):
        acc.set_title(i, t)
    acc.selected_index = 1

    return widgets.VBox([head, widgets.HBox([status]), widgets.HBox([refresh_btn]), acc])


def comfy_console(
    dispatch: Callable[[str, dict], Any],
    *,
    title: str = "🎛️ แผงควบคุม ComfyUI (ใช้ได้จริงบนแพลตฟอร์มนี้)",
) -> Any:
    """แผงสั่งงาน ComfyUI (queue) + โชว์สถานะ/ภาพล่าสุด — คู่กับหน้าจอ HTML ในเซลล์ 3"""
    widgets = _widgets()
    if widgets is None:
        return None

    bridge = _Bridge(dispatch)
    bridge.refresh()

    head = widgets.HTML(
        f"{_CSS}<div class='hy-root'><div class='hy-title'>{_esc(title)}</div>"
        "<div class='hy-sub'>หน้าจอ ComfyUI ด้านบนเป็นชุดเดียวกับ Colab แต่รับคำสั่งจาก JS ไม่ได้บนแพลตฟอร์มนี้ "
        "— กรอกด้านล่างแล้วกดสร้างภาพได้เลย</div></div>"
    )
    status = widgets.HTML(f"<div class='hy-root'>{_status_html(bridge)}</div>")
    prompt = widgets.Textarea(placeholder="คำบรรยายภาพ (positive)…",
                              layout=widgets.Layout(width="100%", height="70px"))
    neg = widgets.Textarea(value="blurry, low quality, deformed, watermark",
                           placeholder="negative…", layout=widgets.Layout(width="100%", height="52px"))
    size_dd = widgets.Dropdown(options=["1024x1024", "832x1216", "1216x832"], value="1024x1024", description="ขนาด")
    steps = widgets.IntSlider(value=8, min=1, max=40, step=1, description="steps",
                              layout=widgets.Layout(width="260px"))
    seed = widgets.IntText(value=-1, description="seed (-1=สุ่ม)", layout=widgets.Layout(width="190px"))
    lora_txt = widgets.Text(value="", description="LoRA", layout=widgets.Layout(width="320px"))
    lora_str = widgets.FloatSlider(value=1.0, min=0.0, max=1.5, step=0.05, description="LoRA ×",
                                   layout=widgets.Layout(width="260px"))
    go = _btn(widgets, "🎨 สร้างภาพ", "suc")
    refresh_btn = widgets.Button(description="🔄 รีเฟรชสถานะ", layout=widgets.Layout(width="150px"))
    out = widgets.HTML("")

    def _render():
        res = bridge.refresh()
        status.value = f"<div class='hy-root'>{_status_html(bridge)}</div>"
        extra = ""
        if res.get("status"):
            extra += f"<div class='hy-pre'>{_esc(res['status'])}</div>"
        out.value = f"<div class='hy-root'>{extra}{_img_html(bridge)}</div>"

    def _on_go(_b):
        w, h = (int(x) for x in str(size_dd.value).split("x"))
        payload = {
            "prompt": prompt.value, "negative": neg.value,
            "width": w, "height": h, "steps": int(steps.value),
            "seed": int(seed.value if seed.value else -1),
            "lora_name": lora_txt.value.strip(), "lora_strength": float(lora_str.value),
        }
        res = bridge.call("queue", payload)
        out.value = f"<div class='hy-root hy-warn'>⏳ เข้าคิวแล้ว — {_esc((res or {}).get('status') or 'กำลังวาด…')}</div>"
        threading.Thread(target=lambda: (_sleep(8), _render()), daemon=True).start()

    go.on_click(_on_go)
    refresh_btn.on_click(lambda _b: _render())
    _render()
    return widgets.VBox([
        head, status,
        widgets.HTML("<div class='hy-root'><b>🖼️ สร้างภาพ</b></div>"),
        prompt, neg, widgets.HBox([size_dd, steps]), widgets.HBox([seed, lora_txt]), lora_str,
        widgets.HBox([go, refresh_btn]), out,
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
