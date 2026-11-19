"""สะพาน JS → kernel สำหรับ frontend ที่ไม่มี Colab RPC (Kaggle / JupyterLab)

ปัญหา: หน้าจอ Studio เป็น HTML5 ที่เรียก ``callRpc()`` จาก JS แต่บน Kaggle ไม่มี
``google.colab.kernel.invokeFunction`` และไม่มี ``Jupyter.notebook.kernel.comm_manager``
ปุ่มทุกปุ่มจึงกดไม่ติด (ต้องใช้ tunnel หรือ widget UI แทน)

วิธีแก้ในโมดูลนี้ — ใช้ของที่ notebook มีอยู่แล้วล้วน ๆ ไม่ต้องติดตั้งอะไรเพิ่ม:

1. ``ipywidgets.Text`` ที่ซ่อนไว้ → JS เขียน JSON ลง ``<input>`` แล้ว dispatch event
   ``input`` ตัว JS ของ ipywidgets จะซิงก์ค่ากลับเข้า kernel ผ่าน comm มาตรฐาน
2. ``ipywidgets.HTML`` ที่ซ่อนไว้ → kernel เขียนผลลัพธ์ (JSON) ลงไป JS อ่าน ``textContent``
3. ``display(Javascript(...))`` ติดตั้ง ``window.__strataHook(act, payload) -> Promise``
   ให้หน้าเว็บเรียกได้เหมือน ``invokeFunction`` ของ Colab

ลำดับความปลอดภัย/ความทนทาน
* ทุกคำขอมี ``id`` ไม่ซ้ำ → ค่า widget เปลี่ยนทุกครั้ง event จึงไม่ถูกกลืน
* kernel ประมวลผลทีละคำขอ (มี lock) และไม่ปล่อยให้ exception ทำให้ comm ตาย
* ติดตั้งแล้วมี self-test (ยิง ``__ping``) ถ้าไม่กลับมาใน 4 วินาที จะรายงาน ``ok=False``
  ให้ผู้เรียก fallback ไปใช้แผง ipywidgets (``strata.hybrid``) ได้
"""

from __future__ import annotations

import html as _html
import json
import secrets
import threading
import time
from typing import Any, Callable

from .hybrid import _widgets  # ใช้ตัวช่วย import ipywidgets แบบ lazy ตัวเดียวกัน

PING_ACT = "__ping"


def _hook_js(token: str, timeout_ms: int) -> str:
    return f"""
(function(){{
  const TOKEN = "{token}";
  const TIMEOUT_MS = {int(timeout_ms)};
  const reqSel = '.strata-hook-req-' + TOKEN + ' input';
  const resSel = '.strata-hook-res-' + TOKEN;
  let chain = Promise.resolve();
  function findReq(){{ return document.querySelector(reqSel); }}
  function findRes(){{ return document.querySelector(resSel); }}
  function readRes(){{
    const el = findRes();
    if(!el) return null;
    const txt = (el.textContent || '').trim();
    if(!txt) return null;
    try {{ return JSON.parse(txt); }} catch(e) {{ return null; }}
  }}
  function send(act, payload){{
    return new Promise(function(resolve, reject){{
      const req = findReq(), res = findRes();
      if(!req || !res){{ reject(new Error('strata-hook: ไม่พบวิดเจ็ตสะพานในหน้านี้')); return; }}
      const id = TOKEN + ':' + Date.now() + ':' + Math.random().toString(36).slice(2);
      const started = Date.now();
      const timer = setInterval(function(){{
        const data = readRes();
        if(data && data.id === id){{ clearInterval(timer); resolve(data.result); return; }}
        if(Date.now() - started > TIMEOUT_MS){{ clearInterval(timer); reject(new Error('strata-hook: kernel ไม่ตอบภายใน ' + Math.round(TIMEOUT_MS/1000) + 's')); }}
      }}, 120);
      req.value = JSON.stringify({{id: id, act: act, payload: payload || {{}}}});
      req.dispatchEvent(new Event('input', {{bubbles: true}}));
      req.dispatchEvent(new Event('change', {{bubbles: true}}));
    }});
  }}
  window.__strataHook = function(act, payload){{
    chain = chain.then(function(){{ return send(act, payload); }}, function(){{ return send(act, payload); }});
    return chain;
  }};
  window.__strataHookToken = TOKEN;
  window.__strataHookError = '';
  /* self-test: ยิง ping หนึ่งครั้งเพื่อพิสูจน์ว่าวนไป-กลับได้จริง */
  send('{PING_ACT}', {{}}).then(function(){{
    window.__strataHookSelfTest = 'ok';
  }}).catch(function(err){{
    window.__strataHookSelfTest = 'fail';
    window.__strataHookError = String(err && err.message || err);
  }});
}})();
"""


def install_hook(
    dispatch: Callable[[str, dict], Any],
    *,
    timeout_ms: int = 180_000,
    verify_s: float = 4.0,
    display_widgets: bool = True,
) -> dict:
    """ติดตั้งสะพาน JS→kernel แล้วคืนผลการติดตั้ง

    คืน dict: ``{"ok": bool, "token": str, "note": str, "widgets": (req, res)}``
    ``ok=False`` แปลว่า frontend นี้ใช้สะพานไม่ได้ → ผู้เรียกควรแสดง ``strata.hybrid`` แทน
    """
    widgets = _widgets()
    if widgets is None:
        return {"ok": False, "token": "", "note": "ไม่มี ipywidgets", "widgets": None}

    try:
        from IPython.display import Javascript, display  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "token": "", "note": f"ไม่มี IPython ({exc})", "widgets": None}

    token = secrets.token_hex(5)
    hidden = {
        "height": "1px", "width": "1px", "min_height": "0px",
        "opacity": "0.01", "overflow": "hidden", "padding": "0px", "margin": "0px",
    }
    req = widgets.Text(value="", layout=widgets.Layout(**hidden))
    res = widgets.HTML(value="{}", layout=widgets.Layout(**hidden))
    req.add_class(f"strata-hook-req-{token}")
    res.add_class(f"strata-hook-res-{token}")

    state = {"pinged": False, "requests": 0, "last_error": ""}
    lock = threading.Lock()

    def _work(body: dict) -> None:
        act = str(body.get("act") or "poll")
        payload = body.get("payload") or {}
        rid = str(body.get("id") or "")
        if act == PING_ACT:
            state["pinged"] = True
            result: Any = {"ok": True, "pong": True, "token": token}
        else:
            try:
                result = dispatch(act, payload)
            except Exception as exc:  # noqa: BLE001 — ห้ามให้ comm ตาย
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        try:
            payload_json = json.dumps({"id": rid, "result": result}, ensure_ascii=False, default=str)
            # escape เป็น HTML entity ก่อน — กันข้อความที่มี <div>/<script> ไปรบกวนการอ่าน textContent
            res.value = _html.escape(payload_json)
        except Exception as exc:  # noqa: BLE001
            state["last_error"] = str(exc)

    def _on_request(change) -> None:
        raw = str(change.get("new") or "").strip()
        if not raw:
            return
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            state["last_error"] = "payload ไม่ใช่ JSON"
            return
        state["requests"] += 1

        def _job() -> None:
            with lock:  # ทีละคำขอ — กัน state ของ kernel ชนกัน
                _work(body if isinstance(body, dict) else {})

        threading.Thread(target=_job, daemon=True, name="strata-jshook").start()

    req.observe(_on_request, names="value")

    if display_widgets:
        display(widgets.HBox([req, res]))
    display(Javascript(_hook_js(token, timeout_ms)))

    # ---- self-test: รอให้ __ping วนกลับมาถึง kernel ----
    note = ""
    ok = False
    deadline = time.time() + max(0.5, verify_s)
    while time.time() < deadline:
        if state["pinged"]:
            ok = True
            break
        time.sleep(0.15)
    if not ok:
        note = (
            "self-test ไม่ผ่าน (JS เรียก kernel ไม่ได้) — frontend นี้อาจบล็อกสคริปต์ใน output "
            "หรือแยก output ไว้คนละ iframe"
        )
        if state["last_error"]:
            note += f" · {state['last_error']}"
    else:
        note = "สะพาน JS→kernel พร้อมใช้งาน (ปุ่มในหน้าจอ HTML กดได้แล้ว)"
    return {"ok": ok, "token": token, "note": note, "widgets": (req, res), "state": state}


__all__ = ["install_hook", "PING_ACT"]
