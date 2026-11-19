"""เสิร์ฟหน้าจอ HTML5 ของ Strata (Studio / ComfyUI) ให้แพลตฟอร์มที่ไม่มี Colab RPC

Colab มี ``google.colab.kernel.invokeFunction`` ให้ JS เรียกกลับเข้า kernel ได้ตรง ๆ
แต่ Kaggle/JupyterLab ไม่มี API แบบนั้น หน้าเว็บจึงคุยกับ kernel ไม่ได้ (เดิมเลยต้องใช้
ipywidgets UI คนละชุด ทำให้หน้าตาไม่เหมือน Colab)

โมดูลนี้แก้ด้วยการรัน HTTP server เล็ก ๆ ในเครื่อง (stdlib ล้วน) แล้ว:

* ``GET /<key>/``          → เสิร์ฟ HTML พร้อมสคริปต์ ``window.__strataRpc`` (fetch → kernel)
* ``POST /<key>/rpc``      → เรียก dispatcher ของเซลล์นั้นแล้วคืน JSON
* ``GET /health``          → ตรวจสุขภาพ server

บน Kaggle เบราว์เซอร์ของผู้ใช้อยู่คนละเครื่องกับ VM จึงเปิดช่องทางด้วย
cloudflared quick tunnel (มีตัวช่วยโหลดไบนารีอยู่แล้วใน ``imagegen``) แล้วฝัง iframe
ใน output ของเซลล์ — ได้หน้าจอเดียวกับ Colab ทั้งหน้าตาและการทำงาน

ความปลอดภัย: ทุกหน้าใช้ token สุ่ม เปิดเฉพาะ 127.0.0.1 ไม่ผูกกับ 0.0.0.0
และเตือนผู้ใช้เสมอว่าลิงก์ที่ได้สามารถสั่งงาน kernel ได้
"""

from __future__ import annotations

import contextlib
import http.server
import json
import secrets
import threading
import time
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

RPC_TIMEOUT_HINT = "ถ้าหน้าเว็บค้าง ให้รันเซลล์นี้ใหม่ (จะออก token/tunnel ใหม่)"

_PAGES: dict[str, dict] = {}
_LOCK = threading.Lock()
_SERVER: http.server.ThreadingHTTPServer | None = None
_SERVER_PORT = 0
_TOKEN = ""
_TUNNEL: dict[str, Any] = {"proc": None, "base": "", "url": ""}


def _page_script(key: str, token: str) -> str:
    url = f"{key}/rpc?t={token}"
    return (
        "<script>window.__strataRpcURL=" + json.dumps(url) + ";"
        "window.__strataRpc=async function(act,payload){"
        "const r=await fetch(window.__strataRpcURL,{method:'POST',headers:{'Content-Type':'application/json'},"
        "credentials:'same-origin',body:JSON.stringify({act:act,payload:payload||{}})});"
        "if(!r.ok){throw new Error('rpc '+r.status);}"
        "const j=await r.json();return (j&&Object.prototype.hasOwnProperty.call(j,'result'))?j.result:j;};"
        "</script>"
    )


def _standalone(html: str, key: str, token: str, title: str) -> str:
    script = _page_script(key, token)
    head = (
        "<!DOCTYPE html><html lang=\"th\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title>{script}"
        "<style>html,body{margin:0;padding:0;background:#0b1120;color:#f8fafc;}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}"
        "a{color:#60a5fa;}</style></head><body>"
    )
    return head + html + "</body></html>"


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "StrataStudio/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # เงียบ ไม่ถ่วง log ของ notebook
        pass

    # ---------- helpers ----------
    def _token_ok(self) -> bool:
        q = parse_qs(urlsplit(self.path).query)
        given = (q.get("t") or [""])[0] or self.headers.get("X-Strata-Token", "")
        return bool(given) and secrets.compare_digest(given, _TOKEN)

    def _send(self, code: int, body: str | bytes, ctype: str) -> None:
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(raw)

    def _send_json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False, default=str), "application/json; charset=utf-8")

    def _path_parts(self) -> tuple[str, str]:
        parts = [p for p in urlsplit(self.path).path.split("/") if p]
        if not parts:
            return "", ""
        return parts[0], (parts[1] if len(parts) > 1 else "")

    # ---------- routes ----------
    def do_GET(self) -> None:  # noqa: N802
        key, rest = self._path_parts()
        if rest == "" and key == "":
            links = "".join(
                f'<li><a href="{k}/?t={_TOKEN}">{v["title"]}</a></li>' for k, v in sorted(_PAGES.items())
            ) or "<li>(ยังไม่มีหน้าจอที่ลงทะเบียน)</li>"
            self._send(200, f"<html><body style='background:#0b1120;color:#f8fafc;font-family:sans-serif'>"
                            f"<h3>Strata Studio server</h3><ul>{links}</ul></body></html>", "text/html; charset=utf-8")
            return
        if key == "health":
            self._send_json(200, {"ok": True, "pages": sorted(_PAGES), "port": _SERVER_PORT})
            return
        page = _PAGES.get(key)
        if page is None:
            self._send(404, "unknown page", "text/plain; charset=utf-8")
            return
        if not self._token_ok():
            self._send(403, "token ไม่ถูกต้อง — เปิดลิงก์จาก output ของเซลล์ใหม่", "text/plain; charset=utf-8")
            return
        self._send(200, _standalone(page["html"], key, _TOKEN, page["title"]), "text/html; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        key, rest = self._path_parts()
        page = _PAGES.get(key)
        if page is None or rest != "rpc":
            self._send(404, "unknown rpc", "text/plain; charset=utf-8")
            return
        if not self._token_ok():
            self._send_json(403, {"error": "bad token"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"payload ไม่ใช่ JSON: {exc}"})
            return
        act = str(body.get("act") or "poll")
        payload = body.get("payload") or {}
        try:
            result = page["dispatch"](act, payload)
        except Exception as exc:  # noqa: BLE001 — ต้องคืน error ให้หน้าเว็บ ไม่ใช่ทำให้ server ล้ม
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, {"result": result})


def ensure_server(port: int = 0) -> tuple[int, str]:
    """สตาร์ท HTTP server (ครั้งเดียว) แล้วคืน (port, token)"""
    global _SERVER, _SERVER_PORT, _TOKEN
    with _LOCK:
        if _SERVER is not None and _TOKEN:
            return _SERVER_PORT, _TOKEN
        _TOKEN = secrets.token_urlsafe(18)
        last: Exception | None = None
        for cand in ([port] if port else []) + [0, 8765, 8766, 8767]:
            try:
                srv = http.server.ThreadingHTTPServer(("127.0.0.1", cand), _Handler)
            except OSError as exc:
                last = exc
                continue
            srv.daemon_threads = True
            _SERVER, _SERVER_PORT = srv, srv.server_address[1]
            threading.Thread(target=srv.serve_forever, name="strata-studio-http", daemon=True).start()
            return _SERVER_PORT, _TOKEN
        raise RuntimeError(f"เปิด HTTP server สำหรับ Studio ไม่ได้: {last}")


def register_page(key: str, html: str, dispatch: Callable[[str, dict], Any], title: str = "Strata Studio") -> str:
    """ลงทะเบียนหน้าจอ (เรียกซ้ำได้ — จะทับของเดิม) แล้วคืน path ภายใน ``/<key>/``"""
    _PAGES[key] = {"html": html, "dispatch": dispatch, "title": title}
    return f"{key}/"


def local_url(key: str, port: int | None = None) -> str:
    p = port or _SERVER_PORT
    return f"http://127.0.0.1:{p}/{key}/?t={_TOKEN}"


def start_quick_tunnel(local_base: str, wait_s: int = 90) -> str:
    """เปิด cloudflared quick tunnel ไปยัง server ในเครื่อง แล้วคืน URL สาธารณะ"""
    with _LOCK:
        if _TUNNEL["proc"] is not None and _TUNNEL["proc"].poll() is None and _TUNNEL["base"] == local_base:
            return _TUNNEL["url"]
    import subprocess

    from .imagegen import ensure_cloudflared, parse_cloudflared_url

    binary = ensure_cloudflared()
    proc = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", local_base],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    out: list[str] = []
    threading.Thread(target=lambda: [out.append(ln) for ln in proc.stdout or []], daemon=True).start()
    deadline = time.time() + wait_s
    url = ""
    while time.time() < deadline:
        url = parse_cloudflared_url("".join(out))
        if url:
            break
        if proc.poll() is not None:
            break
        time.sleep(0.4)
    if not url:
        with contextlib.suppress(Exception):
            proc.terminate()
        tail = "".join(out[-8:]).strip()[-400:] or "(ไม่มี output)"
        raise RuntimeError(f"cloudflared เปิด tunnel ไม่ได้ภายใน {wait_s}s\n{tail}")
    _TUNNEL.update({"proc": proc, "base": local_base, "url": url})
    return url


def stop_tunnel() -> bool:
    proc = _TUNNEL.get("proc")
    stopped = False
    if proc is not None and proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.terminate()
            stopped = True
    _TUNNEL.update({"proc": None, "base": "", "url": ""})
    return stopped


def iframe_html(url: str, title: str = "Strata Studio", height: int = 1000, note: str = "") -> str:
    """การ์ดฝังหน้าจอใน output ของเซลล์ + ปุ่มเปิดแท็บใหม่ (ใช้เมื่อไม่มี Colab proxyPort)"""
    extra = f'<span style="opacity:.75;font-size:12px">{note}</span>' if note else ""
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;color:#e2e8f0">'
        '<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:2px 0 8px 2px">'
        f'<a href="{url}" target="_blank" rel="noopener" '
        'style="background:#2563eb;color:#fff;text-decoration:none;padding:9px 14px;border-radius:8px;'
        f'font-weight:700;font-size:13px">↗ เปิด {title} ในแท็บใหม่</a>{extra}</div>'
        f'<iframe src="{url}" allow="clipboard-read; clipboard-write; fullscreen" '
        f'style="width:100%;height:{height}px;border:1px solid #334155;border-radius:10px;background:#0b1120" '
        f'title="{title}"></iframe></div>'
    )


def mount_page(
    key: str,
    html: str,
    dispatch: Callable[[str, dict], Any],
    *,
    title: str = "Strata Studio",
    height: int = 1000,
    port: int = 0,
    use_tunnel: bool | None = None,
    wait_s: int = 90,
) -> dict:
    """ลงทะเบียนหน้าจอ + สตาร์ท server (+ tunnel ถ้าจำเป็น) แล้วคืนข้อมูลสำหรับฝังใน notebook

    ``use_tunnel=None`` → เปิด tunnel อัตโนมัติเมื่อเบราว์เซอร์ไม่ได้อยู่เครื่องเดียวกับ kernel
    (Kaggle/เครื่องอื่น) และใช้ 127.0.0.1 ตรง ๆ เมื่อเป็นเครื่องเดียวกัน
    """
    register_page(key, html, dispatch, title=title)
    srv_port, token = ensure_server(port)
    base = f"http://127.0.0.1:{srv_port}"
    if use_tunnel is None:
        from .platforms import detect

        use_tunnel = detect()["name"] != "local"
    public = ""
    mode = "local"
    error = ""
    if use_tunnel:
        print(f"⏳ กำลังเปิด cloudflared tunnel สำหรับหน้าจอ {title} (ครั้งแรกอาจ 20-60s)…", flush=True)
        try:
            public = start_quick_tunnel(base, wait_s=wait_s)
            mode = "tunnel"
            print(f"✅ เปิด tunnel แล้ว: {public}", flush=True)
        except Exception as exc:  # noqa: BLE001 — ตกไปใช้ localhost อย่างน้อยที่สุดก็ไม่ล้ม
            error = f"{type(exc).__name__}: {exc}"
            public = ""
    url = f"{public}/{key}/?t={token}" if public else local_url(key, srv_port)
    note = ""
    if mode == "tunnel":
        note = "ลิงก์ส่วนตัว (มี token) — อย่าแชร์ เพราะสั่งงาน kernel ได้"
    elif use_tunnel:
        note = "⚠️ ใช้ 127.0.0.1 ไม่ได้ถ้าเบราว์เซอร์อยู่คนละเครื่องกับ kernel"
    return {
        "key": key, "port": srv_port, "mode": mode, "url": url, "error": error,
        "local_url": local_url(key, srv_port), "public_url": public,
        "iframe": iframe_html(url, title=title, height=height, note=note),
    }


def shutdown() -> None:
    global _SERVER, _SERVER_PORT, _TOKEN
    stop_tunnel()
    srv = _SERVER
    _SERVER = None
    _SERVER_PORT = 0
    _TOKEN = ""
    _PAGES.clear()
    if srv is not None:
        with contextlib.suppress(Exception):
            srv.shutdown()
            srv.server_close()


__all__ = [
    "ensure_server", "register_page", "local_url", "start_quick_tunnel", "stop_tunnel",
    "iframe_html", "mount_page", "shutdown", "RPC_TIMEOUT_HINT",
]
