#!/usr/bin/env python
"""결과 수집 서버(이 서버, 포트 9598) — 센터의 upload_results.py가 보내는 zip을 받아 저장·해제.

  nohup .venv/bin/python scripts/collect_server.py --port 9598 --dir outputs/collected > outputs/collect.log 2>&1 &
저장: outputs/collected/<exp>/<site>/<timestamp>/ (zip 원본 + 풀린 파일). 같은 센터가 다시 보내면 새 타임스탬프 폴더로 보존.
"""
import os, argparse, datetime, zipfile, io, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=9598); ap.add_argument("--dir", default="outputs/collected"); a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            q = parse_qs(urlparse(self.path).query); exp = q.get("exp", ["unknown"])[0]; site = q.get("site", ["unknown"])[0]; sha = q.get("sha", [""])[0]; run = q.get("run", ["norun"])[0]
            n = int(self.headers.get("Content-Length", 0)); data = self.rfile.read(n)
            got = hashlib.sha256(data).hexdigest()[:16]
            if sha and got != sha: self.send_response(400); self.end_headers(); self.wfile.write(b"sha mismatch"); return
            d = os.path.join(a.dir, exp, run, site, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")); os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f"export_{site}.zip"), "wb").write(data)
            try: zipfile.ZipFile(io.BytesIO(data)).extractall(d); msg = f"saved {n/2**20:.1f}MB → {d}"
            except zipfile.BadZipFile: msg = f"saved raw (bad zip) → {d}"
            print(datetime.datetime.now(), exp, site, msg, flush=True)
            self.send_response(200); self.end_headers(); self.wfile.write(msg.encode())

        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"CouinaudFL collect server OK")

        def log_message(self, *args): pass

    os.makedirs(a.dir, exist_ok=True); print(f"collect server on :{a.port} → {a.dir}", flush=True)
    HTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
