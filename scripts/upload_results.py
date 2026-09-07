#!/usr/bin/env python
"""익명화된 내보내기 폴더(export_<site>)를 zip으로 묶어 서버 수집기(collect_server.py)로 전송.
실제 환자 ID는 export 폴더에 없음(export_results.py가 Case N으로 치환, _id_map_LOCAL_ONLY.json은 client_ 폴더에만 존재).

  python scripts/upload_results.py --exp exp4c --site A [--url http://168.131.153.57:9598/upload]
실패 시(망 차단 등) zip 파일은 outputs/<exp>/export_<site>.zip 으로 남으니 수동 전달 가능.
"""
import os, sys, argparse, shutil, urllib.request, json, datetime, hashlib


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--site", required=True)
    ap.add_argument("--url", default=None, help="기본: configs/<exp>.yaml의 upload_url, 없으면 http://168.131.153.57:9598/upload"); ap.add_argument("--out", default="outputs"); ap.add_argument("--run", required=True); a = ap.parse_args()
    if not a.url:
        cfg = f"configs/{a.exp}.yaml"; url = None
        if os.path.exists(cfg):
            for line in open(cfg, encoding="utf-8"):
                if line.startswith("upload_url:"): url = line.split(":", 1)[1].strip().strip('"').strip("'")
        a.url = url or "http://168.131.153.57:9598/upload"
    ex = os.path.join(a.out, a.exp, a.run, f"export_{a.site}")
    if not os.path.isdir(ex): print("export 폴더 없음 — 먼저 export_results.py 실행"); sys.exit(1)
    assert not os.path.exists(os.path.join(ex, "_id_map_LOCAL_ONLY.json")), "ID 매핑 파일이 export 폴더에 있음 — 전송 중단"
    zp = shutil.make_archive(os.path.join(a.out, a.exp, a.run, f"export_{a.site}"), "zip", ex); size = os.path.getsize(zp)
    sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()[:16]
    req = urllib.request.Request(a.url + f"?exp={a.exp}&site={a.site}&run={a.run}&sha={sha}", data=open(zp, "rb").read(), method="POST",
                                 headers={"Content-Type": "application/zip", "Content-Length": str(size)})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: print(f"업로드 완료 {size/2**20:.1f} MB sha {sha} → {r.read().decode()[:200]}")
    except Exception as e:
        print(f"업로드 실패({type(e).__name__}: {e}). zip은 {zp} 에 있음 — 수동 전달"); sys.exit(2)


if __name__ == "__main__":
    main()
