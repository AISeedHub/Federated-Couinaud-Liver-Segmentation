#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""실행 전 서버 연결 점검 (센터에서 run_center 실행 **전에** 1회).

FL 포트는 single(단일센터 5-fold)이 끝난 뒤에야 처음 접속하므로, 방화벽이 막혀 있으면
몇 시간을 버린 뒤에 알게 된다. 이 스크립트는 그 세 관문을 10초 안에 미리 확인한다.

  Linux : .venv/bin/python scripts/check_server.py
  Windows: .venv\\Scripts\\python.exe scripts\\check_server.py
"""
import os, re, sys, socket, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg_get(path, key):
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.match(r'^%s:\s*"?([^"#\s]+)"?' % re.escape(key), line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def tcp(host, port, timeout=5):
    try:
        socket.create_connection((host, int(port)), timeout).close()
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp4c,exp5c", help="점검할 실험 이름(콤마 구분, configs/<exp>.yaml)")
    args = ap.parse_args()
    print("=== 서버 연결 점검 ===")
    targets, upload_url, unresolved = [], None, []
    for exp in [e.strip() for e in args.exp.split(",") if e.strip()]:
        p = os.path.join(HERE, "configs", f"{exp}.yaml")
        if not os.path.exists(p):
            continue
        addr = cfg_get(p, "client_server_address") or cfg_get(p, "server_address")
        uurl = cfg_get(p, "upload_url")
        if uurl and "SERVER_IP" not in uurl:
            upload_url = uurl
        if addr:
            host, _, port = addr.rpartition(":")
            # 센터 배포본은 클라이언트가 붙을 실제 주소여야 한다
            if host in ("0.0.0.0", "", "SERVER_IP"):
                print(f"[{exp}] 클라이언트 접속 주소={addr} — SERVER_IP를 실제 서버 주소로 치환 필요")
                unresolved.append(exp)
                continue
            targets.append((exp, host, port))

    if not targets and not upload_url:
        print("\n결과: 설정에 실제 서버 주소가 없습니다. configs/*.yaml 의 SERVER_IP를 치환한 뒤 다시 실행하세요.")
        return 2

    ok = not unresolved
    for exp, host, port in targets:
        good, err = tcp(host, port)
        print(f"[{exp}] FL {host}:{port} … {'OK' if good else 'FAIL  ' + err}")
        ok &= good

    if upload_url:
        base = upload_url.rsplit("/", 1)[0]
        host = re.sub(r"^https?://", "", base).split("/")[0]
        h, _, p = host.rpartition(":")
        good, err = tcp(h or host, p or 80)
        print(f"[수집] {host} … {'OK' if good else 'FAIL  ' + err}")
        ok &= good
        if good:  # 가중치 엔드포인트까지 확인 (실제 다운로드는 하지 않음)
            try:
                req = urllib.request.Request(base + "/weights", headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    r.read(1)
                    print(f"[가중치] {base}/weights … OK")
            except Exception as e:
                print(f"[가중치] {base}/weights … 확인 실패 ({type(e).__name__}) — 첫 실행 시 GitHub 릴리스로 폴백됩니다")

    print("\n결과: " + ("모든 관문 통과 — run_center 실행 가능" if ok else
                      "실패 항목 있음 — 방화벽/서버 기동 확인 후 재점검 (이 상태로 실행하면 single 종료 후 FL 단계에서 멈춥니다)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
