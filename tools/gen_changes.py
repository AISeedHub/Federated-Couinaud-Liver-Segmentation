#!/usr/bin/env python3
"""센터 배포용 변경 기록 생성기.
기준(tag centers-baseline = 센터들이 현재 실행 중인 커밋) 대비 배포 대상 파일의 변경을
CHANGES_FOR_CENTERS.md 에 기록한다. 수정 후 `python tools/gen_changes.py` 실행.
  - 수정 파일: unified diff 전문
  - 신규 파일: 전체 내용
  - configs/*.yaml 은 상황별로 사용자가 직접 관리하므로 기록 제외
  - docs/변경메모.md 가 있으면 상단에 '변경 이유' 로 포함
"""
import subprocess, os, datetime, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(REPO)
BASE = "centers-baseline"
PATHS = [":(exclude)configs", "couinaudfl", "scripts", "tools", "pyproject.toml", "README.md"]
def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True).stdout
base_sha = git("rev-parse", "--short", BASE).strip()
mod = [l for l in git("diff", BASE, "--name-status", "--", *PATHS).splitlines() if l]
new = [l for l in git("ls-files", "--others", "--exclude-standard", "--", *PATHS).splitlines() if l and not l.endswith(".pyc")]
out = [f"# 센터 배포용 변경 기록", "",
       f"- 기준 커밋: `{base_sha}` (센터들이 현재 실행 중인 버전, tag `{BASE}`)",
       f"- 생성 시각: {datetime.datetime.now():%Y-%m-%d %H:%M}", f"- 생성 명령: `python tools/gen_changes.py`", ""]
if os.path.exists("docs/변경메모.md"):
    out += ["## 변경 이유(수동 메모)", "", open("docs/변경메모.md", encoding="utf-8").read().strip(), ""]
out += ["## 파일 목록", ""]
if not mod and not new: out.append("(기준 대비 변경 없음)")
for l in mod:
    st, p = l.split("\t", 1); out.append(f"- **{'수정' if st.startswith('M') else '삭제' if st.startswith('D') else st}** `{p}`")
for p in new: out.append(f"- **신규** `{p}`")
out.append("")
for l in mod:
    st, p = l.split("\t", 1)
    out += [f"## 수정: `{p}`", "", "센터에서 이 diff를 그대로 적용(`-` 줄 제거, `+` 줄 추가).", "", "```diff", git("diff", BASE, "--", p).rstrip(), "```", ""]
for p in new:
    ext = os.path.splitext(p)[1].lstrip(".") or "text"
    body = open(p, encoding="utf-8", errors="replace").read().rstrip() if os.path.getsize(p) < 200_000 else f"(파일이 커서 생략, {os.path.getsize(p)} bytes — 직접 복사)"
    out += [f"## 신규: `{p}`", "", "센터에 같은 경로로 파일 전체를 생성.", "", f"```{ext}", body, "```", ""]
open("CHANGES_FOR_CENTERS.md", "w", encoding="utf-8").write("\n".join(out))
print(f"CHANGES_FOR_CENTERS.md 갱신: 수정 {len(mod)} / 신규 {len(new)} (기준 {base_sha})")
