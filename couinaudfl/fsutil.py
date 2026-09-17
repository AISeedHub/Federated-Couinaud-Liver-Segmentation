# -*- coding: utf-8 -*-
"""파일시스템 안전 유틸 — 디스크 풀 상황에서의 조용한 오염 방지.
write_marker: DONE 마커를 원자적으로(임시파일→fsync→rename) 기록하고 내용 존재를 검증.
check_free_gb: 경로의 가용 GB. ensure_free: 미달 시 명확한 예외."""
from __future__ import annotations
import os, shutil, tempfile


def write_marker(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".marker_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        if os.path.getsize(tmp) == 0:
            raise IOError(f"marker 기록 실패(0바이트): {path}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except OSError: pass
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"marker 검증 실패: {path}")


def check_free_gb(path: str) -> float:
    return shutil.disk_usage(os.path.abspath(path)).free / 1e9


def ensure_free(path: str, min_gb: float, what: str = "") -> None:
    free = check_free_gb(path)
    if free < min_gb:
        raise RuntimeError(f"DISK_LOW {what}: 가용 {free:.1f}GB < 요구 {min_gb}GB ({path}) — 디스크 정리 후 재실행")
