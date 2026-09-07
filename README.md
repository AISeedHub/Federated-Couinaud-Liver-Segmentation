# Federated Learning for Automatic Segmentation of Nine Couinaud Liver Segments across Four Institutions: A Feasibility Study

<p align="center">
  <img src="assets/figure1.jpg" width="90%" alt="Federated learning pipeline">
</p>


복부 CT에서 Couinaud 간 9분절(I, II, III, IVa, IVb, V–VIII)을 자동 분할하는 다기관 연합학습 프레임워크.

- **모델**: nnU-Net v2의 3D full-resolution U-Net(PlainConvUNet, 31M 파라미터) 구조를 사용하고, 인코더·디코더를 TotalSegmentator(Wasserthal et al., *Radiology: AI* 2023) 공식 가중치로 초기화한다.
- **학습 3단계**: ① 공개 CT 데이터셋(8분절 라벨, 634례)에서 사전학습 → ② 각 기관 데이터로 단일기관 학습과 연합학습(FedAvg · FedProx · FedAdam · FedBN-IN, Flower 기반, 기관별 5-fold 교차검증) → ③ 공개 벤치마크(MSD/Tian)와 외부 코호트(TCIA CRLM, MRI)에서 검증.
- 원자료(CT)는 각 기관 밖으로 나가지 않으며, 서버와는 모델 파라미터만 교환한다. 결과 반출물은 환자 식별자를 자동 익명화(Case N)한다.


---

## 1. 설치 (모든 센터 공통)

### 사전 조건
- NVIDIA 드라이버: `nvidia-smi` 우측 상단 **CUDA Version 12.1 이상** (13.x 포함). GPU 메모리 24 GB 이상.
- 서버 `168.131.153.57`의 **9595(exp4c) / 9596(exp5c)** 포트로 나가는 연결 허용.
- Python 별도 설치 불필요(uv가 3.12를 내려받음). Git이 없으면 GitHub에서 ZIP으로 받아 풀어도 된다.

### Windows (cmd 또는 PowerShell)
```bat
git clone https://github.com/AISeedHub/Federated-Couinaud-Liver-Segmentation.git
cd Federated-Couinaud-Liver-Segmentation
scripts\install.bat
```
PowerShell에서는 `.\scripts\install.bat` 처럼 앞에 `.\`를 붙인다(실행 정책 변경 불필요). 이후 모든 `.bat` 실행도 동일.
### Linux / DGX Spark
```bash
git clone https://github.com/AISeedHub/Federated-Couinaud-Liver-Segmentation.git
cd Federated-Couinaud-Liver-Segmentation
bash scripts/install.sh
```
설치 스크립트가 하는 일: uv 설치 → `.venv` 생성 → `nvidia-smi`의 CUDA 버전으로 torch 휠 인덱스 선택(cu121/cu126/cu128/cu130, ARM은 cu130) → 의존성 설치 → GPU 인식 출력.
마지막 줄이 `torch 2.x.x+cuXXX cuda True <GPU명>` 이면 정상. `cuda False`면 드라이버가 낮거나 CPU 휠이 깔린 것 → 드라이버 업데이트 후 `.venv` 삭제하고 재설치.

### 사전학습 가중치 배치
서버에서 전달한 `best.pth`(~125 MB)를 **`outputs/pretrain/best.pth`** 에 둔다(폴더가 없으면 만든다). 설정 파일의 `init_weights`가 이 경로를 가리킨다.
```
CouinaudFL/
└── outputs/
    └── pretrain/
        └── best.pth
```

---

## 2. 데이터 포맷 (v1과 동일)
```
<data>/
├── 00012345/          ← 환자 폴더(이름은 임의, 산출물에는 Case N으로만 나감)
│   ├── image.npy      uint8 (D, 512, 512)   두부→미부 80슬라이스, 4 mm, 분당 베이크드 강도(WL80/WW225 상당)
│   └── mask.npy       uint8 (18, D, 512, 512)  ch0 배경, ch1–9 = S1 S2 S3 S4a S4b S5 S6 S7 S8, ch10–17 병변(무시)
├── 00023456/
└── ...
```
**부피(mL) 계산용 픽셀 간격**: 데이터 루트에 `spacing.csv`를 두면 환자별 면내 간격을 쓴다(없으면 분당 기본값 0.7305 mm로 계산하고 카탈로그의 `spacing_source`가 `default`로 표시됨 — Dice·HD95에는 영향 없고 mL 값만 달라진다).
```
<data>/spacing.csv
case,spacing_y,spacing_x
00012345,0.7305,0.7305
00023456,0.6836,0.6836
```
값은 원본 DICOM의 PixelSpacing(mm). z는 전처리 규약상 4 mm로 고정이라 적지 않는다. 환자 폴더에 `meta.json`(`{"spacing":[4.0, sy, sx]}`)이 있으면 그것이 우선한다.

**v1에서 만든 센터별 부피 CSV(`volumes_per_patient.csv`: `patient_id, orig_matrix, pixel_spacing, …`)가 있으면 그대로 쓴다.** 데이터 루트에 그 파일명으로 두거나, 다른 위치면 환경변수로 지정:
```bat
set COUINAUD_SPACING_CSV=D:\data\volumes_per_patient.csv
```
```bash
export COUINAUD_SPACING_CSV=/data/volumes_per_patient.csv
```
512 기준 간격 = `pixel_spacing × orig_matrix / 512` 로 자동 환산된다(예: 1024 매트릭스 0.28125 → 0.5625 mm). 탐색 순서: 환경변수 → `<data>/spacing.csv` → `<data>/volumes_per_patient.csv` → 기본값. 카탈로그의 `spacing_source` 열에서 어느 것이 쓰였는지 확인할 수 있다.

첫 실행 때 각 환자 폴더에 `label.npy`(argmax 캐시, 21 MB)를 자동 생성한다. 데이터 폴더에 쓰기 권한이 없으면 환경변수 `COUINAUD_LABEL_CACHE=<쓰기 가능한 폴더>` 를 지정한다.

데이터 확인:
```bat
.venv\Scripts\python -c "from couinaudfl.data import list_cases; c=list_cases(r'D:\data\liver'); print(len(c), '명')"
```

---

## 3. 클라이언트(센터) 실행

### 3-1. 한 줄 실행 (권장)
단일센터 5-fold → FL 클라이언트(fold 0–4 × 방법론 4종) 순으로 **자동** 진행. 서버 세션 전환·재접속·재시작을 스스로 처리한다.

센터별 실제 명령(전부 4센터 exp4c → 5센터 exp5c 순차; 전남대는 exp5c만):

**Windows** — 순천향천안 A · 고려대안산 B · 강릉아산 C (cmd 또는 PowerShell, PowerShell은 앞에 `.\`)
```bat
cd C:\Federated-Couinaud-Liver-Segmentation
scripts\run_center_seq.bat D:\data\liver A exp4c exp5c
```
```bat
scripts\run_center_seq.bat D:\data\liver B exp4c exp5c
```
```bat
scripts\run_center_seq.bat D:\data\liver C exp4c exp5c
```
창을 닫지 말 것(닫으면 60초 후 재시작 로직도 함께 종료됨). 절전·화면 잠금은 스크립트가 해제한다.
재부팅 등으로 중단됐을 때 **이어서** 하려면 run 이름을 환경변수로 주고 같은 명령을 다시 친다(완료분은 건너뜀):
```bat
set RUN_NAME=run_20260903_101500      ← outputs\exp4c\LAST_RUN 파일 내용
scripts\run_center_seq.bat D:\data\liver A exp4c exp5c
```

**DGX Spark (Linux aarch64)** — 전남대 E, exp5c만
```bash
cd ~/Federated-Couinaud-Liver-Segmentation
nohup bash scripts/run_center.sh exp5c /home/crex/fedlr/LiverSegmentation/merged E > /dev/null 2>&1 &
tail -f outputs/exp5c/$(cat outputs/exp5c/LAST_RUN)/client_E/run_center.log
```
exp4c가 끝나 서버가 9596을 열 때까지 "연결 대기"를 반복하는 것이 정상이며, **며칠이 걸려도 무해하다**(30초마다 접속 시도 한 번, GPU·CPU 미사용). 미리 켜 두지 않고 exp4c가 끝날 즈음, 또는 그 이후에 켜도 된다 — 서버는 5개 센터가 모일 때까지 시작하지 않는다.
재부팅에도 자동 복구하려면(선택):
```bash
crontab -e   # 아래 한 줄 추가
@reboot cd ~/Federated-Couinaud-Liver-Segmentation && RUN_NAME=$(cat outputs/exp5c/LAST_RUN 2>/dev/null) bash scripts/run_center.sh exp5c /home/crex/fedlr/LiverSegmentation/merged E
```

**Linux x86** — 분당서울대 D
```bash
cd ~/Federated-Couinaud-Liver-Segmentation
nohup bash scripts/run_center_seq.sh /data/liver D exp4c exp5c > /dev/null 2>&1 &
tail -f outputs/exp4c/$(cat outputs/exp4c/LAST_RUN)/client_D/run_center.log
```
Linux에서 SSH 세션이 끊겨도 `nohup ... &`로 띄운 프로세스는 계속 돈다. 재부팅 후 이어서 하려면:
```bash
RUN_NAME=$(cat outputs/exp4c/LAST_RUN) nohup bash scripts/run_center_seq.sh /data/liver D exp4c exp5c > /dev/null 2>&1 &
```

단일 실험만 돌릴 때:
```bat
scripts\run_center.bat exp4c D:\data\liver A
```
```bash
nohup bash scripts/run_center.sh exp4c /data/liver D > /dev/null 2>&1 &
```
인자: `<실험> <데이터 폴더> <센터 코드> [run 이름]`. 센터 코드는 A(순천향천안) B(고려대안산) C(강릉아산) D(분당서울대) E(전남대).

**실행 단위 분리**: 실행할 때마다 `outputs\<실험>\<run>\` 아래에 별도로 저장된다(run 기본값 = 시작 시각, 예 `run_20260903_101500`). 가중치·지표가 이전 실행을 덮어쓰지 않는다. 중단 후 이어서 하려면 같은 run 이름을 4번째 인자로 준다(마지막 run 이름은 `outputs\<실험>\LAST_RUN`에 기록됨).

**여러 실험 순차 실행** (4센터 exp4c → 5센터 exp5c):
```bat
scripts\run_center_seq.bat D:\data\liver A exp4c exp5c
```
```bash
nohup bash scripts/run_center_seq.sh /data/merged A exp4c exp5c > /dev/null 2>&1 &
```
exp4c가 끝나면 자동으로 exp5c(포트 9596)에 접속한다. exp5c는 `run_single: 0`이라 단일센터 재학습 없이 FL만 수행(전남대 E는 exp5c만 실행하며 필요 시 1로 둔다). 서버는 두 실험 서버를 동시에 띄워 두면 5센터가 모두 exp4c를 마치고 접속하는 시점에 exp5c가 자연히 시작된다.

동작 순서
0. `patient_catalog.py` — 전 환자 메타·GT 부피 카탈로그.
1. `single.py` — fold 0→4 **진짜 로컬**(서버 통신 없음) 단일센터 학습(각 fold: 학습 → best.pth로 test 평가 → CSV·예측 저장). 완료 fold는 `DONE` 마커로 건너뜀. 끝나면 로컬 모델·지표를 즉시 익명 내보내기+서버 업로드.
2. `client.py` — fold 0 FedAvg → FedProx → FedAdam → FedBN → fold 1 … 순으로 서버에 접속. 서버가 아직 그 세션을 안 열었으면 30초 간격으로 재시도하며 대기.
3. 프로세스가 죽으면 60초 후 자동 재시작(완료분은 건너뜀). Ctrl+C·터미널 클릭은 무시된다.

**로그**: `outputs\<실험>\<run>\client_<센터>\run_center.log`, `single.log`, `client.log`, 예외는 `errors.log`.
**중단**: `outputs\<실험>\<run>\client_<센터>\STOP.txt` 파일을 만들면 현재 세션이 끝난 뒤 정상 종료. 다시 시작하려면 STOP.txt 삭제 후 같은 명령.

### 3-2. 단계별 수동 실행
```bat
REM 단일센터만 (fold 지정 가능)
.venv\Scripts\python scripts\single.py --config configs\exp4c.yaml --data D:\data\liver --site A
.venv\Scripts\python scripts\single.py --config configs\exp4c.yaml --data D:\data\liver --site A --folds 0 1

REM FL 클라이언트만 (서버와 같은 fold·방법론 순서여야 함)
.venv\Scripts\python scripts\client.py --config configs\exp4c.yaml --data D:\data\liver --site A
.venv\Scripts\python scripts\client.py --config configs\exp4c.yaml --data D:\data\liver --site A --folds 2 --methods FedAvg
.venv\Scripts\python scripts\client.py --config configs\exp4c.yaml --data D:\data\liver --site A --server 168.131.153.57:9595
```
Linux는 `.venv/bin/python` 으로 바꾸면 동일.

### 3-3. 결과 내보내기·전송 (실제 환자 ID 제거) — `run_center`가 완료 시 자동 수행
수동으로 할 때:
```bat
.venv\Scripts\python scripts\export_results.py --exp exp4c --site A --run run_20260903_101500
.venv\Scripts\python scripts\upload_results.py --exp exp4c --site A --run run_20260903_101500
```
`export_results.py`가 `outputs\exp4c\<run>\export_A\`(Case N 익명)를 만들고, `upload_results.py`가 그 폴더를 zip으로 묶어 서버 수집기(`168.131.153.57:9598`)로 보낸다. 전송이 막히면 `outputs\exp4c\<run>\export_A.zip`이 남으므로 그 파일만 수동 전달하면 된다.
카탈로그(`patient_catalog.csv`: 환자별 슬라이스 수·spacing·분절/간/병변 GT 부피·간 z범위·강도)는 `run_center` 시작 시 자동 생성되어 함께 내보내진다.
실제 ID ↔ Case N 대응표는 `outputs\exp4c\client_A\_id_map_LOCAL_ONLY.json` 에만 남고 **센터 밖으로 보내지 않는다**(업로드 스크립트가 이 파일이 export 폴더에 있으면 전송을 거부한다).
내보내기 내용: 클라이언트 run 폴더 **전체**(모든 로그 run_center/single/client/errors, fold 분할, DONE 마커, 학습 이력, 라운드 기록, test 지표 CSV, **모든 test 예측 라벨맵(npz)**, 환자 카탈로그, 단일센터 best.pth·FL global 가중치, 설정 yaml, 코드 버전)를 텍스트·파일명까지 Case N으로 치환해 복사한다(재개용 옵티마 상태 `*last.pth`만 제외). 이후 어떤 지표·그림도 센터 재방문 없이 재계산 가능.

### 3-4. 정상 동작 확인 포인트
- `client.log`에 `Received: get_parameters` → 서버 연결 성공. 이후 `{"epoch": ...}` 줄이 라운드마다 찍힘.
- `{"round": k, "val_dice": ...}` 줄이 라운드마다 하나. 마지막 라운드에 `test_...` 항목이 붙음.
- `outputs\<실험>\client_<센터>\fold<k>\<방법론>\` 에 `test_metrics.csv`, `test_summary.json`, `global_final.pth`, `global_rXX.pth`.
- `연결 대기 (...) 30s 후 재시도` 가 계속되면: 서버가 다른 fold/방법론 세션 중이거나(정상, 기다리면 됨) 방화벽/IP 문제.

---

## 4. 서버 실행 (이 서버, 168.131.153.57)

**한 번의 실행으로 전부** — 4센터 exp4c(9595) 완료 후 5센터 exp5c(9596) 순차 + 결과 수집 서버(9598) 내장:
```bash
cd /home/dspserver/2025/jin/CouinaudFL   # 서버 작업본
nohup .venv/bin/python scripts/server.py --config configs/exp4c.yaml configs/exp5c.yaml --run run_main > outputs/server_run_main.out 2>&1 &
```
센터 쪽은 `run_center_seq ... exp4c exp5c` 한 줄이므로 양쪽 모두 명령 하나씩이다. 순서: exp4c fold 0–4 × 4방법론 → 모두 끝나면 exp5c 서버가 열리고, 5개 센터가 접속하는 대로 시작.
실험이 모두 끝나도 서버 프로세스는 **결과 수집기(9598)를 켠 채 대기**한다(센터의 최종 업로드가 그 뒤에 오므로). 모든 센터의 업로드가 `outputs/collected/`에 들어온 것을 확인한 뒤 `outputs/STOP_SERVER.txt`를 만들면 종료된다.
라운드 제한시간 `round_timeout_sec`(기본 3시간): 라운드 중 응답이 끊긴 센터를 그 시간까지만 기다리고 나머지 센터로 집계해 다음 라운드로 넘어간다. 죽은 센터는 스스로 재시작·재접속해 다음 라운드부터 다시 참여한다.

부분 실행·수동 운영:
```bash
.venv/bin/python scripts/server.py --config configs/exp4c.yaml --run run_main --folds 0 --methods FedAvg   # 일부만
.venv/bin/python scripts/server.py --config configs/exp5c.yaml --run run_main --collect-port 0            # 수집기 없이
nohup .venv/bin/python scripts/collect_server.py --port 9598 --dir outputs/collected > outputs/collect.log 2>&1 &   # 수집기만 따로
```
동작: (fold, method) 세션마다 Flower 서버를 새로 열고 `min_clients`만큼 접속하면 `rounds`×`local_epochs` 학습 → 완료 시 `DONE_<method>` 마커 → 다음 세션. 재실행하면 완료 세션은 건너뜀.
초기 파라미터는 `init_weights`(사전학습)로 배포하므로 모든 센터·방법론이 같은 시작점에서 출발한다.

**로그·산출물**: `outputs/<실험>/<run>/server/server.log`, `fold<k>/global_<method>.pth`(최종), `global_<method>_rXX.pth`(라운드별), `round_history_<method>.json`(라운드별 클라이언트 loss·val Dice·최종 test 요약).

세션 상태 보기:
```bash
tail -f outputs/exp4c/run_main/server/server.log
```

---

## 5. 설정 파일 (`configs/*.yaml`)
| 키 | 의미 |
|---|---|
| `experiment` | 출력 폴더 이름 (`outputs/<experiment>/`) |
| `server_address` / `client_server_address` | 서버 바인드 주소 / 클라이언트가 접속할 주소:포트 |
| `min_clients` | 라운드 시작에 필요한 센터 수 (exp4c 4, exp5c 5) |
| `rounds`, `local_epochs` | 총 에폭 = rounds × local_epochs (단일센터도 동일 에폭) |
| `methods`, `folds` | 실행 순서 — **서버와 클라이언트가 같아야 함** |
| `amp` | 0 = f32 (센터 24 GB 기준 13.6 GB), 1 = AMP |
| `init_weights` | 사전학습 가중치 경로 |
| `round_timeout_sec` | 라운드 제한시간(초). 응답 없는 센터를 이 시간까지만 기다림 |
| `run_single` | 1=단일센터 5-fold 수행, 0=생략(exp5c) |
| `upload_url` | 결과 수집 서버 주소 |
코드는 고정이고 실험 조건은 이 파일로만 바꾼다.

---

## 6. 이 서버 전용 스크립트
```bash
scripts/pretrain.py --mode pretrain                 # 공용 634건 사전학습 + 외부 테스트(Lee-50, MedSeg-9, MR, CRLM)
scripts/pretrain.py --mode tian5fold --fold 0       # TS 570 프로토콜(Tian 5-fold) 재현
scripts/pretrain.py --mode mr5fold --fold 0 --ts /data/datasets/couinaud_public/checkpoints/ts/730   # MR 모델
scripts/eval_ts_zero_shot.py --task liver_segments --cases <폴더>       # TS 570 zero-shot
scripts/eval_gunetr_zero_shot.py --cases <폴더>                          # G-UNETR++ zero-shot (GT 간 마스크 oracle)
scripts/run_rehearsal.sh                                                 # 정식 모의시험(서버+모의센터 2) + verify_run.py
scripts/run_seq_test.sh                                                  # 순차 실행(seq1→seq2) 통합 테스트 + 수집기 대조
scripts/run_crash_test.sh                                                # 라운드 도중 클라이언트 강제 종료 → 서버 진행·재합류 검증
python tools/gen_changes.py                                              # 센터 배포용 변경기록
```

## 7. 문제 해결
| 증상 | 조치 |
|---|---|
| `cuda False` / `[오류] GPU를 잡지 못했습니다` | 드라이버 CUDA 버전 확인(12.1+) 후 `scripts\install.bat` 재실행. run_center가 시작 전에 자동 점검하므로 CPU로 몰래 돌지 않음 |
| `undefined symbol: nccl...` 또는 torch가 사라짐 | **`uv sync`·`uv run`을 쓰지 말 것** — pyproject에 torch가 없어 제거/교체됨. 항상 `.venv\Scripts\python`(스크립트가 자동 사용)으로 실행하고, 꼬였으면 install 스크립트 재실행 |
| `연결 대기` 반복 | 서버 세션 대기 중이면 정상. 5분 이상이면 `client_server_address`·방화벽 확인 |
| `CUDA out of memory` | 다른 GPU 프로세스 종료. 그래도 부족하면 yaml `amp: 1` (서버와 협의) |
| 학습 중 예외 | `errors.log` 확인 후 전달. 프로세스는 자동 재시작됨 |
| 처음부터 다시 | 새 run 이름으로 실행(이전 run은 그대로 보존) |

## 8. 설계 기록
`docs/설계_v2.md` — 모델·데이터·오염 감사·평가 매트릭스 결정 근거.

---

## 9. 전체 커맨드 목록 (스크립트별 1개 예시)
Windows는 `.venv\Scripts\python`, Linux는 `.venv/bin/python`. `<run>`은 실행 이름(예 `run_20260903_101500`).

| 스크립트 | 예시 | 용도 |
|---|---|---|
| `scripts/install.bat` / `install.sh` | `scripts\install.bat` | 환경 설치(CUDA 자동 감지) |
| `scripts/run_center.bat` / `.sh` | `scripts\run_center.bat exp4c D:\data\liver A [run]` | 센터 원커맨드(카탈로그→단일 5-fold→업로드→FL→업로드) |
| `scripts/run_center_seq.bat` / `.sh` | `scripts\run_center_seq.bat D:\data\liver A exp4c exp5c` | 여러 실험 순차 |
| `scripts/patient_catalog.py` | `python scripts\patient_catalog.py --data D:\data\liver --exp exp4c --site A --run <run>` | 환자 메타·GT 부피 카탈로그 |
| `scripts/single.py` | `python scripts\single.py --config configs\exp4c.yaml --data D:\data\liver --site A --run <run> [--folds 0 1]` | 진짜 로컬 단일센터 5-fold |
| `scripts/client.py` | `python scripts\client.py --config configs\exp4c.yaml --data D:\data\liver --site A --run <run> [--folds 0] [--methods FedAvg] [--server IP:PORT]` | FL 클라이언트 |
| `scripts/export_results.py` | `python scripts\export_results.py --exp exp4c --site A --run <run>` | run 폴더 전체 익명 내보내기 |
| `scripts/upload_results.py` | `python scripts\upload_results.py --exp exp4c --site A --run <run> [--url http://168.131.153.57:9598/upload]` | 서버로 zip 전송 |
| `scripts/server.py` | `python scripts/server.py --config configs/exp4c.yaml configs/exp5c.yaml --run run_main [--folds 0] [--methods FedAvg] [--collect-port 9598]` | FL 서버(실험→fold→방법론 순차, 수집기 내장) |
| `scripts/collect_server.py` | `python scripts/collect_server.py --port 9598 --dir outputs/collected` | 결과 수집 서버 |
| `scripts/verify_run.py` | `python scripts/verify_run.py --config configs/rehearsal.yaml --sites A B --data-roots <A폴더> <B폴더> --run <run>` | 산출물 자동 검증 |
| `scripts/run_rehearsal.sh` | `bash scripts/run_rehearsal.sh run_final1` | 정식 모의시험(서버+모의센터 2+검증) |
| `scripts/run_seq_test.sh` | `bash scripts/run_seq_test.sh seqtest2` | 순차 실행 통합 테스트 |
| `scripts/run_crash_test.sh` | `bash scripts/run_crash_test.sh` | 클라이언트 강제 종료·재합류 테스트 |
| `scripts/pretrain.py` | `python scripts/pretrain.py --mode pretrain --epochs 200 --amp 0` / `--mode tian5fold --fold 0` / `--mode mr5fold --fold 0 --ts .../ts/730` | 공용 사전학습·5-fold 재현 |
| `scripts/eval_ts_zero_shot.py` | `python scripts/eval_ts_zero_shot.py --task liver_segments --cases <폴더> [--list split.json:key] [--mr]` | TS 570/576 zero-shot |
| `scripts/eval_gunetr_zero_shot.py` | `python scripts/eval_gunetr_zero_shot.py --cases <폴더> [--liver-mask gt\|none]` | G-UNETR++ zero-shot |
| `tools/gen_changes.py` | `python tools/gen_changes.py` | 센터 배포용 변경기록 생성 |

공통 옵션: `--workers N`(DataLoader 워커), `--limit N`(디버그용 케이스 수 제한, pretrain/eval), `--no-hd`(HD95 생략).
