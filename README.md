# Label & Text Recognition

이 프로젝트는 **open_vision_factory** 안에서
동작하도록 설계된 “라벨/문자 인식(OCR) 모듈”입니다.  
카메라로 이미지를 캡처하거나, 이미지 파일을 입력으로 받아  
**PaddleOCR**로 텍스트를 인식하고 결과를 **JSON / 이미지**로 저장합니다.  
구조는  `demos/ → src/ → assets/` 흐름으로 구성되어 있습니다.

---

## 1. 사전 조건 (Prerequisites)

- Python 3.10 이상 권장
- 먼저 **open_vision_factory**를 클론해서 기본 환경을 만들어야 합니다.
- OCR 의존성은 모두 `open_vision_factory/requirements_ocr.txt` 에 정의되어 있습니다.

---

## 2. 설치 (Installation)

### 2-1. open_vision_factory 클론
```bash
git clone https://github.com/totocm00/open_vision_factory.git
cd open_vision_factory
```

### 2-2. 가상환경 생성 및 활성화
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2-3. OCR 전용 의존성 설치
`open_vision_factory` 루트에 있는 **requirements_ocr.txt** 를 먼저 설치합니다.

```bash
pip install -U pip
pip install -r requirements_ocr.txt
```

> 이 파일 안에 `paddleocr`, `opencv-python`, `numpy`, `pyyaml` 등 OCR 모듈이 동작하는 데 필요한 패키지가 들어 있습니다.

### 2-4. label_text_recognition 모듈 클론
이제 같은 루트(open_vision_factory) 안에 이 레포를 넣습니다.

```bash
cd open_vision_factory
git clone https://github.com/totocm00/label_text_recognition.git
cd label_text_recognition
```

최종 구조 예시는 다음과 같습니다.

```text
open_vision_factory/
├── requirements_ocr.txt
├── requirements_run.txt
├── requirements_dev.txt
└── label_text_recognition/
    ├── demos/
    ├── src/
    └── assets/
```

### 2-5. 설치 확인
```bash
python -c "import paddleocr, cv2, yaml; print('✅ OCR environment OK')"
```

---

## 3. 프로젝트 구조

```text
label_text_recognition/
├── README.md
├── demos/                      # 실행 가능한 예제 스크립트
│   ├── camera_ocr_demo.py      # 카메라로 OCR 테스트
│   └── image_ocr_demo.py       # 이미지 파일로 OCR 테스트
├── src/                        # 실제 파이썬 코드
│   └── label_text_recognition/
│       ├── __init__.py
│       ├── camera/
│       │   └── camera_loop.py  # 카메라 캡처 + OCR 루프
│       ├── ocr/
│       │   ├── ocr_engine.py   # PaddleOCR 여러 언어 로더
│       │   ├── ocr_runner.py   # 이미지 1장 OCR → 결과 반환
│       │   └── ocr_utils.py    # 박스 병합, 시각화 유틸
│       ├── config/
│       │   ├── ocr_config.yaml # 매직넘버 대신 여기서 관리
│       │   └── loader.py       # YAML 로더
│       └── exporters/
│           └── json_exporter.py# 결과 JSON 저장
├── assets/                     # 실행 결과/샘플
│   ├── pictures/               # 캡처 이미지 저장
│   └── json/                   # OCR 결과 JSON 저장
```

---

## 4. 빠른 실행 (Quickstart)

### 4-1. 카메라로 OCR
```bash
cd label_text_recognition
python demos/camera_ocr_demo.py
```
- 창이 열리면 **SPACE** 를 누를 때마다 현재 프레임을 캡처해서 OCR을 수행합니다.
- 결과는 아래 경로에 저장됩니다.
  - 이미지: `assets/pictures/`
  - JSON: `assets/json/`
- **q** 를 누르면 종료됩니다.

### 4-2. 이미지 파일로 OCR
```bash
python demos/image_ocr_demo.py --image assets/pictures/sample.jpg
```
- 지정한 이미지 한 장에 대해 OCR을 수행하고
- 결과 JSON을 `assets/json/` 에 저장합니다.

---

## 5. 설정 (Config)

자주 바꿔야 하는 값은 **코드 안에 매직 넘버로 두지 않고** 모두 YAML에서 관리합니다.  
설정 파일 위치:  
`src/label_text_recognition/config/ocr_config.yaml`

```yaml
# src/label_text_recognition/config/ocr_config.yaml

camera_index: 0          # 사용할 카메라 번호
frame_width: 960         # 캡처 해상도 가로
frame_height: 540        # 캡처 해상도 세로

ocr_langs:               # PaddleOCR에서 동시에 로드할 언어
  - en
  - korean

output_dir_images: "assets/pictures"  # 이미지 저장 위치
output_dir_json: "assets/json"        # OCR 결과 저장 위치

conf_threshold: 0.5      # 이 값보다 낮으면 OCR 결과에서 제외
```

이 파일만 수정하면,
- 카메라 번호가 바뀌어도
- 해상도를 바꿔도
- 저장 폴더를 옮겨도  
코드를 고칠 필요 없이 그대로 반영됩니다.

---

## 6. 주요 모듈 설명

| 경로 | 설명 |
|------|------|
| `demos/` | “이렇게 실행하세요”를 보여주는 예제 스크립트 모음 |
| `src/label_text_recognition/camera/camera_loop.py` | 웹캠을 열고 SPACE 키로 캡처 → OCR → 저장까지 하는 메인 루프 |
| `src/label_text_recognition/ocr/ocr_engine.py` | YAML에 적힌 언어 목록으로 PaddleOCR 엔진을 여러 개 만드는 곳 |
| `src/label_text_recognition/ocr/ocr_runner.py` | 이미지 1장을 받아서 OCR→후처리를 한 번에 실행하는 진입점 |
| `src/label_text_recognition/ocr/ocr_utils.py` | OCR 결과를 한 줄로 합치고 이미지에 박스를 그려주는 유틸 |
| `src/label_text_recognition/config/loader.py` | `ocr_config.yaml`을 읽어서 dict로 넘겨주는 설정 로더 |
| `src/label_text_recognition/exporters/json_exporter.py` | OCR 결과(list[dict])를 JSON 파일로 저장하는 Exporter |
| `assets/` | 실행 중 생성되는 산출물이 떨어지는 곳 (git에 안 올려도 되는 폴더) |

---

## 7. 왜 `demos/ → src/ → assets/` 인가?

- **demos/** : 사용자가 바로 실행해서 볼 수 있는 자리. “이 모듈이 이런 식으로 동작한다”는 걸 한눈에 보여줍니다.  
- **src/** : 실제 로직이 들어 있는 파이썬 패키지. 나중에 다른 프로젝트에서 `import label_text_recognition...` 형태로 재사용할 때 이 안만 보면 됩니다.  
- **assets/** : 결과물과 샘플을 모아두는 곳. 실행할수록 쌓이니까 코드와 분리했습니다.


**이미지/비전 쪽에 익숙한 사람들도 금방 이해**할 수 있도록 제작했습니다.

---

## 8. 향후 확장 계획

- YOLO 기반 라벨/영역 감지 모듈 추가 (`src/label_text_recognition/detectors/` 예정)
- Streamlit UI 데모 추가 (`demos/streamlit_demo.py`)
- Exporter 확장 (CSV / 이미지 오버레이 / REST 응답 포맷)
- open_vision_factory 실행 스크립트에서 바로 이 모듈을 불러쓸 수 있도록 통합

---

## 9. 요약

1. **open_vision_factory** 를 먼저 클론한다.  
2. 거기 있는 **requirements_ocr.txt** 를 설치한다.  
3. 그 안에 **label_text_recognition** 을 클론한다.  
4. `python demos/camera_ocr_demo.py` 만 실행하면 OCR이 돌아간다.  
5. 카메라 번호·언어·출력 폴더는 전부 `ocr_config.yaml` 에서 바꾼다.

이 순서만 지키면 다른 환경에서도 바로 재현이 가능합니다. ✅
