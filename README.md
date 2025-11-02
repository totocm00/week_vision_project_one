#  Week 1 - Label & Text Recognition Project

> **Vision Project : Step 1**  
> **기간:** 2025.10.31 ~ 진행 중  
> **목표:** 제품 라벨의 부착 여부와 문자 인쇄 여부를 비전 기반으로 자동 판별

> [Notion](https://www.notion.so/Home-29d9eb4f232780c6be2acbe1b4432c7e?source=copy_link)

---

## 🖼 프로젝트 대표 이미지

<img width="100" height="100" alt="Image" src="https://github.com/user-attachments/assets/fdf88d95-ff22-4de8-9fee-49964e50dbc5" />

> *(예시 이미지 — 라벨 감지 및 문자 인식 시각화 결과 / 완성시 교체!!!)*  
> YOLO로 라벨 검출 → EasyOCR로 텍스트 유무 판별

---

## 🏁 진행 단계

1. 프로젝트 아이디어 정하기 / 주제와 방향성 선정  
2. 필요한 스킬셋 및 자료 탐색  
3. 프로젝트 구조 및 README 작성  
4. 시퀀스 다이어그램 / 클래스 다이어그램 작성  
5. 코드 작성  
6. 이슈 및 PR 정리  
7. 확장성 및 개선사항 검토  

---

## 💡 프로젝트 아이디어

### 주제

` 라벨 부착 및 문자 인쇄 여부 확인 `   

### 방향성

> 본 프로젝트는 원래 **PLC 팀 프로젝트(PCB 납땜 검사)** 중  
> 비전 파트를 담당하고자 시작된 개인 성장 프로젝트입니다.

팀 프로젝트의 진행 구조가 명확히 잡히지 않은 상태에서  
직접 비전 분야 역량을 키우기 위해 별도로 시작했습니다.  
짧은 기간 안에 **1주 단위 MVP 완성**을 목표로 하며,  
빠른 시도와 결과물을 통해 실질적인 성장과 실무 감각을 얻고자 합니다.

---

## 🛠 필요한 기술 스택

- **Detection** : 라벨 부착 영역 검출 (YOLO 기반 OK/NG 분류)  
- **OCR** : 문자 인쇄 유무 판별 (EasyOCR / PaddleOCR)  
- **데이터 전처리** : 크롭, 색상 보정, 명암 조정  
- **학습 환경** : GPU 기반 (유니온 모두AI(겟티) / Kaggle)  
- **UI / 배포** : Streamlit 대시보드 구성  
- **라벨링 도구** : Label Studio, Roboflow  

---

## 🧩 데이터셋 구성

### 1️⃣ 라벨 부착 감지용 (YOLO 학습)

- **목표:** 라벨이 붙은 제품 vs 라벨이 없는 제품  
- **클래스:** `0 label_present`, `1 label_absent`  
- **데이터셋 예시:**  
  - AI-Hub: 산업용 제품 이미지, 공정불량 탐지  
  - Roboflow: label/sticker/bottle detection  
  - Kaggle: Product label / Bottle quality dataset  

---

### 2️⃣ 문자 인쇄 여부 판별용 (OCR)

- **목표:** 라벨 내 문자가 인쇄되어 있는지 여부  
- **구조:**  
  1. YOLOv8 → 라벨 영역 검출  
  2. OpenCV → ROI 추출  
  3. EasyOCR → 문자 유무 판별  
  4. Streamlit → 결과 시각화 (OK/NG 리포트)  
- **데이터 출처:**  
  - AI-Hub OCR 데이터셋  
  - Kaggle Printed Text / Blank Surface 등  

---

## ⚙️ 주의사항

1. 다양한 각도·조명·손상 데이터를 포함  
2. YOLO 구조
3. EasyOCR 결과가 빈 문자열이면 "문자 없음(NG)"  
4. 낮은 confidence 값은 보류 처리  

---

## 🔁 유니온모두AI 대체 플랫폼

| 목적 | 플랫폼 | 비고 |
|------|----------|------|
| **Detection(라벨)** | Ultralytics HUB / Roboflow / Kaggle | 노코드·무료 |
| **OCR(문자)** | PaddleOCR / EasyOCR / docTR | Python 기반 |
| **데이터셋** | AI-Hub / Roboflow / Kaggle | 공개 데이터 |
| **라벨링 도구** | Label Studio / Supervisely | 무료 & 오픈소스 |

**추천 파이프라인**  
> Roboflow(라벨링) → Kaggle(학습) → YOLOv8 추론 → EasyOCR 결과 판별 → Streamlit 리포트  

---

## 🧠 프로젝트 목표

- **Task 1:** 라벨 부착 감지 (YOLOv8)  
- **Task 2:** 문자 인쇄 유무 판별 (EasyOCR / PaddleOCR)  
- **Task 3:** 결과 리포트 및 UI 구성 (Streamlit)  

---

## 👥 GitHub 참여자

```bash
Account 1 : HomeToTo
역할   : 집에서 구조 설계 및 보고서 담당
Plus   : UI·문서화 및 전략 정리

Account 2 : CodeToTo
역할   : 학원에서 코드 중심 개발 담당
Plus   : 실험, 필기, 코드 개선 담당

