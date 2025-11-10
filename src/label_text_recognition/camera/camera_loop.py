# ==========================================================
# 웹캠을 열고, 사용자가 SPACE를 누를 때마다 현재 프레임을 캡처해서
# OCR을 수행한 뒤 이미지/JSON으로 저장하는 메인 루프입니다.
# 설정은 전부 YAML에서 불러오므로 코드 안에 매직 넘버를 거의 쓰지 않습니다.
# ==========================================================

import os
import time
import cv2

from label_text_recognition.config.loader import load_ocr_config
from label_text_recognition.ocr.ocr_engine import build_ocr_engines
from label_text_recognition.ocr.ocr_runner import run_ocr_on_image
from label_text_recognition.exporters.json_exporter import export_to_json

def get_definition_score(frame):
    """
    입력 프레임의 선명도를 대략적으로 계산합니다.
    값이 클수록 선명한 이미지입니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def start_camera_ocr() -> None:
    """
    카메라를 열어서 실시간 OCR 데모를 실행합니다.
    SPACE 입력 시 OCR 수행, q 입력 시 종료합니다.
    """
    cfg = load_ocr_config()

    camera_id = cfg.get("camera_index", 0)
    frame_w = cfg.get("frame_width", 960)
    frame_h = cfg.get("frame_height", 540)
    conf_threshold = cfg.get("conf_threshold", 0.5)

    out_img_dir = cfg.get("output_dir_images", "assets/pictures")
    out_json_dir = cfg.get("output_dir_json", "assets/json")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)

    ocr_langs = cfg.get("ocr_langs", ["en"])
    ocr_engines = build_ocr_engines(ocr_langs)
    main_lang = ocr_langs[0]
    main_engine = ocr_engines[main_lang]

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    if not cap.isOpened():
        print(f"❌ 카메라 {camera_id} 를 열 수 없습니다.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    print("✅ Camera OCR ready. [SPACE] 캡처+OCR, [q] 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임을 읽을 수 없습니다.")
            break

        display = frame.copy()
        cv2.putText(
            display,
            "Press [SPACE] to OCR, [q] to quit",
            (10, 30),
            font,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Label Text Recognition - Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == 32:  # SPACE
            ts = time.strftime("%Y%m%d_%H%M%S")
            print(f"\n📸 캡처 {ts} → OCR 중...")

            # 1) 선명도 점수 계산
            def_score = get_definition_score(frame)
            print(f"🔎 Definition score: {def_score:.2f}")

            # 2) OCR 실행
            results, vis_img = run_ocr_on_image(frame.copy(), main_engine, conf_threshold)

            # 저장 경로
            img_path = os.path.join(out_img_dir, f"capture_{ts}.jpg")
            json_path = os.path.join(out_json_dir, f"capture_{ts}.json")

            cv2.imwrite(img_path, vis_img)
            export_to_json(results, json_path)

            print(f"✅ 저장됨:\n- 이미지: {img_path}\n- JSON:   {json_path}")

            # 3) OCR 결과 로그
            # results가 리스트 형태라고 가정
            if results:
                # 라인별 출력
                for r in results:
                    text = r.get("text", "")
                    avg_conf = r.get("avg_conf", 0.0)
                    print(f"- {text} ({avg_conf:.2f})")

                # 전체 평균 신뢰도도 한 번 찍어주자
                confs = [r.get("avg_conf", 0.0) for r in results]
                overall_conf = sum(confs) / len(confs)
                print(f"📈 전체 평균 OCR 신뢰도: {overall_conf:.2f}")

                # 선명도와 신뢰도를 같이 판단
                if def_score < 200:
                    print("⚠️ 이미지가 다소 흐립니다. 조명/초점 확인하세요.")
                elif overall_conf < conf_threshold:
                    print("⚠️ 인식은 되었으나 신뢰도가 낮습니다. 각도/거리 조정 필요.")
                else:
                    print("✅ 선명도와 인식률 모두 양호합니다.")
            else:
                print("⚠️ OCR 결과가 비어 있습니다.")

    cap.release()
    cv2.destroyAllWindows()
    print("🟢 종료되었습니다.")
