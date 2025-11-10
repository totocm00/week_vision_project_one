# ==========================================================
# camera_loop 모듈
# 실시간 웹캠 화면을 띄워두고,
# 사용자가 [SPACE]를 누를 때마다 현재 프레임에 대해 OCR을 수행한 뒤
# 원본 이미지 / 시각화 이미지 / JSON 결과를 저장하는 데모입니다.
#
# 변경점
# - 카메라 화면에 실시간 선명도(Definition) 값을 표시합니다.
# - threshold보다 높으면 초록, 낮으면 빨간색으로 표시해서
#   "지금 초점 괜찮다/흐리다"를 촬영 전에 바로 알 수 있습니다.
# - camera_index가 "auto"여도 동작하도록 camera_initializer 를 사용합니다.
# ==========================================================

import os
import time
import cv2

from label_text_recognition.config.loader import load_ocr_config
from label_text_recognition.ocr.ocr_engine import build_ocr_engines
from label_text_recognition.ocr.ocr_runner import run_ocr_on_image
from label_text_recognition.exporters.json_exporter import export_to_json
from label_text_recognition.camera.camera_initializer import init_camera  # auto 처리 포함


def get_definition_score(frame):
    """
    프레임의 '선명도(blur 정도)'를 대략적으로 계산합니다.
    값이 클수록 선명한 이미지입니다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def start_camera_ocr() -> None:
    """
    실시간 카메라 OCR 데모 진입점.
    [SPACE] → OCR 실행
    [q]     → 종료
    """
    # ------------------------------------------------------
    # 1. 설정 불러오기
    # ------------------------------------------------------
    cfg = load_ocr_config()

    conf_threshold = cfg.get("conf_threshold", 0.5)
    definition_threshold = cfg.get("definition_threshold", 200)
    cls_enable = cfg.get("ocr_cls_enable", True)

    # 결과 저장 경로
    out_img_dir = cfg.get("output_dir_images", "assets/pictures")
    out_img_origin_dir = cfg.get("output_dir_images_origin", "assets/pictures-origin")
    out_json_dir = cfg.get("output_dir_json", "assets/json")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_img_origin_dir, exist_ok=True)
    os.makedirs(out_json_dir, exist_ok=True)

    # ------------------------------------------------------
    # 2. OCR 엔진 준비 (여러 언어 설정 가능)
    # ------------------------------------------------------
    ocr_langs = cfg.get("ocr_langs", ["en"])
    ocr_engines = build_ocr_engines(ocr_langs)
    main_engine = ocr_engines[ocr_langs[0]]  # 첫 번째 언어를 메인으로 사용

    # ------------------------------------------------------
    # 3. 카메라 열기 (auto 지원되는 초기화기 사용)
    #    → 여기서 이미 "감지된 카메라 인덱스: [...]" 가 출력됨
    # ------------------------------------------------------
    cap = init_camera(cfg)
    if cap is None:
        print("❌ 카메라를 열 수 없어 프로그램을 종료합니다.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    print("✅ Camera OCR ready. [SPACE] 캡처+OCR, [q] 종료")

    # ------------------------------------------------------
    # 4. 메인 루프
    # ------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임을 읽을 수 없습니다. 카메라 상태를 확인하세요.")
            break

        # 현재 프레임의 선명도 계산 (라이브 표시용)
        live_def = get_definition_score(frame)

        # 화면에 안내 문구 + 선명도 표시
        display = frame.copy()

        # 4-1) 안내 문구
        cv2.putText(
            display,
            "Press [SPACE] to OCR, [q] to quit",
            (10, 30),
            font,
            0.6,
            (255, 255, 255),
            2,
        )

        # 4-2) 선명도 표시 (threshold 넘으면 초록, 아니면 빨강)
        color = (0, 255, 0) if live_def >= definition_threshold else (0, 0, 255)
        cv2.putText(
            display,
            f"Definition: {live_def:.1f} (th={definition_threshold})",
            (10, 60),
            font,
            0.55,
            color,
            2,
        )

        cv2.imshow("Label Text Recognition - Camera", display)

        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord("q"):
            break

        # --------------------------------------------------
        # SPACE 눌렀을 때 OCR 수행
        # --------------------------------------------------
        if key == 32:  # 32 == spacebar
            ts = time.strftime("%Y%m%d_%H%M%S")
            print(f"\n📸 캡처 {ts} → OCR 중...")

            # 4-3) 이때는 방금 계산한 live_def 값을 그대로 사용
            def_score = live_def
            print(f"🔎 Definition score: {def_score:.2f}")

            # 4-4) OCR 실행
            results, vis_img, msg = run_ocr_on_image(
                frame.copy(),
                main_engine,
                conf_threshold,
                cls_enable,
            )

            # 4-5) 결과 저장
            img_path_origin = os.path.join(out_img_origin_dir, f"capture_{ts}.jpg")
            img_path = os.path.join(out_img_dir, f"capture_{ts}.jpg")
            json_path = os.path.join(out_json_dir, f"capture_{ts}.json")

            cv2.imwrite(img_path_origin, frame)   # 원본
            cv2.imwrite(img_path, vis_img)        # 박스 그린 이미지
            export_to_json(results, json_path)    # JSON

            print(
                "✅ 저장됨:\n"
                f"- 원본:   {img_path_origin}\n"
                f"- 이미지: {img_path}\n"
                f"- JSON:   {json_path}"
            )

            # 4-6) 결과 분석
            if msg.startswith("ERROR"):
                print(f"❌ OCR 처리 중 오류 발생: {msg}")
                continue

            if not results:
                if def_score < definition_threshold:
                    print(f"⚠️ OCR 결과가 비어 있습니다. (원인: 흐림 / Definition {def_score:.2f})")
                else:
                    print(f"⚠️ OCR 결과가 비어 있습니다. ({msg})")
                continue

            # 텍스트별로 출력
            for r in results:
                text = r.get("text", "")
                avg_conf = r.get("avg_conf", 0.0)
                print(f"- {text} ({avg_conf:.2f})")

            # 전체 평균 신뢰도
            confs = [r.get("avg_conf", 0.0) for r in results]
            overall_conf = sum(confs) / len(confs)
            print(f"📈 전체 평균 OCR 신뢰도: {overall_conf:.2f}")

            # 품질 판정
            if def_score < definition_threshold:
                print(f"⚠️ 이미지가 다소 흐립니다. (Definition {def_score:.2f} < {definition_threshold})")
            elif overall_conf < conf_threshold:
                print(f"⚠️ 인식은 되었으나 신뢰도가 낮습니다. (avg_conf: {overall_conf:.2f} < {conf_threshold})")
            else:
                print("✅ 선명도와 인식률 모두 양호합니다.")

    # ------------------------------------------------------
    # 5. 종료 처리
    # ------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("🟢 종료되었습니다.")