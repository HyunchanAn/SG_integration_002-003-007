# ruff: noqa
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor


class AIContactAngleAnalyzer:
    """
    SAM 2.1 (Segment Anything Model 2.1) 기반의 고정밀 액적 및 참조 물체 분석기.
    RTX 5080 등 하이엔드 GPU 및 macOS MPS 가속을 지원합니다.
    Streamlit Cloud 등 메모리 제한 환경을 위해 Tiny 모델 자동 전환 로직이 포함되어 있습니다.
    """

    def __init__(self, model_id=None, device=None):
        # 1. 디바이스 자동 감지
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # 2. 모델 아이디 결정 (하드웨어 및 환경에 따른 자동 선택)
        if model_id is None:
            # CUDA 인 경우 고사양으로 간주하여 Large 사용 (RTX 5080 대응)
            if self.device == "cuda":
                model_id = "facebook/sam2.1-hiera-large"
            # MPS(macOS)나 CPU인 경우 메모리 효율을 위해 Tiny 혹은 Base 사용 검토
            # 사용자의 요청에 따라 저사양/클라우드 환경 대응을 위해 기본적으로 Tiny 모델 로직 도입
            elif self.device == "cpu":
                model_id = "facebook/sam2.1-hiera-tiny"
            else:
                # MPS 등 기타 가속 환경에서는 기본적으로 Large 사용 시도
                model_id = "facebook/sam2.1-hiera-large"

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(
                f"SAM 2.1 모델 ({model_id})을 GPU에서 로드 중: {gpu_name} ({vram_total:.1f}GB VRAM)..."
            )
            torch.backends.cudnn.benchmark = True
        else:
            print(f"SAM 2.1 모델 ({model_id})을 {self.device}에서 로드 중...")

        # 3. 모델 빌드 (Hugging Face 자동 다운로드 활용)
        try:
            self.model = build_sam2_hf(model_id, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            print(f"SAM 2.1 ({model_id}) 로드 완료.")
        except Exception as e:
            print(f"SAM 2.1 모델 로드 실패: {e}")
            if "large" in model_id:
                print("저사양 모델(tiny)로 재시도합니다...")
                try:
                    self.model = build_sam2_hf("facebook/sam2.1-hiera-tiny", device=self.device)
                    self.predictor = SAM2ImagePredictor(self.model)
                    print("Tiny 모델로 정상 복구되었습니다.")
                except Exception as e2:
                    raise RuntimeError(f"모델 복구 시도 실패: {e2}")
            else:
                raise e

    def set_image(self, image_rgb):
        """
        SAM2 예측기를 위해 이미지를 설정함. 성능을 위해 내부적으로 1024px로 최적화 리사이징을 수행할 수 있음.
        """
        h_orig, w_orig = image_rgb.shape[:2]
        self.orig_size = (h_orig, w_orig)

        # 성능 최적화: 1024px를 초과하는 고해상도는 리사이징하여 추론 속도 개선
        self.target_size = 1024
        if max(h_orig, w_orig) > self.target_size:
            scale = self.target_size / float(max(h_orig, w_orig))
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            self.image_proc = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.scale = scale
        else:
            self.image_proc = image_rgb
            self.scale = 1.0

        self.predictor.set_image(self.image_proc)

    def predict_mask(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        """
        프롬프트(점, 박스)를 기반으로 마스크를 생성함.
        """
        # 스케일 보정
        p_coords = None
        if point_coords is not None:
            p_coords = np.array(point_coords) * self.scale

        p_box = None
        if box is not None:
            p_box = np.array(box) * self.scale

        masks, scores, logits = self.predictor.predict(
            point_coords=p_coords,
            point_labels=point_labels,
            box=p_box,
            multimask_output=multimask_output,
        )

        # 가장 점수가 높은 마스크 선택
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        # 원본 해상도로 복구
        if self.scale != 1.0:
            best_mask = cv2.resize(
                best_mask.astype(np.uint8),
                (self.orig_size[1], self.orig_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            return best_mask > 0, scores[best_idx]

        return best_mask, scores[best_idx]

    def auto_detect_coin_candidate(self, image_cv2):
        """
        [V2] 원주 우선(Rim-first) 전략을 사용하여 동전을 감지하고 하이브리드 프롬프트를 생성함.
        """
        h, w = image_cv2.shape[:2]
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        candidates = []
        img_area = h * w

        def process_contours(binary_img, method_name):
            cnts, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                area_ratio = area / img_area
                if area_ratio < 0.0005 or area_ratio > 0.40:
                    continue

                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue

                solidity = float(area) / hull_area
                peri = cv2.arcLength(hull, True)
                circularity = 4 * np.pi * hull_area / (peri * peri) if peri > 0 else 0

                if len(hull) >= 5:
                    ellipse = cv2.fitEllipse(hull)
                    (xc, yc), (d1, d2), angle = ellipse
                    ar = max(d1, d2) / (min(d1, d2) + 1e-6)
                else:
                    ar = 100

                if solidity < 0.70 or ar > 4.0 or circularity < 0.4:
                    continue

                # 중앙부 우선 및 크기 기중치 패널티 보완
                dist_from_center = np.sqrt((xc - w / 2) ** 2 + (yc - h / 2) ** 2)
                max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
                center_score = 1.0 - (dist_from_center / (max_dist + 1e-6))
                
                # 면적 비율이 너무 작으면 패널티 부여 (동전은 미세한 먼지가 아님)
                size_penalty = 1.0 if area_ratio >= 0.005 else (area_ratio / 0.005)

                score = (circularity * 0.25) + (solidity * 0.25) + (center_score * 0.3) + (min(1.0, area_ratio / 0.05) * 0.2)
                score *= size_penalty

                x, y, bw, bh = cv2.boundingRect(hull)
                candidates.append(
                    {
                        "box": [
                            max(0, x - 5),
                            max(0, y - 5),
                            min(w, x + bw + 5),
                            min(h, y + bh + 5),
                        ],
                        "score": score,
                        "center": [int(xc), int(yc)],
                    }
                )

        # 다중 이진화 전략
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )
        process_contours(thresh_adapt, "Adaptive")
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        process_contours(thresh_otsu, "Otsu")

        if not candidates:
            return None, None

        best = max(candidates, key=lambda x: x["score"])
        bx1, by1, bx2, by2 = best["box"]

        # 하이브리드 프롬프트: 중앙 포인트 + 박스
        points = np.array([[best["center"][0], best["center"][1]]])
        labels = np.array([1])
        box = np.array(best["box"])

        return box, (float(best["center"][0]), float(best["center"][1]), float((bx2 - bx1) / 2))

    def auto_detect_droplet_candidate(self, image_cv2, exclude_box=None):
        """
        [V2] 금속 표면 및 투명 액적 최적화 탐지 로직 (하이브리드 프롬프트 대응).
        """
        h, w = image_cv2.shape[:2]
        b, g, r_ch = cv2.split(image_cv2)
        gray = cv2.addWeighted(b, 0.6, g, 0.4, 0)  # 파란색 계열 강조 (난반사 억제)
        smoothed = cv2.medianBlur(gray, 7)
        blurred = cv2.GaussianBlur(smoothed, (9, 9), 0)

        edged = cv2.Canny(blurred, 20, 80)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5
        )
        combined = cv2.bitwise_or(edged, thresh)

        if exclude_box is not None:
            ex1, ey1, ex2, ey2 = map(int, exclude_box)
            combined[max(0, ey1 - 10) : min(h, ey2 + 10), max(0, ex1 - 10) : min(w, ex2 + 10)] = 0

        cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = []
        img_area = h * w

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 200 or area > img_area * 0.10:
                continue
            peri = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

            if circularity > 0.3:
                x, y, bw, bh = cv2.boundingRect(cnt)
                valid_candidates.append(
                    {
                        "center": [int(x + bw / 2), int(y + bh / 2)],
                        "box": [
                            max(0, x - 5),
                            max(0, y - 5),
                            min(w, x + bw + 5),
                            min(h, y + bh + 5),
                        ],
                        "score": circularity * (area / img_area),
                    }
                )

        if not valid_candidates:
            return None

        best = max(valid_candidates, key=lambda x: x["score"])
        return np.array(best["box"])

    def get_binary_mask(self, mask):
        return (mask * 255).astype(np.uint8)
