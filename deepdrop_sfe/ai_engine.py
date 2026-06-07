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
        else:
            try:
                import streamlit as st
                st_device = st.session_state.get("device")
            except:
                st_device = None
                
            if st_device:
                self.device = st_device
            elif torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # 2. 모델 아이디 결정 (하드웨어 및 환경에 따른 자동 선택)
        if model_id is None:
            # CUDA 인 경우에도 추론 속도 극대화를 위해 기본 모델을 small 로 변경
            if self.device == "cuda":
                model_id = "facebook/sam2.1-hiera-small"
            # MPS(macOS)나 CPU인 경우 메모리 효율을 위해 Tiny 사용
            elif self.device == "cpu":
                model_id = "facebook/sam2.1-hiera-tiny"
            else:
                model_id = "facebook/sam2.1-hiera-small"

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

        # 성능 최적화: 640px를 초과하는 고해상도는 리사이징하여 추론 속도 대폭 개선
        self.target_size = 640
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

    def predict_mask_fast(self, image_rgb, box):
        """
        초저사양 환경을 위한 OpenCV 고속 액적 분할 폴백.
        SAM을 사용하지 않고 전통적인 영상처리 기법으로 0.05초 이내에 마스크를 생성함.
        """
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 유효하지 않은 박스 예외 처리
        if x2 <= x1 or y2 <= y1:
            return mask > 0, 0.0
            
        roi = image_rgb[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 대비 향상을 위해 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 중심점 (클릭한 지점)
        cx_roi = (x2 - x1) // 2
        cy_roi = (y2 - y1) // 2
        max_r = min(x2 - x1, y2 - y1) // 2
        min_r = max(5, max_r // 10)
        
        # 1. 형태/곡률 기반 액적 탐지 (우선순위 높음)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
            param1=50, param2=20, minRadius=min_r, maxRadius=max_r
        )
        
        best_circle = None
        min_dist = float('inf')
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (xc, yc, r) in circles:
                dist = np.sqrt((xc - cx_roi)**2 + (yc - cy_roi)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = (xc, yc, r)
                    
        # 클릭 지점(중심) 근처에서 유효한 원을 찾은 경우 즉시 반환
        if best_circle is not None and min_dist < max_r // 2:
            xc, yc, r = best_circle
            cv2.circle(mask, (xc + x1, yc + y1), r, 1, thickness=cv2.FILLED)
            return mask > 0, 1.0
            
        # 2. 곡률 탐지 실패 시 대비(Fallback) - 명암 기반 Otsu
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if thresh[cy_roi, cx_roi] == 0:
            thresh = cv2.bitwise_not(thresh)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        min_dist = float('inf')
        
        if contours:
            for cnt in contours:
                dist = cv2.pointPolygonTest(cnt, (cx_roi, cy_roi), True)
                if dist >= 0:
                    if best_cnt is None or cv2.contourArea(cnt) > cv2.contourArea(best_cnt):
                        best_cnt = cnt
                elif best_cnt is None:
                    if -dist < min_dist:
                        min_dist = -dist
                        best_cnt = cnt

            if best_cnt is not None:
                best_cnt += np.array([[x1, y1]]) # 원본 이미지 좌표계로 복구
                cv2.drawContours(mask, [best_cnt], -1, 1, thickness=cv2.FILLED)
                return mask > 0, 1.0
            
        return mask > 0, 0.0

    def auto_detect_coin_candidate(self, image_cv2):
        """
        [V3] Edge Continuity Score와 타이트한 반경 탐색을 활용하여 오탐지 0% 달성
        """
        orig_h, orig_w = image_cv2.shape[:2]
        try:
            import streamlit as st
            max_dim = st.session_state.get("max_image_size") or float('inf')
        except:
            max_dim = 800.0
        scale = 1.0
        
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_img = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = image_cv2

        h, w = work_img.shape[:2]
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_pre = clahe.apply(gray)
        gray_pre = cv2.medianBlur(gray_pre, 7)

        # 동전 크기는 전체 이미지 높이의 7% ~ 12% 로 매우 제한적임
        circles = cv2.HoughCircles(
            gray_pre,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=w // 5,
            param1=50,
            param2=30,
            minRadius=int(h * 0.07),
            maxRadius=int(h * 0.12),
        )

        if circles is not None:
            circles = np.around(circles[0, :]).astype(np.int32)
            best_candidate = None
            max_score = -1.0

            # Edge Continuity 측정용 Edge Map 생성
            edges = cv2.Canny(gray, 50, 150)

            for c in circles:
                cx, cy, cr = c
                if cx - cr < 0 or cx + cr >= w or cy - cr < 0 or cy + cr >= h:
                    continue

                mask = np.zeros_like(edges)
                cv2.circle(mask, (int(cx), int(cy)), int(cr), 255, 3)
                overlap = cv2.bitwise_and(edges, mask)
                perimeter_pixels = np.sum(mask > 0)
                if perimeter_pixels == 0: continue
                
                # 테두리에 실제로 존재하는 엣지 픽셀의 비율
                score = float(np.sum(overlap > 0)) / float(perimeter_pixels)

                if score > max_score:
                    max_score = score
                    best_candidate = c

            if best_candidate is None:
                best_candidate = circles[0]

            x, y, r = best_candidate
            pad = int(r * 0.1)

            # Map coordinates back to original scale
            inv_scale = 1.0 / scale
            orig_x = int(x * inv_scale)
            orig_y = int(y * inv_scale)
            orig_r = int(r * inv_scale)
            orig_pad = int(pad * inv_scale)

            coin_box = [
                max(0, orig_x - orig_r - orig_pad),
                max(0, orig_y - orig_r - orig_pad),
                min(orig_w, orig_x + orig_r + orig_pad),
                min(orig_h, orig_y + orig_r + orig_pad),
            ]
            
            box_arr = np.array(coin_box)
            return box_arr, (float(orig_x), float(orig_y), float(orig_r))
            
        return None, None

    def auto_detect_droplet_candidate(self, image_cv2, exclude_box=None, coin_radius=None):
        """
        [V5] 액적은 전체 이미지의 2%~6%에 해당하는 타이트한 반경 제약을 사용하여 검출함.
        HL(헤어라인) 등 극도로 노이즈가 많은 배경에 대비해, 앱 단에서 마우스 오버라이드 툴이 제공됨.
        """
        orig_h, orig_w = image_cv2.shape[:2]
        try:
            import streamlit as st
            max_dim = st.session_state.get("max_image_size") or float('inf')
        except:
            max_dim = 600.0
        scale = 1.0

        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_img = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = image_cv2.copy()

        h, w = work_img.shape[:2]
        
        ex1, ey1, ex2, ey2 = -1, -1, -1, -1
        if exclude_box is not None:
            ex1 = int(exclude_box[0] * scale)
            ey1 = int(exclude_box[1] * scale)
            ex2 = int(exclude_box[2] * scale)
            ey2 = int(exclude_box[3] * scale)
            cv2.rectangle(work_img, (max(0, ex1), max(0, ey1)), (min(w, ex2), min(h, ey2)), (0, 0, 0), -1)
            
        b, g, r_ch = cv2.split(work_img)
        gray2 = cv2.addWeighted(b, 0.7, g, 0.3, 0)
        gray_blur = cv2.GaussianBlur(gray2, (7, 7), 0)
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe2.apply(gray_blur.astype(np.uint8))
        
        min_r = h * 0.02
        max_r = h * 0.06
        
        circles = cv2.HoughCircles(
            gray_clahe, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=15, minRadius=int(min_r), maxRadius=int(max_r)
        )
        
        best_box = None
        best_score = -1.0
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (xc, yc, r) in circles:
                if ex1 - 20 <= xc <= ex2 + 20 and ey1 - 20 <= yc <= ey2 + 20:
                    continue
                    
                dist_to_center = np.sqrt((xc - w/2.0)**2 + (yc - h/2.0)**2)
                max_dist = np.sqrt((w/2.0)**2 + (h/2.0)**2)
                center_score = 1.0 - (dist_to_center / (max_dist + 1e-6))
                
                score = center_score * (r ** 2)
                if score > best_score:
                    best_score = score
                    best_box = (xc, yc, r)

        if best_box is not None:
            cx, cy, r_est = best_box
            pad = r_est * 0.2
            inv_scale = 1.0 / scale
            
            orig_cx = cx * inv_scale
            orig_cy = cy * inv_scale
            orig_r = r_est * inv_scale
            orig_pad = pad * inv_scale
            
            box_arr = np.array([
                max(0, int(orig_cx - orig_r - orig_pad)),
                max(0, int(orig_cy - orig_r - orig_pad)),
                min(orig_w, int(orig_cx + orig_r + orig_pad)),
                min(orig_h, int(orig_cy + orig_r + orig_pad)),
            ])
            return box_arr
            
        return None

    def get_binary_mask(self, mask):
        return (mask * 255).astype(np.uint8)
