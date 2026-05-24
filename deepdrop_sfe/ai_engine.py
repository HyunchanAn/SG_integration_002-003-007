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
        [V2] V-SAMS에서 검증된 강건한 허프 변환(HoughCircles) 및 CLAHE 알고리즘을 사용하여 동전을 감지함.
        """
        orig_h, orig_w = image_cv2.shape[:2]
        max_dim = 800.0
        scale = 1.0
        
        # Optimization: Downsample if image is too large
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_img = cv2.resize(image_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = image_cv2

        h, w = work_img.shape[:2]
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)

        # Preprocessing: Maximize contrast using CLAHE and median blur
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_pre = clahe.apply(gray)
        gray_pre = cv2.medianBlur(gray_pre, 7)

        circles = cv2.HoughCircles(
            gray_pre,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=w // 5,
            param1=50,
            param2=30,
            minRadius=int(h * 0.05),
            maxRadius=int(h * 0.25),
        )

        if circles is not None:
            circles = np.around(circles[0, :]).astype(np.int32)
            best_candidate = None
            max_score = -1.0

            for c in circles:
                cx, cy, cr = c
                if cx - cr < 0 or cx + cr >= w or cy - cr < 0 or cy + cr >= h:
                    continue

                # 1. Positional Score (Closer to center is higher)
                dist_to_center = np.sqrt((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
                pos_score = 1.0 - (dist_to_center / (np.sqrt((w / 2) ** 2 + (h / 2) ** 2)))

                # 2. Texture Complexity Score (Coin internal detail)
                roi = gray[cy - cr : cy + cr, cx - cr : cx + cr]
                texture_score = float(np.std(roi)) / 128.0

                score = pos_score * 0.4 + texture_score * 0.6

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
        [V4.1] 2B 금속 표면의 렌즈 효과(Lens Effect) 병합 알고리즘 파라미터 튜닝.
        액적 내부의 조각난 렌즈 왜곡(스크래치 파편)들을 하나의 거대한 Blob으로 완전히 뭉치도록
        형태학적 결합(Closing) 커널을 대폭 확대하고, 최소 반경 커트라인을 상향 조정.
        """
        orig_h, orig_w = image_cv2.shape[:2]
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
        
        # 기준 반지름 계산 (coin_radius가 있으면 적극 활용, 없으면 화면 10%)
        ref_radius = (coin_radius * scale) if (coin_radius is not None and coin_radius > 0) else (h * 0.1)
        
        # 1. 제외 영역(동전)을 완전히 까맣게 마스킹
        ex1, ey1, ex2, ey2 = -1, -1, -1, -1
        if exclude_box is not None:
            ex1 = int(exclude_box[0] * scale)
            ey1 = int(exclude_box[1] * scale)
            ex2 = int(exclude_box[2] * scale)
            ey2 = int(exclude_box[3] * scale)
            cv2.rectangle(work_img, (max(0, ex1), max(0, ey1)), (min(w, ex2), min(h, ey2)), (0, 0, 0), -1)
            
        # 2. 로컬 분산(Variance) 맵 기반 렌즈 효과 포착
        b, g, r_ch = cv2.split(work_img)
        gray = cv2.addWeighted(b, 0.7, g, 0.3, 0).astype(np.float32)
        
        # 블러 크기를 동전 반지름의 15% 수준으로 넉넉히 주어 파편화를 1차 방지
        win_size = int(ref_radius * 0.15) | 1
        if win_size < 3: win_size = 3
        
        mean_gray = cv2.blur(gray, (win_size, win_size))
        mean_gray_sq = cv2.blur(gray**2.0, (win_size, win_size))
        variance = mean_gray_sq - mean_gray**2.0
        
        stddev = np.sqrt(np.maximum(variance, 0))
        stddev_norm = cv2.normalize(stddev, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 3. 임계값을 통한 강한 렌즈 효과(대비 뭉침) 추출
        _, thresh = cv2.threshold(stddev_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. 강력한 형태학적 닫힘(Morphological Closing) 연산으로 파편들 하나로 뭉치기
        # 커널 크기를 동전 반지름의 30% 수준으로 대폭 키움 (파편들이 완전히 떡지도록)
        kernel_size = int(ref_radius * 0.30) | 1
        if kernel_size < 5: kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3) # 반복 횟수도 증가
        
        # 너무 자잘한 노이즈만 날리는 가벼운 열림 연산
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, open_kernel, iterations=1)
        
        # 5. 윤곽선(Blob) 검출
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 액적 최소 반경 커트라인 대폭 상향 (5% -> 20%)
        min_r = ref_radius * 0.20
        max_r = ref_radius * 0.80
        
        best_box = None
        best_score = -1.0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
                
            r_est = np.sqrt(area / np.pi)
            
            if min_r <= r_est <= max_r:
                x_b, y_b, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / float(bh)
                
                # 물방울은 극단적으로 길쭉하지 않으므로 비율을 타이트하게 제한
                if 0.5 <= aspect_ratio <= 2.0:
                    cx = x_b + bw / 2.0
                    cy = y_b + bh / 2.0
                    
                    # 마스킹된 동전 영역 근처는 무시
                    if ex1 - 20 <= cx <= ex2 + 20 and ey1 - 20 <= cy <= ey2 + 20:
                        continue
                        
                    dist_to_center = np.sqrt((cx - w/2.0)**2 + (cy - h/2.0)**2)
                    max_dist = np.sqrt((w/2.0)**2 + (h/2.0)**2)
                    center_score = 1.0 - (dist_to_center / (max_dist + 1e-6))
                    
                    # 면적이 클수록 (파편이 아닐수록) 압도적으로 높은 점수 부여
                    score = center_score * (r_est ** 2)
                    
                    if score > best_score:
                        best_score = score
                        best_box = (cx, cy, r_est)
                        
        if best_box is not None:
            cx, cy, r_est = best_box
            # 박스를 살짝 여유있게 잡아줌
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
