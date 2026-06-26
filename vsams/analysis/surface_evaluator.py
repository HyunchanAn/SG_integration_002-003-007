import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from typing import Dict, List, Optional, Union, Any


class SurfaceEvaluator:
    """Core physics-based surface analysis engine of V-SAMS.

    This class provides algorithms to detect standard reference coins in surface images,
    calculate spatial roughness (Ra) and Michelson contrast-based gloss (%), and
    map these metrics to industrial finish standards.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initializes the SurfaceEvaluator with a computational device.

        Args:
            device: Optional torch computational device. If None, automatically detects CUDA, then CPU.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classifier = None

    def analyze(
        self,
        image: Union[Image.Image, np.ndarray],
        custom_boxes: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """Analyzes a surface sample image and extracts physical attributes.

        Args:
            image: The input surface image, either as a PIL Image or OpenCV numpy array.
            custom_boxes: User-specified bounding boxes [[x1, y1, x2, y2], ...] for coin and reflection.

        Returns:
            A dictionary containing physical roughness (Ra), gloss (%), analysis boxes,
            and mapped industrial finish category.
        """
        if isinstance(image, Image.Image):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_cv = image

        # 1. Determine bounding boxes (manual or automatic)
        boxes = custom_boxes
        if boxes is None or len(boxes) < 2:
            auto_boxes = self._auto_detect_boxes(img_cv)
            if auto_boxes:
                boxes = auto_boxes
            else:
                return {
                    "roughness": 0.5,
                    "gloss": 20.0,
                    "has_reflection": False,
                    "error": "동전을 찾을 수 없습니다.",
                }

        # 2. Crop regions of interest
        coin_box, ref_box = boxes[0], boxes[1]
        coin_img = self._crop_box(img_cv, coin_box)
        ref_img = self._crop_box(img_cv, ref_box)

        # 3. Validation guard checks
        if (
            coin_img is None
            or coin_img.size == 0
            or ref_img is None
            or ref_img.size == 0
        ):
            return {
                "roughness": 0.5,
                "gloss": 20.0,
                "has_reflection": False,
                "error": "분석 영역이 유효하지 않습니다 (너무 작거나 경계 밖).",
            }

        # 4. Perform core physical evaluation
        try:
            ra_val, gloss_val = self._estimate_roughness_and_gloss(coin_img, ref_img)

            return {
                "roughness": ra_val,
                "gloss": gloss_val,
                "has_reflection": True,
                "coin_box": coin_box,
                "ref_box": ref_box,
            }
        except Exception as e:
            return {
                "roughness": 0.5,
                "gloss": 20.0,
                "has_reflection": False,
                "error": f"분석 도중 오류 발생: {str(e)}",
            }

    def _auto_detect_boxes(self, img: np.ndarray) -> Optional[List[List[int]]]:
        """Automatically detects the reference coin in the image using HoughCircles and texture std.

        Args:
            img: OpenCV BGR image array.

        Returns:
            A list containing the crop coordinates for [coin_box, reflection_box] if detected, else None.
        """
        orig_h, orig_w = img.shape[:2]
        max_dim = 800.0
        try:
            import streamlit as st
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                max_dim = st.session_state.get("max_image_size") or 800.0
        except:
            max_dim = 800.0
        scale = 1.0
        
        # Optimization: Downsample if image is too large to prevent HoughCircles hanging
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = img

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
            minRadius=int(h * 0.06),
            maxRadius=int(h * 0.18),
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
                pos_score = 1.0 - (
                    dist_to_center / (np.sqrt((w / 2) ** 2 + (h / 2) ** 2))
                )

                # 2. Texture Complexity Score (Coin internal detail)
                roi = gray[cy - cr : cy + cr, cx - cr : cx + cr]
                texture_score = float(np.std(roi)) / 128.0  # type: ignore[arg-type]

                # Weighted score
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

            # Auto-generate corresponding reflection region beneath the coin
            ref_y1 = coin_box[3] + int(15 * inv_scale)
            ref_height = coin_box[3] - coin_box[1]

            ref_box = [
                coin_box[0],
                min(orig_h - int(10 * inv_scale), ref_y1),
                coin_box[2],
                min(orig_h, ref_y1 + ref_height),
            ]

            return [coin_box, ref_box]

        return None

    def _crop_box(self, img: np.ndarray, box: List[int]) -> np.ndarray:
        """Crops an image region specified by the bounding box coords.

        Args:
            img: OpenCV BGR image array.
            box: Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            The cropped sub-image, or empty numpy array if bounds are invalid.
        """
        try:
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                return np.array([])
            return img[y1:y2, x1:x2]
        except Exception:
            return np.array([])

    def _estimate_roughness_and_gloss(self, coin_img: np.ndarray, ref_img: np.ndarray) -> tuple[float, float]:
        """Calculates surface roughness (Ra) and gloss using directionality and sharpness ratios."""
        gray_coin = cv2.cvtColor(coin_img, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

        # Preprocessing to reduce noise
        gray_coin = cv2.GaussianBlur(gray_coin, (3, 3), 0)
        gray_ref = cv2.GaussianBlur(gray_ref, (3, 3), 0)

        # Standard deviation
        std_coin = float(np.std(gray_coin))
        std_ref = float(np.std(gray_ref))

        # Laplacian sharpness
        sharpness_coin = float(cv2.Laplacian(gray_coin, cv2.CV_64F).var())
        sharpness_ref = float(cv2.Laplacian(gray_ref, cv2.CV_64F).var())
        sharp_ratio = sharpness_ref / (sharpness_coin + 1e-6)

        # Sobel gradients to detect directionality (hairline scratches)
        sobelx = cv2.Sobel(gray_ref, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_ref, cv2.CV_64F, 0, 1, ksize=3)
        var_x = float(np.var(sobelx))
        var_y = float(np.var(sobely))
        ratio_xy = var_x / (var_y + 1e-6)

        # Classification and mapping logic
        if ratio_xy > 1.5 or ratio_xy < 0.55:
            # Hairline (HL)
            ra_val = 0.15 + 0.02 * (ratio_xy - 1.5)
            ra_val = max(0.13, min(0.19, ra_val))
            gloss_val = 80.0 + 20.0 * (1.0 / (ratio_xy + 1e-6))
            gloss_val = max(55.0, min(100.0, gloss_val))
        elif sharp_ratio < 0.45:
            # BA (Bright Annealed) / SM
            ra_val = 0.02 + 0.05 * sharp_ratio
            ra_val = max(0.01, min(0.05, ra_val))
            gloss_val = 530.0 + 50.0 * (0.45 - sharp_ratio)
            gloss_val = max(490.0, min(590.0, gloss_val))
        else:
            # 2B (2B/2D)
            ra_val = 0.08 + 0.02 * (sharp_ratio - 0.5)
            ra_val = max(0.06, min(0.09, ra_val))
            gloss_val = 220.0 + 50.0 * (1.0 - sharp_ratio)
            gloss_val = max(170.0, min(240.0, gloss_val))

        return ra_val, gloss_val

    def _estimate_roughness(self, coin_img: np.ndarray, ref_img: np.ndarray) -> float:
        ra, _ = self._estimate_roughness_and_gloss(coin_img, ref_img)
        return ra

    def _estimate_gloss(self, coin_img: np.ndarray, ref_img: np.ndarray) -> float:
        _, gloss = self._estimate_roughness_and_gloss(coin_img, ref_img)
        return gloss



    def get_overlay_image(
        self, image: Union[Image.Image, np.ndarray], result: Dict[str, Any]
    ) -> Image.Image:
        """Overlays detected coin and reflection bounding boxes on the original image.

        Args:
            image: The original image to overlay on.
            result: Analysis output dictionary containing bounding box coordinates.

        Returns:
            PIL Image overlaid with color-coded detection boxes.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(image)
        if "coin_box" in result:
            draw.rectangle(result["coin_box"], outline="blue", width=5)
        if "ref_box" in result:
            draw.rectangle(result["ref_box"], outline="red", width=5)

        return image
