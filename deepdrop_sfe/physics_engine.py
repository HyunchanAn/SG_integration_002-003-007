# ruff: noqa: RUF012, UP007, E501, N806
from typing import Any, Optional, Union

import cv2
import numpy as np
from scipy.optimize import brentq


class DropletPhysics:
    """
    물체 부피와 접촉 직경을 기반으로 접촉각을 계산하는 물리 엔진.
    Spherical Cap 지오메트리를 가정합니다.
    """

    # 액체 물성 데이터 (20도 기준)
    LIQUID_DATA: dict[str, dict[str, float]] = {
        "Water(DI)": {"g": 72.8, "d": 21.8, "p": 51.0},
        "Diiodomethane": {"g": 50.8, "d": 50.8, "p": 0.0},
        "Ethylene Glycol": {"g": 48.0, "d": 29.0, "p": 19.0},
        "Glycerol": {"g": 64.0, "d": 34.0, "p": 30.0},
        "Formamide": {"g": 58.0, "d": 39.0, "p": 19.0},
    }

    @staticmethod
    def calculate_pixels_per_mm(coin_radius_pixel: float, real_coin_diameter_mm: float) -> float:
        """
        픽셀 스케일(pixels per mm)을 계산합니다.

        Args:
            coin_radius_pixel (float): 원근 보정된 동전의 반경 (pixels).
            real_coin_diameter_mm (float): 실제 동전 직경 (mm).

        Returns:
            float: 1mm당 픽셀 수 (pixels per mm). 반경이 0 이하인 경우 0.0을 반환합니다.
        """
        if coin_radius_pixel <= 0:
            return 0.0
        coin_diameter_pixel = 2.0 * coin_radius_pixel
        return coin_diameter_pixel / real_coin_diameter_mm

    @staticmethod
    def calculate_contact_diameter(
        droplet_mask: np.ndarray, pixels_per_mm: float, return_extra: bool = False
    ) -> Union[float, tuple[float, float]]:
        """
        액적 마스크로부터 실제 접촉 직경(mm)을 계산합니다.

        Args:
            droplet_mask (np.ndarray): 액적 영역이 표시된 이진 마스크 (boolean 또는 uint8).
            pixels_per_mm (float): 픽셀 스케일 변환 계수.
            return_extra (bool, optional): True일 경우 원형도(Circularity) 지표를 함께 반환합니다. 기본값은 False.

        Returns:
            Union[float, Tuple[float, float]]: 접촉 직경 (mm).
                return_extra가 True인 경우 (접촉 직경, 원형도 점수) 튜플을 반환합니다.
        """
        if pixels_per_mm <= 0:
            return (0.0, 0.0) if return_extra else 0.0

        contours, _ = cv2.findContours(
            droplet_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return (0.0, 0.0) if return_extra else 0.0

        cnt = max(contours, key=cv2.contourArea)
        area_pixels = float(cv2.contourArea(cnt))
        peri_pixels = float(cv2.arcLength(cnt, True))

        if area_pixels < 10:
            return (0.0, 0.0) if return_extra else 0.0

        # 원형도 계산
        circularity = (4.0 * np.pi * area_pixels) / (peri_pixels**2) if peri_pixels > 0 else 0.0

        # 지오메트릭 피팅: 타원 피팅 (사선 촬영 보정 후에도 잔존하는 비대칭성 대응)
        if len(cnt) >= 5:
            _, (d1, d2), _ = cv2.fitEllipse(cnt)
            diameter_pixels = float(max(d1, d2))
        else:
            _, radius = cv2.minEnclosingCircle(cnt)
            diameter_pixels = float(2.0 * radius)

        diameter_mm = diameter_pixels / pixels_per_mm

        if return_extra:
            return diameter_mm, float(circularity)
        return diameter_mm

    @staticmethod
    def calculate_contact_angle(
        volume_ul: float, diameter_mm: float, return_info: bool = False
    ) -> Union[float, tuple[float, dict[str, Any]]]:
        """
        접촉각(Theta)을 수치 해석적으로 역산합니다.

        원형 캡 지오메트리 방정식:
        V = (pi * r^3 * (1 - cos(theta))^2 * (2 + cos(theta))) / (3 * sin(theta)^3)

        Args:
            volume_ul (float): 주입한 액체의 실제 부피 (microliter, uL).
            diameter_mm (float): 측정된 액적의 접촉 직경 (mm).
            return_info (bool, optional): True일 경우 역산 수치 분석 메타데이터를 함께 반환합니다. 기본값은 False.

        Returns:
            Union[float, Tuple[float, Dict[str, Any]]]: 계산된 접촉각 (degree).
                return_info가 True인 경우 (접촉각, 수치 분석 진단 딕셔너리) 튜플을 반환합니다.
        """
        diag: dict[str, Any] = {
            "v_low": 0.0,
            "v_high": 0.0,
            "target_V": volume_ul,
            "r": diameter_mm / 2.0 if diameter_mm else 0.0,
            "v_full": 0.0,
            "status": "Initializing",
        }

        if diameter_mm <= 0.0 or volume_ul <= 0.0:
            if volume_ul <= 0.0:
                return (0.0, diag) if return_info else 0.0
            if diameter_mm <= 0.0:
                diag["status"] = "Capped: Non-wetting"
                return (180.0, diag) if return_info else 180.0

        r = diameter_mm / 2.0
        target_V = volume_ul
        v_full = (4.0 / 3.0) * np.pi * (r**3)
        diag["v_full"] = v_full

        def volume_eq(theta_deg: float) -> float:
            theta_deg_clipped = np.clip(theta_deg, 1e-7, 179.99)
            theta_rad = np.radians(theta_deg_clipped)
            sin_t = np.sin(theta_rad)
            cos_t = np.cos(theta_rad)
            term = ((1 - cos_t) ** 2 * (2 + cos_t)) / (sin_t**3)
            V_calc = (np.pi * r**3 / 3.0) * term
            return float(V_calc - target_V)

        try:
            v_low = volume_eq(1e-7)
            v_high = volume_eq(179.9)
            diag["v_low"] = v_low
            diag["v_high"] = v_high

            if v_low * v_high > 0:
                if target_V >= v_full:
                    diag["status"] = "Capped: Volume exceeds sphere"
                    return (180.0, diag) if return_info else 180.0
                diag["status"] = "Warning: Sign mismatch"
                return (0.0, diag) if return_info else 0.0

            theta_sol = float(brentq(volume_eq, 1e-7, 179.9))
            diag["status"] = "Success"

            # --- 민감도 분석 (수치 미분) ---
            eps = 0.01

            def get_angle(v: float, d: float) -> float:
                r_l = d / 2.0

                def eq(t: float) -> float:
                    tr = np.radians(np.clip(t, 1e-7, 179.99))
                    val = (
                        (np.pi * r_l**3 / 3.0)
                        * ((1 - np.cos(tr)) ** 2 * (2 + np.cos(tr)))
                        / (np.sin(tr) ** 3)
                    )
                    return float(val - v)

                try:
                    return float(brentq(eq, 1e-7, 179.9))
                except Exception:
                    return theta_sol

            angle_v_plus = get_angle(target_V * (1.0 + eps), diameter_mm)
            diag["sensitivity_v"] = (angle_v_plus - theta_sol) / eps

            angle_d_plus = get_angle(target_V, diameter_mm * (1.0 + eps))
            diag["sensitivity_d"] = (angle_d_plus - theta_sol) / eps

            return (theta_sol, diag) if return_info else theta_sol
        except Exception as e:
            diag["status"] = f"Error: {e}"
            return (0.0, diag) if return_info else 0.0

    @staticmethod
    def calculate_owrk(measurements: list[dict[str, Any]]) -> tuple[Optional[float], float, float]:
        """
        OWRK 표면 에너지 계산 (선형 회귀).

        Args:
            measurements (List[Dict[str, Any]]): 측정된 액체 종류와 접촉각 딕셔너리 리스트.
                예: [{'liquid': 'Water(DI)', 'angle': 72.5}, ...]

        Returns:
            Tuple[Optional[float], float, float]: (총 표면 에너지, 분산 성분, 극성 성분).
                측정 데이터가 2개 미만인 경우 (None, 0.0, 0.0)을 반환합니다.
        """
        X_points: list[float] = []
        Y_points: list[float] = []

        for m in measurements:
            name = m["liquid"]
            angle = m["angle"]
            if name not in DropletPhysics.LIQUID_DATA:
                continue
            props = DropletPhysics.LIQUID_DATA[name]
            if props["d"] <= 0:
                continue

            theta_rad = np.radians(angle)
            y_val = (props["g"] * (1 + np.cos(theta_rad))) / (2 * np.sqrt(props["d"]))
            x_val = np.sqrt(props["p"] / props["d"])

            X_points.append(float(x_val))
            Y_points.append(float(y_val))

        if len(X_points) < 2:
            return None, 0.0, 0.0

        A = np.vstack([X_points, np.ones(len(X_points))]).T
        slope, intercept = np.linalg.lstsq(A, Y_points, rcond=None)[0]

        gamma_s_p = float(slope**2)
        gamma_s_d = float(intercept**2)
        total_sfe = gamma_s_d + gamma_s_p

        return total_sfe, gamma_s_d, gamma_s_p
