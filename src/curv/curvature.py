import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class CurvatureAnalyzer:
    def __init__(self, smoothing_sigma: float = 2.0):
        """
        Depth Map을 기반으로 3D 표면의 곡률(Gaussian/Mean Curvature)을 분석하는 모듈.
        :param smoothing_sigma: 노이즈 저감을 위한 Gaussian Smoothing의 강도
        """
        self.sigma = smoothing_sigma

    def calculate_gaussian_curvature(self, depth_map: np.ndarray, mask: np.ndarray = None, pixel_to_mm: float = 1.0, z_scale: float = 1.0) -> np.ndarray:
        """
        Depth Map 식으로부터 Gaussian Curvature(K)를 계산.
        K = (Zxx * Zyy - Zxy^2) / (1 + Zx^2 + Zy^2)^2
        단, 편미분 계산 전 고주파 노이즈 제거를 위해 Smoothing 적용.
        
        :param depth_map: Depth-Anything-V2에서 추출된 2D Depth 배열
        :param mask: 강판 타겟 영역 마스크 (배경 배제)
        :param pixel_to_mm: 1 픽셀당 mm 스케일 (스케일링 인자)
        :param z_scale: Z축(Depth) 변환용 보정 계수
        :return: 각 픽셀별 물리적 스케일이 적용된 Gaussian Curvature 배열 (단위: 1/mm^2)
        """
        # 노이즈를 줄여 미분 오차 방지
        # z_scale을 곱하여 Z축도 물리 단위로 대략적 변환
        Z = gaussian_filter(depth_map.astype(np.float32), sigma=self.sigma) * pixel_to_mm * z_scale
        
        # 1차 편미분 (x, y 방향의 Gradient)
        # spacing(pixel_to_mm)을 적용하여 공간적 거리를 mm 단위로 미분
        Zy, Zx = np.gradient(Z, pixel_to_mm, pixel_to_mm)
        
        # 2차 편미분 (Hessian Matrix 성분)
        Zyy, Zyx = np.gradient(Zy, pixel_to_mm, pixel_to_mm)
        Zxy, Zxx = np.gradient(Zx, pixel_to_mm, pixel_to_mm)
        
        # Gaussian Curvature 연산 (K)
        # 평면 = 0, 볼록/오목 = 양수, 말안장(Saddle) = 음수
        numerator = (Zxx * Zyy) - (Zxy ** 2)
        denominator = (1 + Zx**2 + Zy**2) ** 2
        
        K = numerator / denominator
        
        # 마스크 바깥(배경) 영역의 곡률은 0으로 날림
        if mask is not None:
            K[~mask] = 0.0
            
        return K

    def calculate_mean_curvature(self, depth_map: np.ndarray, mask: np.ndarray = None, pixel_to_mm: float = 1.0, z_scale: float = 1.0) -> np.ndarray:
        """
        Depth Map 식으로부터 Mean Curvature(H)를 계산.
        H = ((1 + Zx^2)*Zyy - 2*Zx*Zy*Zxy + (1 + Zy^2)*Zxx) / (2*(1 + Zx^2 + Zy^2)^(3/2))
        """
        Z = gaussian_filter(depth_map.astype(np.float32), sigma=self.sigma) * pixel_to_mm * z_scale
        
        Zy, Zx = np.gradient(Z, pixel_to_mm, pixel_to_mm)
        Zyy, Zyx = np.gradient(Zy, pixel_to_mm, pixel_to_mm)
        Zxy, Zxx = np.gradient(Zx, pixel_to_mm, pixel_to_mm)
        
        numerator = (1 + Zx**2)*Zyy - 2*Zx*Zy*Zxy + (1 + Zy**2)*Zxx
        denominator = 2 * (1 + Zx**2 + Zy**2)**1.5
        
        # 방어 로직: 분모가 0이 되는 것을 방지
        H = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        if mask is not None:
            H[~mask] = 0.0
            
        return H
        
    def find_critical_points(self, curvature_map: np.ndarray, mask: np.ndarray = None, top_k: int = 5):
        """
        곡률이 가장 심한 임계 영역(Stress Concentration)의 픽셀 좌표를 도출.
        절대값이 클수록 급격하게 꺾임.
        :param curvature_map: Gaussian 또는 Mean Curvature 배열
        :return: (상위 k개의 최대 곡률값, 해당 좌표 리스트)
        """
        search_space = np.abs(curvature_map)
        
        if mask is not None:
            search_space[~mask] = 0.0

        # 가장 곡률 절대값이 높은 인덱스 찾기
        flat_indices = np.argsort(search_space, axis=None)[::-1][:top_k]
        
        # 1D 인덱스를 2D y, x 좌표로 변환
        coords = [np.unravel_index(idx, search_space.shape) for idx in flat_indices]
        values = [curvature_map[y, x] for y, x in coords]
        
        return values, coords

# 테스트 블럭 (직접 실행 시)
if __name__ == "__main__":
    analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    
    # 가상의 반구형 뎁스맵 생성 (중심이 가장 볼록함)
    h, w = 100, 100
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    # 반경 30인 구의 상단부 (Z = sqrt(R^2 - x^2 - y^2))
    r2 = 30**2
    mask = (x**2 + y**2) < r2
    dummy_depth = np.zeros((h, w), dtype=np.float32)
    dummy_depth[mask] = np.sqrt(r2 - x[mask]**2 - y[mask]**2)
    
    # 계산
    gaussian_c = analyzer.calculate_gaussian_curvature(dummy_depth, mask=mask)
    
    # 결과 출력
    critical_vals, critical_coords = analyzer.find_critical_points(gaussian_c, mask=mask, top_k=3)
    print("\n[Curvature Analysis Mock Results]")
    for i, (val, coord) in enumerate(zip(critical_vals, critical_coords), 1):
         print(f"Top {i} Stress Point at {coord}: Curvature = {val:.4f}")
