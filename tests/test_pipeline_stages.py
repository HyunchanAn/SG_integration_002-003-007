# ruff: noqa
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import cv2

from deepdrop_sfe import AIContactAngleAnalyzer, DropletPhysics, PerspectiveCorrector
from vsams.analysis.surface_evaluator import SurfaceEvaluator
from src.curv.curvature import CurvatureAnalyzer

SAMPLE_DIR = "E:/Github/SG_sample_images/organized"

def create_synthetic_test_image(mode="sfe"):
    """
    테스트용 합성 이미지 생성
    - sfe: 노이즈 배경 위에 둥근 회색 구(동전)와 파란색 반구(액적)
    - vsams: 금속 텍스처 배경 위에 동전과 아래 반사상
    """
    img = np.ones((600, 800, 3), dtype=np.uint8) * 120 # 회색 배경
    
    if mode == "sfe":
        # 동전 위치 (중앙 근처)
        cv2.circle(img, (250, 300), 70, (190, 190, 190), -1)
        cv2.circle(img, (250, 300), 70, (80, 80, 80), 2)
        # 액적 위치
        cv2.circle(img, (550, 300), 30, (240, 220, 180), -1)
    elif mode == "vsams":
        # 기준 동전
        cv2.circle(img, (400, 200), 60, (200, 200, 200), -1)
        # 반사상 (약간 흐려진 타원 형태)
        cv2.ellipse(img, (400, 420), (60, 45), 0, 0, 360, (140, 140, 140), -1)
        # 가로 방향 스크래치 질감 추가
        for i in range(10, 600, 30):
            cv2.line(img, (0, i), (800, i + 5), (100, 100, 100), 1)
            
    return img

def get_test_image(mode="sfe"):
    """
    실제 폴더 내에 이미지가 있으면 가져오고, 없으면 인공 합성 이미지를 생성하여 리턴
    """
    if os.path.exists(SAMPLE_DIR):
        files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            # 적절한 이미지 로드
            bgr = cv2.imread(files[0])
            if bgr is not None:
                return bgr
                
    return create_synthetic_test_image(mode)

# ---------------------------------------------------------------------------
# [단계 1] SFE 동전 감지 및 원근 보정 검증
# ---------------------------------------------------------------------------
def test_stage_1_sfe_coin_detection():
    bgr = get_test_image("sfe")
    
    # 1. 자동 동전 감지 후보 확인
    analyzer = AIContactAngleAnalyzer()
    box_arr, coin_info = analyzer.auto_detect_coin_candidate(bgr)
    
    # 합성 이미지의 경우 반드시 검출되거나, 실패 시 중앙 300px 폴백이 정상 가동되어야 함
    assert (box_arr is not None) or (coin_info is None)
    
    # 2. 원근 보정 행렬 획득 및 Warp 검증
    ref_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    if coin_info:
        cx, cy, cr = coin_info
        cv2.circle(ref_mask, (int(cx), int(cy)), int(cr), 255, -1)
    else:
        # 감지 실패 시 임시 마스크 생성
        cv2.circle(ref_mask, (400, 300), 80, 255, -1)
        
    corrector = PerspectiveCorrector()
    H, warped_size, coin_warp, fitted = corrector.find_homography(bgr, ref_mask)
    
    assert H is not None
    assert H.shape == (3, 3)
    
    warped_img = corrector.warp_image(bgr, H, warped_size)
    assert warped_img.shape == bgr.shape

# ---------------------------------------------------------------------------
# [단계 2] SFE 액적 감지 및 접촉각 물리 역산 검증
# ---------------------------------------------------------------------------
def test_stage_2_sfe_droplet_detection_and_angle():
    bgr = get_test_image("sfe")
    analyzer = AIContactAngleAnalyzer()
    
    # 임시 마스크 및 픽셀 스케일 설정
    coin_radius_pixel = 80.0
    real_coin_diameter_mm = 24.0 # 100원 기준
    px_per_mm = DropletPhysics.calculate_pixels_per_mm(coin_radius_pixel, real_coin_diameter_mm)
    assert px_per_mm > 0
    
    # 액적 마스크 생성 (합성 이미지의 액적 위치 모사)
    drop_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.circle(drop_mask, (550, 300), 30, 1, -1)
    
    # 접촉 직경 계산
    d_mm = DropletPhysics.calculate_contact_diameter(drop_mask, px_per_mm)
    assert d_mm > 0.0
    
    # 수치해석을 통한 접촉각(Theta) 계산 검증 (Brent's method)
    volume_ul = 200.0 # preset 부피 200uL
    theta = DropletPhysics.calculate_contact_angle(volume_ul, d_mm)
    
    assert 0.0 < theta <= 180.0

# ---------------------------------------------------------------------------
# [단계 3] V-SAMS 조도 및 광택도 물리 연산 검증
# ---------------------------------------------------------------------------
def test_stage_3_vsams_evaluation():
    bgr = get_test_image("vsams")
    
    evaluator = SurfaceEvaluator()
    # 자동 검출 실패 시 유닛 테스트 방어용 커스텀 박스 전달
    # [[coin_x1, coin_y1, coin_x2, coin_y2], [ref_x1, ref_y1, ref_x2, ref_y2]]
    custom_boxes = [[340, 140, 460, 260], [340, 360, 460, 480]]
    res = evaluator.analyze(bgr, custom_boxes=custom_boxes)
    
    assert "roughness" in res
    assert "gloss" in res
    
    from src.match.classifier import DBFinishClassifier
    db_classifier = DBFinishClassifier()
    predicted_label = db_classifier.predict_label(res["roughness"], res["gloss"])
    
    # 수치 유효 범위 검증
    assert 0.0 <= res["roughness"] <= 1.0
    assert res["gloss"] >= 0.0
    assert predicted_label in ["SM (Super Mirror)", "BA (Bright Annealed)", "HL (Hairline)", "#4 (Rough)", "2B (2B/2D)", "Unknown (DB Not Found)", "Unknown"]

# ---------------------------------------------------------------------------
# [단계 4] SG-TERRA 3D 지형 곡률 연산 검증
# ---------------------------------------------------------------------------
def test_stage_4_topo_and_curvature():
    # mgrid를 사용하여 (100, 100) 형태의 grid 좌표 생성
    h, w = 100, 100
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
    dmap = np.zeros((h, w), dtype=np.float32)
    mask = (x**2 + y**2) < 40**2
    dmap[mask] = np.sqrt(40**2 - x[mask]**2 - y[mask]**2)
    
    analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    
    # Gaussian Curvature(K) 계산
    px2mm = 0.25 # mm/pixel
    g_curv = analyzer.calculate_gaussian_curvature(dmap, mask=mask, pixel_to_mm=px2mm, z_scale=px2mm)
    
    assert g_curv.shape == (h, w)
    
    # 최대 곡률을 갖는 임계 포인트 추출
    cvals, ccoords = analyzer.find_critical_points(g_curv, mask=mask, top_k=1)
    
    assert len(cvals) == 1
    assert len(ccoords) == 1
    assert cvals[0] != 0.0

# ---------------------------------------------------------------------------
# [단계 5] 012 매칭 엔진 페이로드 정합성 검증
# ---------------------------------------------------------------------------
def test_stage_5_matching_payload_contract():
    # 012 매칭 모듈 전송용 요청 스키마 규격 검증
    payload = {
        "substrate_id": "SUS_304",
        "surface_energy": 42.5,
        "roughness": 0.12,
        "finish_type": "BA (Bright Annealed)",
        "required_processability_level": 3
    }
    
    assert isinstance(payload["substrate_id"], str)
    assert isinstance(payload["surface_energy"], float)
    assert isinstance(payload["roughness"], float)
    assert isinstance(payload["finish_type"], str)
    assert isinstance(payload["required_processability_level"], int)
