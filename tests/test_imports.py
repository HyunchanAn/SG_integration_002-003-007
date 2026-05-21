import sys
import os

# 프로젝트 루트를 Python 패스에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def test_imports():
    """모든 주요 통합 서브 모듈이 정상적으로 임포트되는지 검증합니다."""
    # 002 (deepdrop_sfe)
    try:
        from deepdrop_sfe import AIContactAngleAnalyzer, DropletPhysics, PerspectiveCorrector
        assert AIContactAngleAnalyzer is not None
        assert DropletPhysics is not None
        assert PerspectiveCorrector is not None
    except ImportError as e:
        assert False, f"Failed to import deepdrop_sfe components: {e}"

    # 003 (vsams)
    try:
        from vsams.analysis.surface_evaluator import SurfaceEvaluator
        assert SurfaceEvaluator is not None
    except ImportError as e:
        assert False, f"Failed to import vsams components: {e}"

    # 007 (src)
    try:
        from src.seg.sam2_wrapper import SAM2BaseWrapper
        from src.topo.depth_wrapper import DepthAnythingV2Wrapper
        from src.curv.curvature import CurvatureAnalyzer
        assert SAM2BaseWrapper is not None
        assert DepthAnythingV2Wrapper is not None
        assert CurvatureAnalyzer is not None
    except ImportError as e:
        assert False, f"Failed to import src (007) components: {e}"

def test_physics_engine_signature():
    """DropletPhysics의 필수 정적 메소드들이 준비되어 있는지 검증합니다."""
    from deepdrop_sfe import DropletPhysics
    assert hasattr(DropletPhysics, "calculate_pixels_per_mm")
    assert hasattr(DropletPhysics, "calculate_contact_diameter")
    assert hasattr(DropletPhysics, "calculate_contact_angle")
    assert hasattr(DropletPhysics, "calculate_owrk")
