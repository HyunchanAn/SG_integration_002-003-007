import torch
import numpy as np
import cv2
import time

class SAM2BaseWrapper:
    def __init__(self, model_id: str = None, device: str = None,
                 # 하위 호환성을 위해 기존 파라미터는 무시 처리
                 model_cfg: str = None, checkpoint_path: str = None):
        """
        SAM 2.1 기반 Target Segmentation 파이프라인 구성.
        HuggingFace Hub에서 모델을 자동 로드하는 방식(build_sam2_hf)을 사용하여
        로컬 YAML config / 체크포인트 경로 호환성 문제를 근본적으로 회피합니다.

        :param model_id: HuggingFace 모델 ID (예: "facebook/sam2.1-hiera-small")
        :param device: 연산 디바이스 (cuda, mps, cpu 자동 할당)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 모델 ID 결정 (GPU 환경에 따른 자동 선택)
        if model_id is None:
            if self.device.type == "cuda":
                model_id = "facebook/sam2.1-hiera-small"
            else:
                model_id = "facebook/sam2.1-hiera-tiny"
        self.model_id = model_id

        print(f"Initializing SAM 2.1 Wrapper on: {self.device} (model: {self.model_id})")

        self.model = None
        self.predictor = None

        # PyTorch Mixed Precision 추론 최적화 (NVIDIA GPU 한정)
        if self.device.type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def load_model(self, use_mobilesam: bool = False):
        """
        HuggingFace Hub 또는 MobileSAM 체크포인트를 통해 모델 파라미터를 메모리에 적재.
        :param use_mobilesam: True일 경우 엣지 환경을 위한 MobileSAM으로 폴백
        """
        self.is_mobilesam = use_mobilesam
        if use_mobilesam:
            print("Fallback: Loading MobileSAM for Edge Environment...")
            try:
                from mobile_sam import sam_model_registry, SamPredictor
            except ImportError:
                raise ImportError("MobileSAM is required for fallback. Install via: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
                
            import os
            import urllib.request
            
            ckpt_path = "checkpoints/mobile_sam.pt"
            os.makedirs("checkpoints", exist_ok=True)
            if not os.path.exists(ckpt_path):
                print("Downloading mobile_sam.pt...")
                urllib.request.urlretrieve("https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt", ckpt_path)
                
            self.model = sam_model_registry["vit_t"](checkpoint=ckpt_path)
            self.model.to(device=self.device).eval()
            self.predictor = SamPredictor(self.model)
            print("MobileSAM loaded successfully.")
            return

        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print(f"Loading SAM 2.1 ({self.model_id}) via HuggingFace Hub...")
        self.model = build_sam2_hf(self.model_id, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print("SAM 2.1 loaded successfully.")

    def segment_target(self, image: np.ndarray, prompt_points: np.ndarray = None, prompt_labels: np.ndarray = None) -> np.ndarray:
        """
        입력 이미지에서 강판(Target)만을 Segmentation 처리.
        :param image: HxWxC 포맷의 numpy 이미지 (RGB)
        :param prompt_points: Nx2 형태의 픽셀 좌표 배열
        :param prompt_labels: N 형태의 레이블 (1: Target, 0: Background)
        :return: 마스킹 된 boolean 2D 배열
        """
        if self.predictor is None:
            self.load_model()
            
        self.predictor.set_image(image)
        
        if prompt_points is None or prompt_labels is None:
            # Default to center point if no prompts provided
            h, w = image.shape[:2]
            prompt_points = np.array([[w//2, h//2]])
            prompt_labels = np.array([1])
            
        masks, scores, logits = self.predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            multimask_output=False
        )
        
        return masks[0].astype(bool)

# 테스트 블럭 (직접 실행 시)
if __name__ == "__main__":
    wrapper = SAM2BaseWrapper()
    wrapper.load_model()
    
    # 1080p 가상 이미지로 Latency 테스트
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    start = time.time()
    mask = wrapper.segment_target(dummy_img)
    end = time.time()
    
    print(f"Mock Output Mask Shape: {mask.shape}")
    print(f"Inference Latency: {(end - start)*1000:.2f} ms")
