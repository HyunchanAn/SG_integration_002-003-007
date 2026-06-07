import torch
import numpy as np
import cv2
import time

class DepthAnythingV2Wrapper:
    def __init__(self, encoder: str = 'vitl', 
                 checkpoint_path: str = "../../models/depth_anything_v2/depth_anything_v2_vitl.pth", 
                 device: str = None):
        """
        Depth-Anything-V2 기반 3D Topography 재구성 파이프라인.
        :param encoder: 모델 인코더 크기 ('vits', 'vitb', 'vitl'). 기본값은 Large.
        :param checkpoint_path: 가중치 파일 경로
        :param device: 연산 디바이스 (cuda, mps, cpu 자동 할당)
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing Depth-Anything-V2 Wrapper on: {self.device}")
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.model = None
        
        # PyTorch FP16/BF16 Mixed Precision 추론 최적화 (NVIDIA GPU 한정)
        if self.device.type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8: # Ampere 아키텍처 이상 TF32 허용
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def load_model(self):
        """
        모델 파라미터를 메모리에 적재.
        """
        from src.topo.depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        print("Loading Depth-Anything-V2 checkpoints...")
        self.model = DepthAnythingV2(**model_configs[self.encoder])
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        print("Depth-Anything-V2 loaded successfully.")

    def estimate_depth(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        단안 이미지에서 해상도 높은 Depth Map 추정.
        SAM 마스크를 활용하여 ROI 영역에 집중, Latency와 노이즈 최적화.
        :param image: HxWxC 포맷의 numpy 이미지 (RGB)
        :param mask: SAM 2에서 추출한 불리언 2D 마스크
        :return: 픽셀별 상대적 깊이값 배열 (Z-axis)
        """
        if self.model is None:
            self.load_model()
            
        # SAM 마스크가 제공된 경우 배경 픽셀 클렌징으로 노이즈 최소화 
        if mask is not None:
            mask_bool = np.array(mask, dtype=bool).squeeze()
            proc_img = image.copy()
            proc_img[~mask_bool] = 0 
        else:
            proc_img = image
            
        # Real inference
        depth = self.model.infer_image(proc_img)
        
        return depth

# 테스트 블럭 (직접 실행 시)
if __name__ == "__main__":
    wrapper = DepthAnythingV2Wrapper()
    wrapper.load_model()
    
    # 1080p 가상 이미지로 테스트
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_mask = np.ones((1080, 1920), dtype=bool) 
    # 약간의 여백 생성
    dummy_mask[:100, :] = False
    dummy_mask[-100:, :] = False
    
    start = time.time()
    depth_map = wrapper.estimate_depth(dummy_img, mask=dummy_mask)
    end = time.time()
    
    print(f"Mock Output Depth Map Shape: {depth_map.shape}")
    print(f"Inference Latency: {(end - start)*1000:.2f} ms")
