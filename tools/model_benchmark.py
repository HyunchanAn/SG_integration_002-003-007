# ruff: noqa
import os
import time
import numpy as np
import cv2
import torch
import streamlit as st

# 모듈 경로 세팅
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepdrop_sfe import AIContactAngleAnalyzer
from src.seg.sam2_wrapper import SAM2BaseWrapper
from src.topo.depth_wrapper import DepthAnythingV2Wrapper

SAMPLE_DIR = "E:/Github/SG_sample_images/organized"

def generate_synthetic_image():
    """
    실제 이미지가 없을 때 모델 동작 및 속도 측정을 위해 임시로 생성하는 
    동전 및 액적 형상이 포함된 합성 이미지 (1280x720)
    """
    # 1280x720x3 검은색 배경 생성
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # 금속 질감을 흉내 낸 노이즈 삽입
    noise = np.random.normal(120, 15, (720, 1280, 3)).astype(np.uint8)
    img = cv2.addWeighted(img, 0.0, noise, 1.0, 0)
    
    # 동전 형상 그리기 (회색 원, 중심 300, 360, 반경 80)
    cv2.circle(img, (300, 360), 80, (180, 180, 180), -1)
    # 동전 테두리 묘사
    cv2.circle(img, (300, 360), 80, (100, 100, 100), 3)
    
    # 액적 형상 그리기 (파란색 투명 물방울 느낌, 중심 800, 360, 반경 40)
    cv2.circle(img, (800, 360), 40, (230, 200, 150), -1) # 원형 맺힘
    
    return img

def run_benchmark():
    print("=" * 60)
    print("AI MODEL PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Active Device: {device}")
    
    # 1. 대상 이미지 준비
    img_list = []
    if os.path.exists(SAMPLE_DIR):
        files = [os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            print(f"Found {len(files)} target images in {SAMPLE_DIR}")
            for f in files[:3]: # 최대 3장 샘플링
                bgr = cv2.imread(f)
                if bgr is not None:
                    img_list.append((os.path.basename(f), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    
    if not img_list:
        print(f"Warning: No images found in {SAMPLE_DIR}. Generating synthetic testing image.")
        syn_bgr = generate_synthetic_image()
        img_list.append(("synthetic_sample.png", cv2.cvtColor(syn_bgr, cv2.COLOR_BGR2RGB)))

    # 2. 모델 로드 및 초기화
    print("\n[Stage 1] Loading Model Weights...")
    try:
        # Depth Anything V2 Wrapper 초기화
        da_ckpt = "models/depth_anything_v2/depth_anything_v2_vits.pth"
        if not os.path.exists(da_ckpt):
            # 허깅페이스 다운로드 대리 호출
            print("Downloading Depth Anything V2 Vits checkpoint...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Small", filename="depth_anything_v2_vits.pth", local_dir="models/depth_anything_v2")
            
        t0 = time.time()
        depth_w = DepthAnythingV2Wrapper(encoder="vits", checkpoint_path=da_ckpt, device=device)
        depth_w.load_model()
        dt_load_depth = (time.time() - t0) * 1000
        print(f"-> Depth-Anything-V2 Model Load: {dt_load_depth:.1f} ms")
        
        # SAM 2 Wrapper 초기화
        t0 = time.time()
        sam2_w = SAM2BaseWrapper()
        sam2_w.load_model(use_mobilesam=False)
        dt_load_sam = (time.time() - t0) * 1000
        print(f"-> SAM 2 Model Load: {dt_load_sam:.1f} ms")
        
    except Exception as e:
        print(f"Fatal Error: Model weight loading failed. {e}")
        return

    # 3. 모델 추론 벤치마크 수행
    print("\n[Stage 2] Commencing Inference Benchmarks...")
    for name, img_rgb in img_list:
        print("-" * 50)
        print(f"Testing File: {name} (Shape: {img_rgb.shape})")
        
        # A. SAM 2 세그멘테이션 테스트
        h, w = img_rgb.shape[:2]
        # 임의의 프롬프트 포인트 지정 (대략 이미지 중앙 영역)
        prompt_pts = np.array([[w // 4, h // 2]])
        prompt_lbls = np.array([1])
        
        try:
            t0 = time.time()
            mask = sam2_w.segment_target(img_rgb, prompt_points=prompt_pts, prompt_labels=prompt_lbls)
            dt_sam = (time.time() - t0) * 1000
            mask_ratio = (np.sum(mask) / (h * w)) * 100
            print(f"SAM 2 Segment: Latency = {dt_sam:.1f} ms | Mask Density = {mask_ratio:.2f}%")
        except Exception as e:
            print(f"SAM 2 Segment: FAILED - {e}")
            mask = None

        # B. Depth Anything V2 뎁스맵 추론 테스트
        try:
            t0 = time.time()
            dmap = depth_w.estimate_depth(img_rgb, mask=mask)
            dt_depth = (time.time() - t0) * 1000
            d_min, d_max = np.min(dmap), np.max(dmap)
            print(f"Depth Estimate: Latency = {dt_depth:.1f} ms | Depth Range = [{d_min:.1f} ~ {d_max:.1f}]")
        except Exception as e:
            print(f"Depth Estimate: FAILED - {e}")

    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
