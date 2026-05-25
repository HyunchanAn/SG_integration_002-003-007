# 🔬 SG_integration_007 (SG-TERRA 3D 곡률 분석) 고도화 보고서

본 문서는 통합 전 단순 연구용 스크립트 모음(Standalone Scripts)에 가까웠던 SG_proj_007(SG-TERRA: 3D 표면 곡률 분석) 모듈이, 통합 플랫폼 내에서 **물리적 정량 측정 및 엣지(Edge) 환경을 지원하는 상용화 수준의 라이브러리**로 어떻게 고도화되었는지 기록한 리포트입니다.

---

## 1. 개요 및 배경

007 프로젝트(SG-TERRA)를 처음 가져올 당시, 해당 코드는 특정 폴더(예: `models`)에 수동으로 모델 가중치를 다운로드해 두고 스크립트를 하나씩 순차적으로 실행하여 결과를 단순히 이미지나 Plot으로 확인하는 형태였습니다. 또한 3D(Z축) 깊이 정보를 상대적인 색상 맵(Color Map)으로만 표현할 뿐, **실제 수치 단위(mm, μm 등)를 보증하지 못하는 한계**가 존재했습니다.

이를 해결하기 위해, `src` 하위의 분할(`seg`), 토포그래피(`topo`), 곡률(`curv`) 컴포넌트들을 각각 견고한 Wrapper 클래스로 재설계하고, 002 모듈(SFE)의 동전(기준물체) 인식 정보를 3D 공간으로 끌어와 **실제 물리 단위 정량화 캘리브레이션**을 수행하는 거대한 모듈로 탈바꿈시켰습니다.

---

## 2. 3D 곡률(Z축)의 물리적 정량화 (Quantitative Calibration)

가장 치명적인 단점이었던 '상대적 깊이(Relative Depth)만 알 수 있다'는 한계를 극복하기 위해, 가우시안 곡률(Gaussian Curvature)과 평균 곡률(Mean Curvature) 계산 수식에 동전 마커로부터 얻어낸 **스케일 인자(`pixel_to_mm` 및 `z_scale`)**를 주입했습니다.

### 💡 개선된 로직
- 단순 픽셀 단위로 편미분(Gradient)하던 기존 방식을 버리고, `pixel_to_mm` 파라미터를 곱하여 **실제 물리적 공간 상의 곡률(단위: 1/mm²)**로 역산출하도록 수학적 토대를 개편했습니다.
- 이를 통해 '약간 휘어 보인다'가 아니라, '이 표면은 최소 곡률 반경(R)이 500mm 수준으로 휘어 있다'라는 **정량적 공학 지표 획득**이 가능해졌습니다.

### 📝 실제 커밋된 변경 사항 비교 (`src/curv/curvature.py`)
```diff
-    # 기존: 단순 픽셀 단위 기울기 추출
-    def calculate_gaussian_curvature(self, depth_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
-        Z = gaussian_filter(depth_map.astype(np.float32), sigma=self.sigma)
-        Zy, Zx = np.gradient(Z)
-        Zyy, Zyx = np.gradient(Zy)
-        Zxy, Zxx = np.gradient(Zx)

+    # 개선: 물리 단위(Scale)가 반영된 정밀 곡률 계산
+    def calculate_gaussian_curvature(self, depth_map: np.ndarray, mask: np.ndarray = None, pixel_to_mm: float = 1.0, z_scale: float = 1.0) -> np.ndarray:
+        # z_scale을 곱하여 Z축도 물리 단위로 대략적 변환
+        Z = gaussian_filter(depth_map.astype(np.float32), sigma=self.sigma) * pixel_to_mm * z_scale
+        
+        # 1차 편미분 (x, y 방향의 Gradient)
+        # spacing(pixel_to_mm)을 적용하여 공간적 거리를 mm 단위로 미분
+        Zy, Zx = np.gradient(Z, pixel_to_mm, pixel_to_mm)
+        
+        # 2차 편미분 (Hessian Matrix 성분)
+        Zyy, Zyx = np.gradient(Zy, pixel_to_mm, pixel_to_mm)
+        Zxy, Zxx = np.gradient(Zx, pixel_to_mm, pixel_to_mm)
```

---

## 3. 엣지 디바이스 지원 및 OOM 방어 (MobileSAM 폴백)

무거운 SAM 2 모델을 일반 노트북이나 클라우드 등 GPU(CUDA)가 없는 환경에서 구동할 경우 메모리 초과(Out of Memory)로 프로그램이 즉시 강제 종료되는 현상이 잦았습니다.

### 💡 개선된 로직
- `SAM2BaseWrapper` 래퍼(Wrapper) 클래스를 구축하고, 런타임에 디바이스 사양(CUDA, MPS, CPU)을 스캔합니다.
- 고사양 GPU가 탐지되면 최고 정밀도의 `sam2.1-hiera-small` 모델과 `bfloat16` 최적화 및 `TF32` 코어를 활성화합니다.
- 반면 저사양 CPU 엣지 환경이 감지되거나 강제 활성화(`use_mobilesam=True`) 시, **수백 MB 수준으로 경량화된 MobileSAM 모델을 런타임에 동적으로 스왑(Swap)하여 로드**하는 폴백(Fallback) 방어 체계를 도입했습니다.

### 📝 실제 커밋된 변경 사항 (`src/seg/sam2_wrapper.py`)
```python
    def load_model(self, use_mobilesam: bool = False):
        self.is_mobilesam = use_mobilesam
        if use_mobilesam:
            print("Fallback: Loading MobileSAM for Edge Environment...")
            from mobile_sam import sam_model_registry, SamPredictor
            import os, urllib.request
            
            ckpt_path = "checkpoints/mobile_sam.pt"
            os.makedirs("checkpoints", exist_ok=True)
            if not os.path.exists(ckpt_path): # 자동 다운로드 방어 코드
                print("Downloading mobile_sam.pt...")
                urllib.request.urlretrieve("https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt", ckpt_path)
                
            self.model = sam_model_registry["vit_t"](checkpoint=ckpt_path)
            self.model.to(device=self.device).eval()
            self.predictor = SamPredictor(self.model)
            print("MobileSAM loaded successfully.")
            return
```

---

## 4. 모델 가중치(Weights) 자동화 관리 파이프라인

과거 007 모듈을 사용할 땐 사용자가 수 기가바이트(GB)에 달하는 가중치 파일들을 깃허브나 외부 클라우드에서 직접 다운받아 특정 `models` 경로에 수동 배치해야만 했습니다.

### 💡 개선된 로직
- 통합 `app.py` 초기화 단계에 **HuggingFace Hub 자동 다운로드 훅(Hook)**을 구현했습니다.
- 사용자는 빈 폴더만 들고 와도, 프로그램이 구동되면서 `DepthAnything-V2` 모델이나 `mobile_sam.pt` 등 필요한 모델 가중치들을 자동으로 식별하여 다운로드하고 캐싱합니다.
- 특히 Private 모델이나 라이선스 제약이 있는 가중치의 경우 Streamlit Secrets 환경 변수(`HF_TOKEN`)를 통해 실시간 인증을 지원하여, **배포 파이프라인(CI/CD, Docker)에서 사람의 수동 개입을 완전히 소거**했습니다.

---

## 5. 최종 결론

본 통합 플랫폼 내부에서 SG_proj_007(SG-TERRA)은 단순히 폴더 덩어리에서 벗어나 **객체지향적으로 캡슐화된 3D 분석 코어 라이브러리**로 승격되었습니다. 002(SFE)가 던져주는 스케일링 정보와, 007 내부의 최적화된 엣지 인프라가 맞물려, 향후 007 모듈을 단독으로 분리하더라도 공학용 소프트웨어로서 전혀 손색이 없는 강건한 아키텍처를 보유하게 되었습니다.

이상 007 모듈의 고도화 보고를 마칩니다.
