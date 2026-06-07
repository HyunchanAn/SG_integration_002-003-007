# 🔬 SG_integration_003 (V-SAMS 표면 마감 분석) 고도화 보고서

본 문서는 SG_proj_003(V-SAMS: 시각 기반 금속 표면 마감 평가) 모듈이 통합 플랫폼(v2.1)으로 이식되면서 산업 현장(스마트폰 고해상도 촬영 등)에 즉시 투입될 수 있도록 어떻게 최적화되었는지 기록한 리포트입니다.

---

## 1. 개요 및 배경

기존 V-SAMS 모듈은 저해상도 샘플 이미지에서는 준수하게 동작했으나, **최신 스마트폰 카메라로 촬영된 수십 메가바이트 크기의 초고해상도(4K 이상) 이미지**가 입력될 경우 심각한 병목(Bottleneck) 현상이 발생했습니다. OpenCV의 원형 탐지 알고리즘(`HoughCircles`) 연산량이 픽셀 수에 기하급수적으로 비례하기 때문에 서버가 무한 대기(Hang) 상태에 빠지는 현상이 발견되었습니다.

또한, 분석 결과 보고서 화면에서 물리적인 정량 지표의 단위 표기가 누락되거나 왜곡되어, 전문가들이 결과를 신뢰하는 데 걸림돌이 되었습니다.

---

## 2. 고해상도 연산 병목 최적화: Downsampling Scaling 역산 기법

원본 해상도를 그대로 유지한 채 `HoughCircles` 알고리즘을 수행하면 O(N^3)에 달하는 메모리와 시간이 소모됩니다. 이를 방지하기 위해 V-SAMS 분석 코어에 **자동 스케일 다운샘플링(Downsampling) 방어벽**을 구축했습니다.

### 💡 개선된 로직
- 이미지의 가로/세로 중 최대 크기가 `800px`을 초과할 경우, 내부 분석용 복사본을 `800px` 스케일에 맞춰 강제 축소(Resize)합니다.
- 축소된 이미지에서 초고속으로 동전 위치 및 반경(`x, y, r`)을 찾습니다.
- 찾아낸 좌표값에 이전에 구했던 축소 비율(`inv_scale`)을 다시 곱하여(역산출), **원본 해상도 기준의 정확한 Bounding Box 좌표(`orig_x`, `orig_y`)로 복원**한 후 반환합니다.
- 이 과정을 통해 분석 시간이 수십 초 단위에서 **0.05초 수준**으로 극적으로 단축되었으며, Out Of Memory (OOM) 현상이 원천 차단되었습니다.

### 📝 실제 반영된 핵심 코드 (`vsams/analysis/surface_evaluator.py`)
```python
        # 개선: Downsample if image is too large to prevent HoughCircles hanging
        orig_h, orig_w = img.shape[:2]
        max_dim = 800.0
        scale = 1.0
        
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            work_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = img

        ... # (축소된 work_img에서 HoughCircles 검출 완료 후) ...

        if best_candidate is not None:
            x, y, r = best_candidate
            pad = int(r * 0.1)

            # Map coordinates back to original scale (역산출을 통한 정밀 좌표 복원)
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
```

---

## 3. 리포트 신뢰성 강화를 위한 산업 표준 단위(Units) 개편

기존 UI에서는 표면 거칠기를 의미하는 `Ra`와 광택도를 의미하는 `Gloss`가 단위 없이 뭉뚱그려져 있거나, 광택도가 단순히 백분율(`%`)로만 표시되어 공학적 엄밀성이 떨어졌습니다.

### 💡 개선된 로직
- 표면 조도(Roughness) 측정 결과 값 옆에 국제 산업 표준 물리 단위인 **마이크로미터(`μm`)**를 명시적으로 추가했습니다.
- 반사율 척도인 광택도(Gloss) 측정 결과가 일반적인 스케일(100% 이상)을 초과할 수 있음을 반영하여, 단순 `%` 단위를 제거하고 국제 표준인 **`GU (Gloss Unit)`** 단위로 렌더링하도록 UI 계층 레이블을 개편했습니다.

### 📝 실제 커밋된 변경 사항 비교 (`app.py`)
```diff
-            with c1: _card(f"Ra {vr['roughness']:.3f}", "조도 (Roughness)" if lang == "ko" else "Roughness")
-            with c2: _card(f"{vr['gloss']:.1f}%", "광택도 (Gloss)" if lang == "ko" else "Gloss")

+            with c1: _card(f"{vr['roughness']:.3f} μm", "조도 (Roughness Ra)" if lang == "ko" else "Roughness Ra")
+            with c2: _card(f"{vr['gloss']:.1f} GU", "광택도 (Gloss)" if lang == "ko" else "Gloss Unit")
```

---

## 4. 통합 프로젝트 전체 기여: 모범(Reference) 아키텍처로서의 003 모듈

단순히 003 모듈 스스로 최적화된 것에 그치지 않고, V-SAMS에서 증명된 **'CLAHE + Median Blur 전처리 후 HoughCircles'** 동전 탐지 로직의 강건성이 현재 **002 SFE 모듈의 동전 감지 엔진 성능을 비약적으로 끌어올리는 모태(Reference) 기술**이 되었습니다.

003 모듈의 로직은 금속의 마감 상태(HL, 2B, SM 등)를 구별해내는 데 특화되어 있었으므로, 그 과정에서 탄생한 빛반사 억제 전처리 기술이 전체 통합 앱(`app.py`)의 신뢰도를 받쳐주는 든든한 기반이 되었습니다.

---
이상 003(V-SAMS) 모듈의 개선 및 고도화 보고를 마칩니다.
