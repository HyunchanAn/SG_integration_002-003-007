# SG Integration Platform Architecture

이 문서는 `SG_integration_002+003+007` 통합 플랫폼의 내부 데이터 흐름과 아키텍처를 설명합니다.

## System Workflow (Mermaid)

```mermaid
graph TD
    User([User]) --> |Image Uploads & Params| UI[app.py (Streamlit UI)]
    
    subgraph SFE Pipeline (002)
        UI --> |f_polar, f_nonpolar| DropletPhysics
        DropletPhysics --> |Contact Angle, OWRK| SFEResult[SFE Data]
    end
    
    subgraph V-SAMS Pipeline (003)
        UI --> |f_finish| SurfaceEvaluator
        SurfaceEvaluator --> |Ra, Glossiness| FinishResult[Surface Data]
    end
    
    subgraph SG-TERRA Pipeline (007)
        UI --> |f_3d, sigma, ref_len| CurvatureAnalyzer
        CurvatureAnalyzer --> SAM[SAM2BaseWrapper]
        SAM --> Depth[DepthAnythingV2Wrapper]
        Depth --> CurvResult[3D Curvature Map & Max Curvature]
    end
    
    SFEResult --> Report[Tab 4: Consolidated Report]
    FinishResult --> Report
    CurvResult --> Report
    
    Report --> |Metrics & Visuals| UI
```

## 모듈 간 결합 (Dependencies)
- **Frontend/Orchestrator**: `app.py`는 모든 하위 모듈을 호출하고 데이터를 통합하여 리포트(Tab 4)를 생성합니다.
- **DeepDrop-SFE (002)**: 동전 감지 및 액적 검출, 접촉각 OWRK 모델 연산을 담당합니다.
- **V-SAMS (003)**: 반사상의 선명도를 바탕으로 표면 거칠기(Ra) 및 광택도를 평가합니다.
- **SG-TERRA (007)**: SAM2 기반 정밀 분할과 Depth Anything V2를 결합하여 물체의 3D 곡률 지형도를 생성합니다.
