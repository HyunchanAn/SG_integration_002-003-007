"""
Integrated Surface Analysis Platform
- Tab 1: SFE (Surface Free Energy) via SG_proj_002 (deepdrop_sfe)
- Tab 2: Surface Finish (V-SAMS) via SG_proj_003 (vsams)
- Tab 3: 3D Curvature (SG-TERRA) via SG_proj_007 (src)
- Tab 4: Consolidated Report
"""
import io
import os
import sys
import time
import urllib.request

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Streamlit 1.34+ 호환성 패치 (image_to_url 누락 대응)
# ---------------------------------------------------------------------------
import streamlit.elements.image as _st_image
if not hasattr(_st_image, "image_to_url"):
    try:
        from streamlit.runtime import get_instance as _get_inst
        def _image_to_url(data, width, clamp, channels, output_format, image_id):
            rt = _get_inst()
            if not isinstance(data, (bytes, bytearray)):
                img = Image.fromarray(data) if not isinstance(data, Image.Image) else data
                buf = io.BytesIO(); img.save(buf, format="PNG"); data = buf.getvalue()
            return rt.media_file_mgr.add(data, "image/png", width, image_id)
        _st_image.image_to_url = _image_to_url
    except Exception:
        pass

# streamlit_image_coordinates - 선택적 임포트
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_IMG_COORDS = True
except ImportError:
    HAS_IMG_COORDS = False

# ---------------------------------------------------------------------------
# 핵심 라이브러리 임포트 (각 모듈은 프로젝트 루트에 복사되어 있음)
# ---------------------------------------------------------------------------
# Streamlit의 핫 리로드(Hot-Reload) 모듈 와처 스레드와 경합으로 인해 발생하는
# sys.modules KeyError(예: KeyError: 'src.seg', KeyError: 'src.topo') 문제를 방지하기 위해
# 임포트 전 sys.modules에서 로컬 패키지 엔트리를 강제로 초기화
import sys
for _k in list(sys.modules.keys()):
    if _k in ["src", "deepdrop_sfe", "vsams"] or _k.startswith(("src.", "deepdrop_sfe.", "vsams.")):
        sys.modules.pop(_k, None)

from deepdrop_sfe import AIContactAngleAnalyzer, DropletPhysics, PerspectiveCorrector
from vsams.analysis.surface_evaluator import SurfaceEvaluator
from src.seg.sam2_wrapper import SAM2BaseWrapper
from src.topo.depth_wrapper import DepthAnythingV2Wrapper
from src.curv.curvature import CurvatureAnalyzer

# ---------------------------------------------------------------------------
# Page Config (반드시 맨 앞에서 호출해야 함)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Integrated Surface Analysis",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== Design System: Premium Glassmorphism & Touch Optimized ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0B1120;
        --bg-secondary: #111827;
        --bg-card: rgba(26, 35, 50, 0.65);
        --bg-card-hover: rgba(30, 42, 58, 0.8);
        --border-default: rgba(30, 58, 95, 0.6);
        --border-accent: rgba(37, 99, 235, 0.8);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #3B82F6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-violet: #8B5CF6;
        --gradient-primary: linear-gradient(135deg, rgba(30,58,95,0.4) 0%, rgba(11,17,32,0.6) 100%);
        --gradient-accent: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        --shadow-card: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 20px;
        --glass-blur: blur(12px);
    }

    /* ===== Animations ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp {
        animation: fadeIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    /* ===== Global Typography ===== */
    html, body, .main, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary);
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 960px;
    }

    /* ===== Title Area ===== */
    h1 {
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-size: 1.8rem !important;
    }
    h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .stCaption, [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
        font-size: 13px;
        letter-spacing: 0.02em;
    }

    /* ===== Expander (Accordion Cards) - Glassmorphism ===== */
    [data-testid="stExpander"] {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg) !important;
        margin-bottom: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-card);
        transition: border-color .3s ease, background .3s ease;
    }
    [data-testid="stExpander"] summary {
        padding: 16px 20px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderToggleDetails"] {
        font-weight: 600 !important;
        font-size: 16px !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stExpander"] summary span[data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
    }

    /* ===== Metric Card ===== */
    .metric-card {
        background: var(--gradient-primary);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default);
        padding: 24px 16px;
        border-radius: var(--radius-md);
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: var(--gradient-accent);
        border-radius: var(--radius-md) var(--radius-md) 0 0;
    }
    .mv {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #60A5FA;
        letter-spacing: -0.02em;
    }
    .ml {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 8px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* ===== Buttons - Touch Optimized ===== */
    .stButton > button {
        background: var(--gradient-accent) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        min-height: 48px !important; /* Touch target */
        padding: 0 24px !important;
        transition: transform .1s cubic-bezier(0.4, 0, 0.2, 1), opacity .2s !important;
        letter-spacing: 0.01em;
        width: 100%;
    }
    .stButton > button:active {
        transform: scale(0.96);
        opacity: 0.8;
    }
    .stButton > button[kind="secondary"],
    .stDownloadButton > button {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default) !important;
        color: var(--text-primary) !important;
        min-height: 48px !important;
        width: 100%;
    }
    .stDownloadButton > button:active {
        background: rgba(37,99,235,.15) !important;
        border-color: var(--border-accent) !important;
    }

    /* ===== Input Widgets - Touch Optimized ===== */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: rgba(17, 24, 39, 0.7) !important;
        backdrop-filter: var(--glass-blur);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-sm) !important;
        min-height: 48px !important;
        font-size: 15px !important;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus,
    .stSelectbox [data-baseweb="select"]:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,.25) !important;
    }

    /* ===== Slider ===== */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: var(--accent-blue) !important;
        width: 24px !important;
        height: 24px !important;
    }

    /* ===== File Uploader ===== */
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: var(--glass-blur);
        border: 1px dashed var(--border-default);
        border-radius: var(--radius-md);
        padding: 16px;
        transition: border-color .2s;
    }
    [data-testid="stFileUploader"]:active {
        border-color: var(--accent-blue);
        background: rgba(37,99,235,.05);
    }

    /* ===== Radio & Checkbox ===== */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        color: var(--text-secondary) !important;
        min-height: 32px;
        display: flex;
        align-items: center;
    }
    
    /* ===== Dataframe ===== */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    /* ===== Divider ===== */
    hr {
        border-color: var(--border-default) !important;
        opacity: 0.5;
        margin: 24px 0;
    }

    /* ===== Section Labels inside expanders ===== */
    [data-testid="stExpander"] h4 {
        font-size: 16px !important;
        color: var(--accent-cyan) !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 10px;
        margin-bottom: 16px;
    }

    /* ===== Mobile Responsive Override ===== */
    @media (max-width: 768px) {
        .main .block-container { 
            padding-left: 16px; 
            padding-right: 16px; 
        }
        h1 { font-size: 1.5rem !important; }
        .mv { font-size: 20px; }
        .ml { font-size: 11px; }
        .metric-card { padding: 20px 14px; }
        
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State 초기화
# ---------------------------------------------------------------------------
_defaults = {
    "vol_ul": 200.0,
    "coin_d": 24.0,
    "sfe_results": {},        # {liquid_key: {"ca":..., "d_mm":..., ...}}
    "sfe_calc_result": None,  # OWRK 연산 결과
    "m_list": [],             # SFE 테이블 리스트
    "v_sams_result": None,    # V-SAMS 분석 결과 dict
    "curv_result": None,      # 3D curvature 분석 결과 dict
    "pts_007": [],            # 캘리브레이션 포인트 (3D 탭)
    "last_pt_007": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# 다국어 사전
# ---------------------------------------------------------------------------
_T = {
    "ko": {
        "title": "통합 표면 분석 플랫폼",
        "subtitle": "표면 자유 에너지 / 마감 상태 / 3D 곡률 복합 분석",
        "tab1": "표면 에너지 (SFE)",
        "tab2": "표면 마감 (V-SAMS)",
        "tab3": "3D 곡률 (SG-TERRA)",
        "tab4": "통합 리포트",
        "opt_3d": "3D 곡률 분석도 수행",
        "params": "분석 매개변수",
        "upload_h": "이미지 업로드",
        "f1": "액체 #1 이미지 (SFE)",
        "f2": "액체 #2 이미지 (SFE)",
        "f3": "마감 평가 이미지 (V-SAMS)",
        "f4": "3D 곡률 이미지 (수직 촬영)",
        "volume": "액적 부피 (uL)",
        "coin_d": "동전 직경 (mm)",
        "sigma": "곡률 평활화 sigma",
        "sigma_help": "3D 지형(Depth) 복원 시 노이즈를 억제하는 필터 강도입니다. 값이 클수록 표면이 부드럽게 해석되어 전반적인 휘어짐을 파악하기 좋고, 작을수록 미세한 찍힘이 그대로 반영됩니다.",
        "ref_len": "캘리브레이션 길이 (mm)",
        "loading": "모델 가중치를 로드하는 중입니다...",
        "no_img": "상단 설정 영역에서 이미지를 업로드해 주세요.",
        "no_data": "아직 분석 데이터가 없습니다. 각 탭에서 분석을 먼저 수행해 주세요.",
    },
    "en": {
        "title": "Integrated Surface Analysis Platform",
        "subtitle": "SFE / Finish / 3D Curvature — Multi-modal Analysis",
        "tab1": "Surface Energy (SFE)",
        "tab2": "Surface Finish (V-SAMS)",
        "tab3": "3D Curvature (SG-TERRA)",
        "tab4": "Consolidated Report",
        "opt_3d": "Also run 3D curvature analysis",
        "params": "Analysis Parameters",
        "upload_h": "Image Upload",
        "f1": "Liquid #1 image (SFE)",
        "f2": "Liquid #2 image (SFE)",
        "f3": "Finish evaluation image (V-SAMS)",
        "f4": "3D curvature image (vertical shot)",
        "volume": "Droplet volume (uL)",
        "coin_d": "Coin diameter (mm)",
        "sigma": "Curvature smoothing sigma",
        "ref_len": "Calibration length (mm)",
        "loading": "Loading AI model weights...",
        "no_img": "Please upload images in the settings above.",
        "no_data": "No analysis data yet. Run measurements first.",
    },
}

# ---------------------------------------------------------------------------
# Sidebar 제거 및 메인 상단 설정으로 통합 (기기 환경 대응)
# ---------------------------------------------------------------------------
if "lang_sel" not in st.session_state:
    st.session_state["lang_sel"] = "한국어"
if "do_3d" not in st.session_state:
    st.session_state["do_3d"] = False

# 메인 페이지 최상단 타이틀 렌더링을 위해 언어 우선 결정
lang = "ko" if st.session_state["lang_sel"] == "한국어" else "en"
T = _T[lang]

# 브랜드 헤더
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        width:42px; height:42px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        font-size:20px; color:white; flex-shrink:0;
    ">&#x1F52C;</div>
    <div>
        <div style="font-size:1.5rem; font-weight:700;
            background: linear-gradient(90deg,#60A5FA,#A78BFA,#34D399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            letter-spacing:-0.02em;">""" + T["title"] + """</div>
        <div style="font-size:12px; color:#64748B; letter-spacing:0.04em; margin-top:2px;">
            """ + T["subtitle"] + """ &nbsp;|&nbsp; <span style="
                background:rgba(37,99,235,.15); color:#60A5FA;
                padding:2px 8px; border-radius:4px; font-size:11px;
                font-weight:600;">v2.1</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# STEP 1. 설정 및 이미지 등록 (Expander)
with st.expander("STEP 1.  설정 및 이미지 등록" if lang == "ko" else "STEP 1.  Settings & Image Uploads", expanded=True):
    col_lang, col_opt = st.columns(2)
    with col_lang:
        lang_sel = st.radio("Language / 언어", ["한국어", "English"], index=0 if st.session_state["lang_sel"] == "한국어" else 1, horizontal=True)
        if lang_sel != st.session_state["lang_sel"]:
            st.session_state["lang_sel"] = lang_sel
            st.rerun()
    with col_opt:
        do_3d = st.checkbox(T["opt_3d"], value=st.session_state["do_3d"])
        if do_3d != st.session_state["do_3d"]:
            st.session_state["do_3d"] = do_3d
            st.rerun()

    st.markdown("---")
    st.markdown("##### " + T["params"])
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        def update_vol():
            if st.session_state.vol_preset:
                st.session_state.vol_ul = float(st.session_state.vol_preset)

        vol_ul = st.number_input(T["volume"], 0.1, 1000.0, key="vol_ul", step=1.0)
        st.pills("Preset", ["20", "50", "100", "200", "500"], key="vol_preset", on_change=update_vol, label_visibility="collapsed")
        
    with col_p2:
        def update_coin():
            if st.session_state.coin_preset:
                labels = ["10원", "50원", "100원", "500원"]
                vals = [18.0, 21.6, 24.0, 26.5]
                idx = labels.index(st.session_state.coin_preset)
                st.session_state.coin_d = vals[idx]

        coin_d = st.number_input(T["coin_d"], 1.0, 100.0, key="coin_d", step=0.5)
        st.pills("Coin", ["10원", "50원", "100원", "500원"], key="coin_preset", on_change=update_coin, label_visibility="collapsed")

    st.markdown("---")

    # 이미지 등록 슬롯 카드 시스템
    _slot_cfg = [
        {
            "label": T["f1"],
            "desc": "접촉각 측정용 첫 번째 시약 액적 사진" if lang == "ko" else "Droplet photo for first reagent",
            "color": "#3B82F6",
            "icon": "&#x1F4A7;",
            "key_up": "up1",
            "key_cam": "cam1",
            "var": "f_polar",
        },
        {
            "label": T["f2"],
            "desc": "접촉각 측정용 두 번째 시약 액적 사진" if lang == "ko" else "Droplet photo for second reagent",
            "color": "#8B5CF6",
            "icon": "&#x1F4A7;",
            "key_up": "up2",
            "key_cam": "cam2",
            "var": "f_nonpolar",
        },
        {
            "label": T["f3"],
            "desc": "동전 반사상 기반 표면 마감 평가용" if lang == "ko" else "Coin reflection for surface finish evaluation",
            "color": "#10B981",
            "icon": "&#x1F50D;",
            "key_up": "up3",
            "key_cam": "cam3",
            "var": "f_finish",
        },
    ]
    if do_3d:
        _slot_cfg.append({
            "label": T["f4"],
            "desc": "수직 촬영 사진으로 3D 곡률 복원" if lang == "ko" else "Vertical shot for 3D curvature reconstruction",
            "color": "#F59E0B",
            "icon": "&#x1F4D0;",
            "key_up": "up4",
            "key_cam": "cam4",
            "var": "f_3d",
        })

    use_camera = st.toggle(
        "카메라로 직접 촬영" if lang == "ko" else "Use camera",
        value=False,
        key="use_cam_toggle",
    )

    _uploaded = {}
    for slot in _slot_cfg:
        st.markdown(f"""
        <div style="
            background: var(--bg-secondary, #111827);
            border: 1px solid {slot['color']}33;
            border-left: 4px solid {slot['color']};
            border-radius: 10px;
            padding: 12px 14px 4px 14px;
            margin-bottom: 8px;
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="font-size:16px;">{slot['icon']}</span>
                <span style="font-weight:600; font-size:14px; color:{slot['color']};">{slot['label']}</span>
            </div>
            <div style="font-size:11px; color:#64748B; margin-bottom:8px;">{slot['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        if slot["var"] in ["f_polar", "f_nonpolar"]:
            from deepdrop_sfe.physics_engine import DropletPhysics as _DP
            _liq_names = list(_DP.LIQUID_DATA.keys())
            idx = 0 if slot["var"] == "f_polar" else 1
            
            def _sync_liq(var_name=slot["var"]):
                up_key = f"chem_{var_name}_up"
                target_key = "chem_liquid_1" if var_name == "f_polar" else "chem_liquid_2"
                st.session_state[target_key] = st.session_state[up_key]
                
            st.selectbox(
                "시약 종류", _liq_names, index=idx, 
                key=f"chem_{slot['var']}_up", 
                on_change=_sync_liq, kwargs={"var_name": slot["var"]},
                label_visibility="collapsed"
            )
            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

        if use_camera:
            _uploaded[slot["var"]] = st.camera_input(
                slot["label"],
                key=slot["key_cam"],
                label_visibility="collapsed",
            )
        else:
            _uploaded[slot["var"]] = st.file_uploader(
                slot["label"],
                type=["jpg", "jpeg", "png"],
                key=slot["key_up"],
                label_visibility="collapsed",
            )

    f_polar = _uploaded.get("f_polar")
    f_nonpolar = _uploaded.get("f_nonpolar")
    f_finish = _uploaded.get("f_finish")
    f_3d = _uploaded.get("f_3d") if do_3d else None

    sigma_v = 2.0
    ref_len = 100.0
    if do_3d:
        st.markdown("---")
        st.markdown("##### 3D 곡률 세부 매개변수" if lang == "ko" else "##### 3D Curvature Parameters")
        c_3d1, c_3d2 = st.columns(2)
        with c_3d1:
            sigma_v = st.slider(T["sigma"], 0.5, 5.0, 2.0, 0.1, help=T.get("sigma_help", ""))
        with c_3d2:
            ref_len = st.number_input(T["ref_len"], 1.0, 5000.0, 100.0, 1.0, help=T.get("ref_len_help", ""))

# ---------------------------------------------------------------------------
# Model 캐싱 로드
# ---------------------------------------------------------------------------
import hashlib
from huggingface_hub import hf_hub_download

@st.cache_resource(show_spinner=False)
def _load_engines():
    """AI 모델 자원 일괄 로드 및 무결성 검증"""
    # 1. Depth-Anything-V2
    da_enc    = "vits"
    da_ckpt   = "models/depth_anything_v2/depth_anything_v2_vits.pth"
    # HF Hub 리포지토리 지정 (실제 계정으로 변경 필요)
    HF_REPO_ID = "HyunchanAn/sg-integration-weights"
    
    os.makedirs("models/depth_anything_v2", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("vsams/data", exist_ok=True)
    
    weights_info = [
        {"path": da_ckpt, "hf_repo": "depth-anything/Depth-Anything-V2-Small", "hf_file": "depth_anything_v2_vits.pth"},
        {"path": "checkpoints/v_sams_model.pth", "hf_repo": HF_REPO_ID, "hf_file": "v_sams_model.pth"},
        {"path": "vsams/data/visual_library.pth", "hf_repo": HF_REPO_ID, "hf_file": "visual_library.pth"},
        {"path": "checkpoints/mobile_sam.pt", "hf_repo": HF_REPO_ID, "hf_file": "mobile_sam.pt"},
    ]
    
    for w in weights_info:
        if not os.path.exists(w["path"]):
            try:
                print(f"Downloading {w['hf_file']} from Hugging Face Hub...")
                hf_hub_download(repo_id=w["hf_repo"], filename=w["hf_file"], local_dir=os.path.dirname(w["path"]))
            except Exception as e:
                print(f"Warning: Failed to download {w['hf_file']} from {w['hf_repo']}. Error: {e}")
                # 수동 다운로드 폴백을 위해 패스 (앱 런타임에서 에러 처리)
                pass

    sfe_az  = AIContactAngleAnalyzer()
    sfe_pc  = PerspectiveCorrector()
    vs_eval = SurfaceEvaluator()
    sam_w   = SAM2BaseWrapper()
    dep_w   = DepthAnythingV2Wrapper(encoder=da_enc, checkpoint_path=da_ckpt)
    cur_a   = CurvatureAnalyzer(smoothing_sigma=2.0)
    
    # 엣지 환경 대응 (UI에서 toggle로 제어할 수도 있으나, 기본 SAM2 호출)
    sam_w.load_model(use_mobilesam=False)
    dep_w.load_model()
    return sfe_az, sfe_pc, vs_eval, sam_w, dep_w, cur_a

with st.spinner(T["loading"]):
    sfe_analyzer, sfe_corrector, vsams_eval, sam2_w, depth_w, curv_a = _load_engines()

# ---------------------------------------------------------------------------
# Helper: 이미지 로딩 (UploadedFile -> BGR / RGB)
# ---------------------------------------------------------------------------
def _load_img(uploaded):
    """UploadedFile -> (bgr, rgb) numpy arrays"""
    raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    uploaded.seek(0)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb

# Helper: metric card
def _card(value: str, label: str):
    st.markdown(
        f'<div class="metric-card"><div class="mv">{value}</div>'
        f'<div class="ml">{label}</div></div>',
        unsafe_allow_html=True,
    )

# Helper: droplet profile 그래프
def _droplet_plot(ca_deg: float):
    fig, ax = plt.subplots(figsize=(4, 2))
    theta = np.radians(ca_deg)
    ax.plot([-1.5, 1.5], [0, 0], color="#475569", linewidth=2)
    r = 1.0
    if ca_deg < 90:
        R = r / np.sin(theta); yc = -r / np.tan(theta)
        c = plt.Circle((0, yc), R, color="#38BDF8", alpha=0.55)
        ax.add_patch(c); ax.set_ylim(0, 1.5)
    else:
        R = r / np.sin(np.pi - theta); yc = -r / np.tan(theta)
        c = plt.Circle((0, yc), R, color="#0284C7", alpha=0.55)
        ax.add_patch(c); ax.set_ylim(0, 2.2)
    ax.set_xlim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    return fig


# =========================================================================
# STEP 2 — Surface Free Energy (SFE)
# =========================================================================
with st.expander("STEP 2.  " + T["tab1"], expanded=True):
    st.markdown(
        "OWRK 모델 기반 2-액체 접촉각 측정으로 표면 자유 에너지를 정밀하게 산출합니다."
        if lang == "ko"
        else "Calculates Surface Free Energy via OWRK model using contact angles from two liquids."
    )

    if f_polar is None or f_nonpolar is None:
        st.info(T["no_img"])
    else:
        # Step 1에서 선택한 시약 종류 획득
        chem_name_1 = st.session_state.get("chem_f_polar_up", "Water")
        chem_name_2 = st.session_state.get("chem_f_nonpolar_up", "Diiodomethane")

        # 루프를 통해 액체 #1과 #2의 측정 영역을 세로로 연속 배치
        for liq_key, active_file, chem_name in [
            ("liquid_1", f_polar, chem_name_1),
            ("liquid_2", f_nonpolar, chem_name_2),
        ]:
            st.markdown(f"### {'액체 #1' if liq_key == 'liquid_1' else '액체 #2'} ({chem_name})")
            
            bgr, rgb = _load_img(active_file)

            # --- Step 1: 동전 감지 ---
            st.markdown("---")
            st.markdown("#### 1. 기준 물체(동전) 감지" if lang == "ko" else "#### 1. Reference Coin Detection")
            
            # 수동 지정 제어 체크박스 추가
            manual_coin = st.checkbox("동전 영역 수동 지정" if lang == "ko" else "Manual Coin Input", key=f"manual_{liq_key}")
            
            coin_box = None
            coin_ok = False
            
            col_o, col_d = st.columns(2)
            
            if manual_coin:
                # 수동 지정 모드
                with col_o:
                    st.markdown("이미지에서 동전의 중심을 클릭하세요." if lang == "ko" else "Click the center of the coin on the image.")
                    
                    pt_key = f"pt_coin_{liq_key}"
                    if pt_key not in st.session_state:
                        # 다른 탭에서 이미 설정한 공유 좌표가 있다면 기본값으로 할당
                        st.session_state[pt_key] = st.session_state.get("shared_coin_pt", None)
                    
                    disp_rgb = rgb.copy()
                    click_pt = st.session_state[pt_key]
                    
                    # 동전 반경을 조절할 수 있는 슬라이더 (공유 반경 동기화 적용)
                    default_r = st.session_state.get("shared_coin_r", 300)
                    r_val = st.slider("동전 반경 (px)" if lang == "ko" else "Coin Radius (px)", 10, 800, default_r, key=f"r_{liq_key}")
                    st.session_state["shared_coin_r"] = r_val
                    
                    if click_pt is not None:
                        cx, cy = click_pt
                        cv2.circle(disp_rgb, (cx, cy), 12, (0, 255, 0), -1)
                        cv2.rectangle(disp_rgb, (cx - r_val, cy - r_val), (cx + r_val, cy + r_val), (0, 255, 0), 6)
                    
                    if HAS_IMG_COORDS:
                        display_w = 340
                        h_orig, w_orig = rgb.shape[:2]
                        click_res = streamlit_image_coordinates(disp_rgb, width=display_w, key=f"coords_coin_{liq_key}")
                        if click_res is not None:
                            scale_factor = w_orig / display_w
                            clicked_coord = (
                                int(click_res["x"] * scale_factor),
                                int(click_res["y"] * scale_factor)
                            )
                            if clicked_coord != st.session_state.get(f"last_coords_{liq_key}"):
                                st.session_state[pt_key] = clicked_coord
                                st.session_state[f"last_coords_{liq_key}"] = clicked_coord
                                st.session_state["shared_coin_pt"] = clicked_coord
                                # 세션 상태 강제 동기화로 number_input 재사용 캐시 꼬임 해결
                                st.session_state[f"cx_in_{liq_key}"] = clicked_coord[0]
                                st.session_state[f"cy_in_{liq_key}"] = clicked_coord[1]
                                st.rerun()
                    else:
                        st.image(disp_rgb, width="stretch")
                        st.warning("streamlit-image-coordinates 패키지가 비활성화되어 수동 클릭을 사용할 수 없습니다.")
                
                with col_d:
                    # 숫자 입력을 이용한 강제 좌표 지정 폴백
                    h, w = rgb.shape[:2]
                    st.markdown("좌표 미세 조정 및 폴백 입력" if lang == "ko" else "Coordinate Fine-tuning & Fallback")
                    
                    default_cx = st.session_state[pt_key][0] if st.session_state[pt_key] else w // 2
                    default_cy = st.session_state[pt_key][1] if st.session_state[pt_key] else h // 2
                    
                    # 강제 위젯 세션 값 매핑 초기화
                    if f"cx_in_{liq_key}" not in st.session_state or st.session_state[pt_key] is None:
                        st.session_state[f"cx_in_{liq_key}"] = int(default_cx)
                    if f"cy_in_{liq_key}" not in st.session_state or st.session_state[pt_key] is None:
                        st.session_state[f"cy_in_{liq_key}"] = int(default_cy)
                    
                    cx_input = st.number_input("중심 X 좌표 (px)" if lang == "ko" else "Center X (px)", 0, w, key=f"cx_in_{liq_key}")
                    cy_input = st.number_input("중심 Y 좌표 (px)" if lang == "ko" else "Center Y (px)", 0, h, key=f"cy_in_{liq_key}")
                    
                    # 수동 좌표를 기반으로 최종 바운딩 박스(coin_box) 빌드
                    coin_box = np.array([
                        max(0, cx_input - r_val),
                        max(0, cy_input - r_val),
                        min(w, cx_input + r_val),
                        min(h, cy_input + r_val)
                    ])
                    # 미세조정 값을 글로벌 공유 세션에 업데이트
                    st.session_state["shared_coin_pt"] = (cx_input, cy_input)
                    st.session_state["shared_coin_r"] = r_val
                    coin_ok = True
                    
                    # 최종 적용된 바운딩 박스 및 실시간 SAM 2 코인 마스크 오버레이 시각화
                    prev = rgb.copy()
                    with st.spinner("Calculating coin mask..."):
                        try:
                            sfe_analyzer.set_image(rgb)
                            mask_coin, _ = sfe_analyzer.predict_mask(box=coin_box)
                            mask_bin = sfe_analyzer.get_binary_mask(mask_coin)
                            
                            # 빨간색 마스크 쉐이딩 추가
                            overlay = prev.copy()
                            overlay[mask_bin > 0] = [255, 100, 100]
                            cv2.addWeighted(overlay, 0.4, prev, 0.6, 0, prev)
                            
                            # 윤곽선도 함께 매핑
                            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(prev, contours, -1, (255, 50, 50), 4)
                        except Exception as mask_err:
                            st.warning(f"마스크 오버레이 생성 오류: {mask_err}")
        
                    cv2.rectangle(prev, (int(coin_box[0]), int(coin_box[1])), (int(coin_box[2]), int(coin_box[3])), (0, 255, 0), 8)
                    st.image(prev, caption="Detected Coin Mask (Red) & Target Area (Green)", width="stretch")
                    
                    if st.button("수동 지정 포인트 초기화" if lang == "ko" else "Reset Manual Target", key=f"rst_manual_{liq_key}"):
                        st.session_state[pt_key] = None
                        st.session_state["shared_coin_pt"] = None
                        st.rerun()
            else:
                # 자동 감지 모드
                h, w = rgb.shape[:2]
                with col_o:
                    st.image(rgb, caption="Original", width="stretch")
                with col_d:
                    # 이미 신뢰할 수 있게 설정된 다른 탭의 공유 좌표가 있는지 확인
                    shared_pt = st.session_state.get("shared_coin_pt", None)
                    shared_r = st.session_state.get("shared_coin_r", 300)
                    
                    coin_box = None
                    
                    if shared_pt is not None:
                        # 이전에 확정된 동전 정보를 우선적으로 재사용하여 탭 간 무결성 확보
                        cx_s, cy_s = shared_pt
                        coin_box = np.array([
                            max(0, cx_s - shared_r),
                            max(0, cy_s - shared_r),
                            min(w, cx_s + shared_r),
                            min(h, cy_s + shared_r)
                        ])
                        st.info("이전 분석에서 확정된 동전 좌표 정보를 기반으로 자동 세그멘테이션을 수행합니다." if lang == "ko"
                                else "Using confirmed coin coordinates from previous analysis for auto segmentation.")
                    else:
                        with st.spinner("Detecting coin..."):
                            coin_box, _ = sfe_analyzer.auto_detect_coin_candidate(bgr)
                        
                        is_too_small = False
                        if coin_box is not None:
                            c_area = (coin_box[2] - coin_box[0]) * (coin_box[3] - coin_box[1])
                            if c_area / (h * w) < 0.005: # 너무 작은 노이즈 영역은 차단
                                is_too_small = True
                                
                        # OpenCV 검출에 실패했거나 노이즈인 경우 중앙부 가상 박스로 폴백
                        if coin_box is None or is_too_small:
                            coin_box = np.array([
                                max(0, w // 2 - 300),
                                max(0, h // 2 - 300),
                                min(w, w // 2 + 300),
                                min(h, h // 2 + 300)
                            ])
                            st.info("동전 외곽을 자동 검출하지 못하여 이미지 중앙을 기준으로 자동 세그멘테이션을 시도합니다." if lang == "ko"
                                    else "Coin boundaries not found. Attempting auto segmentation on image center.")
                    
                    if coin_box is not None:
                        # 획득된 동전 위치를 글로벌 공유 좌표 세션에 저장
                        st.session_state["shared_coin_pt"] = (int((coin_box[0] + coin_box[2])/2), int((coin_box[1] + coin_box[3])/2))
                        st.session_state["shared_coin_r"] = int((coin_box[2] - coin_box[0])/2)
                        
                        prev = rgb.copy()
                        # 자동 감지에서도 획득된 SAM 2 마스크를 빨간색으로 오버레이하여 사전 정합성 검증
                        with st.spinner("Calculating coin mask..."):
                            try:
                                sfe_analyzer.set_image(rgb)
                                mask_coin, _ = sfe_analyzer.predict_mask(box=coin_box)
                                mask_bin = sfe_analyzer.get_binary_mask(mask_coin)
                                
                                overlay = prev.copy()
                                overlay[mask_bin > 0] = [255, 100, 100]
                                cv2.addWeighted(overlay, 0.4, prev, 0.6, 0, prev)
                                
                                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(prev, contours, -1, (255, 50, 50), 4)
                            except Exception as mask_err:
                                pass
                                
                        x1, y1, x2, y2 = map(int, coin_box)
                        cv2.rectangle(prev, (x1, y1), (x2, y2), (0, 255, 0), 8)
                        st.image(prev, caption="Detected Coin Mask (Red)", width="stretch")
                        coin_ok = st.checkbox(
                            "동전 위치 확인" if lang == "ko" else "Confirm coin",
                            value=True, key=f"ck_{liq_key}",
                        )

            # --- Step 2: Homography + 액적 분석 ---
            if coin_ok and coin_box is not None:
                st.markdown("---")
                st.markdown("#### 2. 원근 보정 및 액적 분석" if lang == "ko" else "#### 2. Perspective Correction & Droplet Analysis")

                sfe_analyzer.set_image(rgb)
                mask_coin, _ = sfe_analyzer.predict_mask(box=coin_box)
                mask_bin = sfe_analyzer.get_binary_mask(mask_coin)
                H, ws, coin_info, _ = sfe_corrector.find_homography(rgb, mask_bin)

                if H is not None:
                    warped = sfe_corrector.warp_image(rgb, H, ws)
                    
                    # 액적 영역 수동 지정 제어 체크박스 추가
                    manual_droplet = st.checkbox("액적 영역 수동 지정" if lang == "ko" else "Manual Droplet Input", key=f"manual_drop_{liq_key}")
                    
                    col_w1, col_w2 = st.columns(2)
                    drop_box = None
                    
                    if manual_droplet:
                        # 액적 수동 지정 모드
                        with col_w1:
                            st.markdown("보정 이미지에서 액적의 중심을 클릭하세요." if lang == "ko" else "Click the center of the droplet on the warped image.")
                            
                            drop_pt_key = f"pt_drop_{liq_key}"
                            if drop_pt_key not in st.session_state:
                                st.session_state[drop_pt_key] = None
                            
                            disp_warp = warped.copy()
                            click_drop = st.session_state[drop_pt_key]
                            
                            if click_drop is not None:
                                dcx, dcy = click_drop
                                cv2.circle(disp_warp, (dcx, dcy), 8, (255, 80, 80), -1)
                            
                            if HAS_IMG_COORDS:
                                display_w_warp = 340
                                hw_orig, ww_orig = warped.shape[:2]
                                click_res_warp = streamlit_image_coordinates(disp_warp, width=display_w_warp, key=f"coords_drop_{liq_key}")
                                if click_res_warp is not None:
                                    scale_factor_warp = ww_orig / display_w_warp
                                    clicked_coord_warp = (
                                        int(click_res_warp["x"] * scale_factor_warp),
                                        int(click_res_warp["y"] * scale_factor_warp)
                                    )
                                    if clicked_coord_warp != st.session_state.get(f"last_coords_drop_{liq_key}"):
                                        st.session_state[drop_pt_key] = clicked_coord_warp
                                        st.session_state[f"last_coords_drop_{liq_key}"] = clicked_coord_warp
                                        # 세션 상태 강제 주입으로 number_input 재사용 캐시 꼬임 원천 해결
                                        st.session_state[f"dcx_in_{liq_key}"] = clicked_coord_warp[0]
                                        st.session_state[f"dcy_in_{liq_key}"] = clicked_coord_warp[1]
                                        st.rerun()
                            else:
                                st.image(disp_warp, width="stretch")
                                st.warning("streamlit-image-coordinates 패키지가 비활성화되어 수동 클릭을 사용할 수 없습니다.")
                        
                        with col_w2:
                            hw, ww = warped.shape[:2]
                            st.markdown("액적 좌표 미세 조정" if lang == "ko" else "Droplet Fine-tuning")
                            default_dcx = st.session_state[drop_pt_key][0] if st.session_state[drop_pt_key] else ww // 2
                            default_dcy = st.session_state[drop_pt_key][1] if st.session_state[drop_pt_key] else hw // 2
                            
                            # 강제 위젯 세션 값 매핑 초기화
                            if f"dcx_in_{liq_key}" not in st.session_state or st.session_state[drop_pt_key] is None:
                                st.session_state[f"dcx_in_{liq_key}"] = int(default_dcx)
                            if f"dcy_in_{liq_key}" not in st.session_state or st.session_state[drop_pt_key] is None:
                                st.session_state[f"dcy_in_{liq_key}"] = int(default_dcy)
                            
                            dcx_input = st.number_input("중심 X 좌표 (px)" if lang == "ko" else "Center X (px)", 0, ww, key=f"dcx_in_{liq_key}")
                            dcy_input = st.number_input("중심 Y 좌표 (px)" if lang == "ko" else "Center Y (px)", 0, hw, key=f"dcy_in_{liq_key}")
                            
                            # 최종 액적 포인트 및 실시간 SAM 2 액적 마스크 오버레이 시각화
                            dprev = warped.copy()
                            with st.spinner("Calculating droplet mask..."):
                                try:
                                    sfe_analyzer.set_image(warped)
                                    pt_coords = np.array([[dcx_input, dcy_input]])
                                    pt_labels = np.array([1])
                                    d_mask, _ = sfe_analyzer.predict_mask(point_coords=pt_coords, point_labels=pt_labels)
                                    d_mask_bin = sfe_analyzer.get_binary_mask(d_mask)
                                    
                                    # 빨간색 액적 마스크 쉐이딩 추가
                                    d_overlay = dprev.copy()
                                    d_overlay[d_mask_bin > 0] = [255, 100, 100]
                                    cv2.addWeighted(d_overlay, 0.4, dprev, 0.6, 0, dprev)
                                    
                                    # 윤곽선 추가
                                    d_contours, _ = cv2.findContours(d_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    cv2.drawContours(dprev, d_contours, -1, (255, 50, 50), 3)
                                except Exception as d_mask_err:
                                    st.warning(f"액적 마스크 생성 실패: {d_mask_err}")
                                    
                            cv2.circle(dprev, (int(dcx_input), int(dcy_input)), 8, (255, 80, 80), -1)
                            st.image(dprev, caption="Detected Droplet Mask (Red) & Target Point", width="stretch")
                            
                            if st.button("수동 지정 포인트 초기화" if lang == "ko" else "Reset Droplet Point", key=f"rst_drop_{liq_key}"):
                                st.session_state[drop_pt_key] = None
                                st.rerun()
                    else:
                        # 액적 자동 감지 모드
                        with col_w1:
                            st.image(warped, caption="Top-View", width="stretch")
                        with col_w2:
                            with st.spinner("Detecting droplet..."):
                                # 원근 보정 후 계산된 동전의 새로운 좌표계를 바탕으로 exclude_box를 구성
                                if coin_info is not None:
                                    ccx, ccy, ccr = coin_info
                                    exclude_box_warped = [
                                        max(0, ccx - ccr),
                                        max(0, ccy - ccr),
                                        ccx + ccr,
                                        ccy + ccr
                                    ]
                                    drop_box = sfe_analyzer.auto_detect_droplet_candidate(warped, exclude_box=exclude_box_warped, coin_radius=ccr)
                                else:
                                    drop_box = sfe_analyzer.auto_detect_droplet_candidate(warped)
                            if drop_box is not None:
                                dprev = warped.copy()
                                # 자동 감지 모드에서도 마스크 실시간 오버레이
                                with st.spinner("Calculating droplet mask..."):
                                    try:
                                        sfe_analyzer.set_image(warped)
                                        d_mask, _ = sfe_analyzer.predict_mask(box=drop_box)
                                        d_mask_bin = sfe_analyzer.get_binary_mask(d_mask)
                                        
                                        d_overlay = dprev.copy()
                                        d_overlay[d_mask_bin > 0] = [255, 100, 100]
                                        cv2.addWeighted(d_overlay, 0.4, dprev, 0.6, 0, dprev)
                                        
                                        d_contours, _ = cv2.findContours(d_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        cv2.drawContours(dprev, d_contours, -1, (255, 50, 50), 3)
                                    except Exception as d_mask_err:
                                        pass
                                        
                                dx1, dy1, dx2, dy2 = map(int, drop_box)
                                cv2.rectangle(dprev, (dx1, dy1), (dx2, dy2), (255, 80, 80), 8)
                                st.image(dprev, caption="Detected Droplet Mask (Red) & Target Area (Green)", width="stretch")
                            else:
                                st.error("액적 자동 감지 실패. 상단의 '액적 영역 수동 지정' 체크박스를 켜고 마우스로 중심을 지정해 주세요." if lang == "ko"
                                         else "Droplet automatic detection failed. Please enable 'Manual Droplet Input' checkbox.")

                    is_ready = False
                    if manual_droplet and st.session_state[drop_pt_key] is not None:
                        is_ready = True
                    elif not manual_droplet and drop_box is not None:
                        is_ready = True

                    if is_ready:
                        if st.button(
                            "접촉각 분석 실행" if lang == "ko" else "Analyze Contact Angle",
                            key=f"btn_ca_{liq_key}", type="primary", width="stretch",
                        ):
                            start_time = time.time()
                            sfe_analyzer.set_image(warped)
                            if manual_droplet:
                                pt_coords = np.array([[dcx_input, dcy_input]])
                                pt_labels = np.array([1])
                                d_mask, _ = sfe_analyzer.predict_mask(point_coords=pt_coords, point_labels=pt_labels)
                            else:
                                d_mask, _ = sfe_analyzer.predict_mask(box=drop_box)

                            px_mm = DropletPhysics.calculate_pixels_per_mm(coin_info[2], coin_d)
                            d_mm = DropletPhysics.calculate_contact_diameter(d_mask, px_mm)
                            ca_val = DropletPhysics.calculate_contact_angle(vol_ul, d_mm)
                            
                            inference_time = time.time() - start_time

                            st.session_state["sfe_results"][liq_key] = {
                                "ca": ca_val, "d_mm": d_mm, "px_mm": px_mm,
                                "liquid": chem_name,
                                "time": inference_time,
                            }

                    # --- 결과 표시 ---
                    if liq_key in st.session_state["sfe_results"]:
                        res = st.session_state["sfe_results"][liq_key]
                        st.markdown("---")
                        st.markdown("#### 3. 측정 결과" if lang == "ko" else "#### 3. Measurement Result")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: _card(f"{res['px_mm']:.1f} px/mm", "Pixel Scale")
                        with c2: _card(f"{res['d_mm']:.3f} mm", "Contact Diameter")
                        with c3: _card(f"{res['ca']:.1f}\u00b0", "Contact Angle")
                        with c4: _card(f"{res.get('time', 0):.2f} s", "Inference Time" if lang == "en" else "추론 시간")
                        st.pyplot(_droplet_plot(res["ca"]))

                        # SFE 테이블에 추가
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button("SFE 테이블에 추가" if lang == "ko" else "Add to SFE table",
                                         key=f"add_{liq_key}", width="stretch"):
                                exists = any(m["Liquid"] == res["liquid"] for m in st.session_state["m_list"])
                                if not exists:
                                    st.session_state["m_list"].append({
                                        "Liquid": res["liquid"],
                                        "Angle": round(res["ca"], 2),
                                        "Volume": vol_ul,
                                        "Inf. Time (s)": round(res.get("time", 0), 2),
                                    })
                                    st.success("추가 완료" if lang == "ko" else "Added")
                                else:
                                    # 기존 항목 업데이트
                                    for m in st.session_state["m_list"]:
                                        if m["Liquid"] == res["liquid"]:
                                            m["Angle"] = round(res["ca"], 2)
                                            m["Volume"] = vol_ul
                                            m["Inf. Time (s)"] = round(res.get("time", 0), 2)
                                    st.info("기존 측정값이 갱신되었습니다." if lang == "ko" else "Existing measurement updated.")
                        with b2:
                            if st.button("테이블 초기화" if lang == "ko" else "Clear table",
                                         key=f"clr_{liq_key}", width="stretch"):
                                st.session_state["m_list"] = []
                                st.session_state["sfe_results"] = {}
                                st.session_state["sfe_calc_result"] = None
                                st.rerun()
                else:
                    st.error("Homography 보정 실패." if lang == "ko" else "Homography failed.")

            # 액체 영역 구분 구분선
            st.markdown("<hr style='border:1px dashed rgba(255,255,255,0.1); margin:40px 0;'>", unsafe_allow_html=True)

        # --- SFE 테이블 & OWRK 계산 ---
        if st.session_state["m_list"]:
            st.markdown("---")
            st.markdown("#### OWRK Surface Free Energy")
            st.dataframe(pd.DataFrame(st.session_state["m_list"]), width="stretch")

            if len(st.session_state["m_list"]) >= 2:
                calc_in = [{"liquid": m["Liquid"], "angle": m["Angle"]} for m in st.session_state["m_list"]]
                tot, gd, gp = DropletPhysics.calculate_owrk(calc_in)
                if tot is not None:
                    st.session_state["sfe_calc_result"] = {"total": tot, "disp": gd, "polar": gp}
                    c1, c2, c3 = st.columns(3)
                    with c1: _card(f"{tot:.2f} mN/m", "Total SFE")
                    with c2: _card(f"{gd:.2f} mN/m", "Dispersive")
                    with c3: _card(f"{gp:.2f} mN/m", "Polar")

# =========================================================================
# STEP 3 — Surface Finish (V-SAMS)
# =========================================================================
with st.expander("STEP 3.  " + T["tab2"], expanded=False):
    st.markdown(
        "동전 반사상 분석을 통해 표면 조도(Ra), 광택도(Gloss), 마감 종류를 판정합니다."
        if lang == "ko"
        else "Evaluates roughness (Ra), gloss (%), and industrial finish class from coin reflection analysis."
    )
    if f_finish is None:
        st.info(T["no_img"])
    else:
        bgr_f, rgb_f = _load_img(f_finish)
        
        # UI 렌더링 최적화: 고해상도 이미지를 그대로 st.image에 넘기면 브라우저 인코딩 병목(Hang)이 발생함
        # 화면 표시용(display) 해상도는 최대 800px로 축소하여 렌더링
        disp_rgb = rgb_f.copy()
        if max(disp_rgb.shape[:2]) > 800:
            scale_disp = 800.0 / max(disp_rgb.shape[:2])
            disp_rgb = cv2.resize(disp_rgb, (int(disp_rgb.shape[1] * scale_disp), int(disp_rgb.shape[0] * scale_disp)), interpolation=cv2.INTER_AREA)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(disp_rgb, caption="Input Image", width="stretch")
        with col_b:
            with st.spinner("Analyzing surface..."):
                ev = vsams_eval.analyze(rgb_f)
            if "error" not in ev:
                st.session_state["v_sams_result"] = ev
                overlay = vsams_eval.get_overlay_image(rgb_f.copy(), ev)
                
                # 오버레이 이미지(PIL)도 렌더링을 위해 리사이즈
                if max(overlay.size) > 800:
                    overlay.thumbnail((800, 800))
                st.image(overlay, caption="Analysis Overlay", width="stretch")
            else:
                st.error(ev["error"])

        if st.session_state["v_sams_result"] is not None:
            vr = st.session_state["v_sams_result"]
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1: _card(f"{vr['roughness']:.4f} μm", "조도 (Ra)" if lang == "ko" else "Roughness (Ra)")
            with c2: _card(f"{vr['gloss']:.1f} GU", "광택도 (Gloss)" if lang == "ko" else "Gloss Unit")
            with c3: _card(vr["predicted_label"], "마감 분류" if lang == "ko" else "Finish Class")

# =========================================================================
# STEP 4 — 3D Curvature (SG-TERRA)
# =========================================================================
with st.expander("STEP 4.  " + T["tab3"], expanded=False):
    st.markdown(
        "단안 깊이 추정(Depth-Anything-V2) 및 SAM 2 세그멘테이션으로 3D 지형을 복원하고 곡률을 연산합니다."
        if lang == "ko"
        else "Reconstructs 3D topography via Depth-Anything-V2 and SAM 2, computing Gaussian curvature."
    )

    if not do_3d:
        st.info(
            "상단 설정 영역에서 '3D 곡률 분석도 수행' 옵션을 활성화해 주세요."
            if lang == "ko"
            else "Enable '3D curvature analysis' in the upper settings."
        )
    elif f_3d is None:
        st.info(T["no_img"])
    else:
        bgr3, rgb3 = _load_img(f_3d)

        # --- 캘리브레이션 포인트 수집 ---
        st.markdown(
            "이미지에서 실제 길이를 아는 두 지점을 순서대로 클릭하세요."
            if lang == "ko"
            else "Click two points with a known physical distance on the image."
        )

        if HAS_IMG_COORDS:
            # 표시용 이미지에 기존 포인트 그리기
            disp = rgb3.copy()
            for pt in st.session_state["pts_007"]:
                cv2.circle(disp, (pt[0], pt[1]), 10, (0, 255, 0), -1)
            if len(st.session_state["pts_007"]) == 2:
                cv2.line(disp, tuple(st.session_state["pts_007"][0]),
                         tuple(st.session_state["pts_007"][1]), (0, 255, 0), 3)

            display_w3 = 340
            h_orig3, w_orig3 = rgb3.shape[:2]
            click = streamlit_image_coordinates(disp, width=display_w3, key="ic_3d")
            if click is not None:
                scale_factor3 = w_orig3 / display_w3
                clicked_pt = (
                    int(click["x"] * scale_factor3),
                    int(click["y"] * scale_factor3)
                )
                if clicked_pt != st.session_state["last_pt_007"]:
                    st.session_state["last_pt_007"] = clicked_pt
                    if len(st.session_state["pts_007"]) < 2:
                        st.session_state["pts_007"].append(clicked_pt)
                        st.rerun()
        else:
            st.image(rgb3, width="stretch")
            st.warning("streamlit-image-coordinates 미설치. 기본 중앙점을 사용합니다.")
            h3, w3 = rgb3.shape[:2]
            st.session_state["pts_007"] = [(w3 // 4, h3 // 2), (3 * w3 // 4, h3 // 2)]

        # 스케일 계산
        px2mm = 1.0
        sam_pts = None
        sam_lbls = None
        pts = st.session_state["pts_007"]

        if len(pts) == 1:
            st.info("첫 번째 포인트 지정 완료. 두 번째 포인트를 클릭하세요." if lang == "ko"
                     else "Point 1 set. Click second point.")
        elif len(pts) >= 2:
            d_px = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
            if d_px > 0:
                px2mm = ref_len / d_px
            st.success(f"Scale: {px2mm:.4f} mm/pixel")

            cx = (pts[0][0] + pts[1][0]) // 2
            cy = (pts[0][1] + pts[1][1]) // 2
            sam_pts = np.array([[cx, cy]])
            sam_lbls = np.array([1])

        if st.button("포인트 초기화" if lang == "ko" else "Reset Points",
                      key="rst_3d", width="stretch"):
            st.session_state["pts_007"] = []
            st.session_state["last_pt_007"] = None
            st.rerun()

        # --- 3D 분석 실행 ---
        if st.button("3D 분석 실행" if lang == "ko" else "Run 3D Analysis",
                      key="run_3d", type="primary", width="stretch"):
            if len(pts) < 2:
                st.error("캘리브레이션 포인트 2개를 먼저 지정해 주세요." if lang == "ko"
                          else "Set 2 calibration points first.")
            else:
                c_l, c_r = st.columns(2)
                with st.spinner("AI 파이프라인 가동 중..." if lang == "ko" else "Running AI pipeline..."):
                    # SAM 2 Segmentation
                    t0 = time.time()
                    mask_t = sam2_w.segment_target(rgb3, prompt_points=sam_pts, prompt_labels=sam_lbls)
                    dt_seg = (time.time() - t0) * 1000

                    # Depth Estimation
                    t0 = time.time()
                    dmap = depth_w.estimate_depth(rgb3, mask=mask_t)
                    dt_dep = (time.time() - t0) * 1000

                    # Curvature
                    curv_a.sigma = sigma_v
                    # 캘리브레이션으로 확보한 px2mm(픽셀당 mm) 스케일을 X, Y, Z축에 적용
                    g_curv = curv_a.calculate_gaussian_curvature(dmap, mask=mask_t, pixel_to_mm=px2mm, z_scale=px2mm)
                    cvals, ccoords = curv_a.find_critical_points(g_curv, mask=mask_t, top_k=1)
                    k_max = cvals[0]
                    # K의 역수를 제곱근하여 R 연산 (물리적 단위가 이미 반영되어 있으므로, px2mm를 다시 곱할 필요가 없음)
                    r_mm = 1.0 / np.sqrt(np.abs(k_max)) if k_max != 0 else 0
                    r_mm = round(r_mm, 2)

                    st.session_state["curv_result"] = {
                        "max_k": k_max, "min_r_mm": r_mm,
                        "depth_map": dmap, "mask": mask_t,
                        "ccoord": ccoords[0],
                    }

                with c_l:
                    st.markdown("##### Segmentation")
                    ov = np.zeros_like(rgb3)
                    ov[mask_t] = [0, 255, 0]
                    blend = cv2.addWeighted(rgb3, 0.7, ov, 0.3, 0)
                    st.image(blend, caption=f"SAM 2 Mask ({dt_seg:.0f} ms)", width="stretch")

                with c_r:
                    st.markdown("##### Depth Map")
                    d_vis = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    d_col = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
                    st.image(cv2.cvtColor(d_col, cv2.COLOR_BGR2RGB),
                             caption=f"Depth ({dt_dep:.0f} ms)", width="stretch")

                # Plotly 3D Surface
                st.markdown("---")
                st.markdown("##### 3D Topographic Surface")
                pct = 20
                w_s = int(dmap.shape[1] * pct / 100)
                h_s = int(dmap.shape[0] * pct / 100)
                ds = cv2.resize(dmap, (w_s, h_s), interpolation=cv2.INTER_AREA)

                iy_s = min(max(int(ccoords[0][0] * pct / 100), 0), h_s - 1)
                ix_s = min(max(int(ccoords[0][1] * pct / 100), 0), w_s - 1)

                fig3 = go.Figure(data=[
                    go.Surface(z=ds, colorscale="Inferno",
                               contours={"z": {"show": True,
                                               "size": (np.max(ds) - np.min(ds)) / 15,
                                               "color": "white"}}),
                    go.Scatter3d(x=[ix_s], y=[iy_s], z=[ds[iy_s, ix_s] + 0.04],
                                 mode="markers",
                                 marker=dict(size=9, color="cyan", symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 name="Max K"),
                ])
                fig3.update_layout(
                    height=550, margin=dict(l=0, r=0, b=0, t=10),
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False),
                        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.35),
                    ),
                )
                st.plotly_chart(fig3, width="stretch")

        # 저장된 결과 표시
        if st.session_state["curv_result"] is not None:
            cr = st.session_state["curv_result"]
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1: _card(f"{cr['max_k']:.5f}", "Max Gaussian Curvature (K)")
            with c2: _card(f"R \u2248 {cr['min_r_mm']} mm", "Min Curvature Radius (R)")

# =========================================================================
# STEP 5 — Consolidated Report
# =========================================================================
with st.expander("STEP 5.  " + T["tab4"], expanded=False):
    sfe_c = st.session_state.get("sfe_calc_result")
    vs_r  = st.session_state.get("v_sams_result")
    cv_r  = st.session_state.get("curv_result")

    if sfe_c is None and vs_r is None and cv_r is None:
        st.info(T["no_data"])
    else:
        rows = {}
        total_inf_time = 0.0
        
        # SFE 결과에서 추론 시간 합산
        m_list = st.session_state.get("m_list", [])
        if m_list:
            for m in m_list:
                total_inf_time += float(m.get("Inf. Time (s)", 0.0))

        if sfe_c:
            rows["Total SFE (mN/m)"] = round(sfe_c["total"], 3)
            rows["Dispersive (mN/m)"] = round(sfe_c["disp"], 3)
            rows["Polar (mN/m)"] = round(sfe_c["polar"], 3)
        if vs_r:
            rows["Roughness (Ra)"] = round(vs_r["roughness"], 4)
            rows["Gloss (%)"] = round(vs_r["gloss"], 2)
            rows["Metal Finish"] = vs_r["predicted_label"]
        if cv_r:
            rows["Max Gaussian K"] = round(cv_r["max_k"], 5)
            rows["Min Radius R (mm)"] = cv_r["min_r_mm"]
            
        if total_inf_time > 0:
            rows["Total AI Inference Time (s)"] = round(total_inf_time, 3)

        df = pd.DataFrame(list(rows.items()), columns=["Parameter", "Value"])
        st.dataframe(df, width="stretch", hide_index=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "JSON 다운로드" if lang == "ko" else "Download JSON",
                data=pd.Series(rows).to_json(indent=4, force_ascii=False),
                file_name="surface_report.json", mime="application/json",
                width="stretch",
            )
        with c2:
            st.download_button(
                "CSV 다운로드" if lang == "ko" else "Download CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="surface_report.csv", mime="text/csv",
                width="stretch",
            )
