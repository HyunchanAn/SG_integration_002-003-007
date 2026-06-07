import streamlit as st
import cv2
import numpy as np
import os
import json
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")
st.title("디버깅 툴: 진짜 동전과 액적 찾기")
st.markdown("이미지마다 아래 후보들 중에서 **진짜 동전**과 **진짜 액적**을 하나씩 골라주세요!")

IMAGES = [
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\2B\2B_glycerol.jpg',
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\2B\2B_water.jpg',
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\BA\BA_glycerol.jpg',
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\BA\BA_water.jpg',
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\HL\HL_glycerol.jpg',
    r'c:\Users\chema\Github\SG_integration_002-003-007\test_image_260521\HL\HL_water.jpg'
]

OUT_JSON = r'c:\Users\chema\Github\SG_integration_002-003-007\ground_truth.json'

@st.cache_data
def get_candidates(img_path):
    img = cv2.imread(img_path)
    if img is None: return [], []
    
    orig_h, orig_w = img.shape[:2]
    max_dim = 800.0
    scale = max_dim / float(max(orig_h, orig_w))
    work_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    h, w = work_img.shape[:2]
    
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_pre = clahe.apply(gray)
    gray_pre = cv2.medianBlur(gray_pre, 7)
    
    circles_coin = cv2.HoughCircles(
        gray_pre, cv2.HOUGH_GRADIENT, dp=1.1, minDist=w//10,
        param1=50, param2=25, minRadius=int(h*0.05), maxRadius=int(h*0.25)
    )
    coin_cands = []
    if circles_coin is not None:
        for c in np.round(circles_coin[0, :]).astype("int"):
            coin_cands.append(c)
            
    b, g, r_ch = cv2.split(work_img)
    gray2 = cv2.addWeighted(b, 0.7, g, 0.3, 0)
    gray_blur = cv2.GaussianBlur(gray2, (7, 7), 0)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe2.apply(gray_blur.astype(np.uint8))
    
    circles_drop = cv2.HoughCircles(
        gray_clahe, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=15, minRadius=int(h*0.02), maxRadius=int(h*0.1)
    )
    drop_cands = []
    if circles_drop is not None:
        for c in np.round(circles_drop[0, :]).astype("int"):
            drop_cands.append(c)
            
    return coin_cands[:10], drop_cands[:10], img, scale

def crop_candidate(img, c, scale):
    orig_h, orig_w = img.shape[:2]
    inv_scale = 1.0 / scale
    x, y, r = c
    cx = int(x * inv_scale)
    cy = int(y * inv_scale)
    cr = int(r * inv_scale)
    
    pad = int(cr * 0.5)
    x1 = max(0, cx - cr - pad)
    y1 = max(0, cy - cr - pad)
    x2 = min(orig_w, cx + cr + pad)
    y2 = min(orig_h, cy + cr + pad)
    
    crop = img[y1:y2, x1:x2].copy()
    
    context_img = img.copy()
    cv2.circle(context_img, (cx, cy), cr, (0, 255, 0), 10)
    context_img = cv2.resize(context_img, (0,0), fx=0.15, fy=0.15)
    
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cv2.cvtColor(context_img, cv2.COLOR_BGR2RGB), (cx, cy, cr)

if 'results' not in st.session_state:
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, 'r') as f:
            st.session_state.results = json.load(f)
    else:
        st.session_state.results = {}
        
if 'manual_mode' not in st.session_state:
    st.session_state.manual_mode = {}

for img_path in IMAGES:
    filename = os.path.basename(img_path)
    st.markdown(f"### {filename}")
    
    if filename not in st.session_state.results:
        st.session_state.results[filename] = {"coin": None, "droplet": None}
    if filename not in st.session_state.manual_mode:
        st.session_state.manual_mode[filename] = {"coin": False, "droplet": False}
        
    coin_cands, drop_cands, img, scale = get_candidates(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    disp_max = 700.0
    disp_scale = disp_max / float(max(orig_h, orig_w))
    disp_w = int(orig_w * disp_scale)
    disp_h = int(orig_h * disp_scale)
    disp_img = cv2.resize(img_rgb, (disp_w, disp_h))
    
    # --- Coin Selection ---
    st.markdown("**1. 진짜 동전을 골라주세요**")
    if st.session_state.results[filename]["coin"] is not None:
        cx, cy, cr = st.session_state.results[filename]['coin']
        st.success("동전 선택 완료!")
        
        preview = img_rgb.copy()
        cv2.circle(preview, (cx, cy), cr, (0, 255, 0), 15)
        pad = int(cr * 1.5)
        x1, y1 = max(0, cx - pad), max(0, cy - pad)
        x2, y2 = min(orig_w, cx + pad), min(orig_h, cy + pad)
        st.image(preview[y1:y2, x1:x2], caption="선택된 동전 확인")
        
        if st.button("동전 다시 선택하기", key=f"reset_c_{filename}"):
            st.session_state.results[filename]["coin"] = None
            st.session_state.manual_mode[filename]["coin"] = False
            st.rerun()
            
    elif st.session_state.manual_mode[filename]["coin"]:
        st.info("아래 이미지에서 진짜 동전의 **중심**을 클릭해주세요. (자동으로 크기가 조절되어 보입니다)")
        value = streamlit_image_coordinates(disp_img, key=f"img_c_{filename}")
        if value is not None:
            cx = int(value['x'] / disp_scale)
            cy = int(value['y'] / disp_scale)
            cr = int(orig_w * 0.08) # 대략적인 반지름
            st.session_state.results[filename]["coin"] = [cx, cy, cr]
            st.rerun()
    else:
        if st.button("🚨 10개 중에 정답이 없습니다 (동전 수동 클릭)", key=f"coin_none_{filename}"):
            st.session_state.manual_mode[filename]["coin"] = True
            st.rerun()
            
        cols = st.columns(5)
        for i, c in enumerate(coin_cands):
            crop_rgb, ctx_rgb, orig_c = crop_candidate(img, c, scale)
            with cols[i % 5]:
                st.image(crop_rgb, caption=f"후보 {i+1} (확대)", use_container_width=True)
                st.image(ctx_rgb, caption=f"전체 위치", use_container_width=True)
                if st.button(f"이게 동전입니다", key=f"coin_{filename}_{i}"):
                    st.session_state.results[filename]["coin"] = orig_c
                    st.rerun()
                    
    # --- Droplet Selection ---
    st.markdown("**2. 진짜 액적을 골라주세요**")
    if st.session_state.results[filename]["droplet"] is not None:
        dx, dy, dr = st.session_state.results[filename]['droplet']
        st.success("액적 선택 완료!")
        
        preview = img_rgb.copy()
        cv2.circle(preview, (dx, dy), dr, (255, 0, 0), 15)
        pad = int(dr * 2.0)
        x1, y1 = max(0, dx - pad), max(0, dy - pad)
        x2, y2 = min(orig_w, dx + pad), min(orig_h, dy + pad)
        st.image(preview[y1:y2, x1:x2], caption="선택된 액적 확인")
        
        if st.button("액적 다시 선택하기", key=f"reset_d_{filename}"):
            st.session_state.results[filename]["droplet"] = None
            st.session_state.manual_mode[filename]["droplet"] = False
            st.rerun()
            
    elif st.session_state.manual_mode[filename]["droplet"]:
        st.info("아래 이미지에서 진짜 액적의 **중심**을 클릭해주세요. (자동으로 크기가 조절되어 보입니다)")
        value = streamlit_image_coordinates(disp_img, key=f"img_d_{filename}")
        if value is not None:
            dx = int(value['x'] / disp_scale)
            dy = int(value['y'] / disp_scale)
            dr = int(orig_w * 0.05) # 대략적인 반지름
            st.session_state.results[filename]["droplet"] = [dx, dy, dr]
            st.rerun()
    else:
        if st.button("🚨 10개 중에 정답이 없습니다 (액적 수동 클릭)", key=f"drop_none_{filename}"):
            st.session_state.manual_mode[filename]["droplet"] = True
            st.rerun()
            
        cols = st.columns(5)
        for i, c in enumerate(drop_cands):
            crop_rgb, ctx_rgb, orig_c = crop_candidate(img, c, scale)
            with cols[i % 5]:
                st.image(crop_rgb, caption=f"후보 {i+1} (확대)", use_container_width=True)
                st.image(ctx_rgb, caption=f"전체 위치", use_container_width=True)
                if st.button(f"이게 액적입니다", key=f"drop_{filename}_{i}"):
                    st.session_state.results[filename]["droplet"] = orig_c
                    st.rerun()
    st.divider()

if st.button("모든 선택 저장하기", type="primary"):
    with open(OUT_JSON, 'w') as f:
        json.dump(st.session_state.results, f, indent=4)
    st.success(f"성공적으로 {OUT_JSON} 에 저장되었습니다! 채팅창으로 돌아가서 AI에게 '저장 완료했어'라고 말씀해 주세요.")
