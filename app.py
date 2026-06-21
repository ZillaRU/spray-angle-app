#!/usr/bin/env python3
"""
喷雾锥角测量系统 - Streamlit 前端
Spray Cone Angle Measurement - Web UI

功能:
1. 机位标定: 上传无喷雾背景图 + 点击标定喷口位置
2. 角度测量: 上传喷雾图 → 自动作差 + 测角 + 可视化
3. 结果展示: 在图上画出识别的锥角

运行: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# 导入测量核心
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spray_angle import (
    preprocess, detect_nozzle, measure_near_hough, measure_gradient_profile,
    measure_width_regression, measure_multi_band_gradient, measure_texture,
    fuse_results, _subtract_background, projection_warning, CONFIG
)

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")
os.makedirs(CALIBRATION_DIR, exist_ok=True)

st.set_page_config(page_title="喷雾锥角测量", page_icon="🔬", layout="wide")


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def load_calibrations() -> dict:
    """加载所有已保存的机位标定"""
    cals = {}
    cal_file = os.path.join(CALIBRATION_DIR, "calibrations.json")
    if os.path.exists(cal_file):
        with open(cal_file, "r") as f:
            cals = json.load(f)
    return cals


def save_calibration(name: str, nozzle_pos: tuple, bg_path: str, original_size: tuple):
    """保存机位标定"""
    cals = load_calibrations()
    cals[name] = {
        "nozzle_pos": list(nozzle_pos),
        "background_path": bg_path,
        "original_size": list(original_size),
        "created": str(datetime.now()),
    }
    cal_file = os.path.join(CALIBRATION_DIR, "calibrations.json")
    with open(cal_file, "w") as f:
        json.dump(cals, f, indent=2, ensure_ascii=False)


def draw_cone_angle(image: np.ndarray, nozzle: tuple, cone_angle: float,
                    details: dict) -> np.ndarray:
    """在图像上绘制锥角可视化"""
    vis = image.copy()
    h, w = vis.shape[:2]
    nx, ny = nozzle

    # 画喷口标记
    cv2.circle(vis, (nx, ny), 8, (0, 0, 255), -1)
    cv2.circle(vis, (nx, ny), 12, (0, 0, 255), 2)

    # 画锥角边界线
    half_angle = np.radians(cone_angle / 2)
    line_length = min(h - ny - 20, w // 2)

    # 左边界
    lx = int(nx - line_length * np.sin(half_angle))
    ly = int(ny + line_length * np.cos(half_angle))
    cv2.line(vis, (nx, ny), (lx, ly), (0, 255, 0), 3)

    # 右边界
    rx = int(nx + line_length * np.sin(half_angle))
    ry = int(ny + line_length * np.cos(half_angle))
    cv2.line(vis, (nx, ny), (rx, ry), (0, 255, 0), 3)

    # 中轴线 (虚线效果)
    cx, cy = nx, ny + line_length
    for i in range(0, line_length, 15):
        y1 = ny + i
        y2 = min(ny + i + 8, cy)
        cv2.line(vis, (nx, y1), (nx, y2), (255, 255, 0), 1)

    # 画角度弧线
    arc_r = min(80, line_length // 3)
    start_angle = int(90 - cone_angle / 2)
    end_angle = int(90 + cone_angle / 2)
    cv2.ellipse(vis, (nx, ny), (arc_r, arc_r), -90, start_angle, end_angle, (0, 200, 255), 2)

    # 标注文字
    text = f"{cone_angle:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
    tx = nx - text_size[0] // 2
    ty = ny + arc_r + 35
    cv2.putText(vis, text, (tx, ty), font, 1.2, (0, 200, 255), 2, cv2.LINE_AA)

    # 置信度
    conf = details.get("confidence", 0)
    conf_text = f"Conf: {conf:.0%}"
    cv2.putText(vis, conf_text, (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def measure_with_visualization(spray_img: np.ndarray, bg_img: np.ndarray,
                               nozzle_orig: tuple, orig_size: tuple):
    """完整测量流程 (含分辨率适配)"""
    h_spray, w_spray = spray_img.shape[:2]
    h_orig, w_orig = orig_size

    # 分辨率适配: 将喷口坐标从标定分辨率映射到当前图分辨率
    scale_x = w_spray / w_orig
    scale_y = h_spray / h_orig
    nozzle_adapted = (int(nozzle_orig[0] * scale_x), int(nozzle_orig[1] * scale_y))

    # 背景作差
    if bg_img is not None:
        bh, bw = bg_img.shape[:2]
        if (bh, bw) != (h_spray, w_spray):
            bg_img = cv2.resize(bg_img, (w_spray, h_spray), interpolation=cv2.INTER_AREA)
        img_f = spray_img.astype(np.float32)
        bg_f = bg_img.astype(np.float32)
        diff = np.abs(img_f - bg_f)
        diff_gray = np.max(diff, axis=2) if len(diff.shape) == 3 else diff
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_norm = cv2.GaussianBlur(diff_norm, (3, 3), 0)
        process_img = cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR)
    else:
        process_img = spray_img.copy()

    # 预处理
    pp = preprocess(process_img)
    gray, enhanced, scale = pp["gray"], pp["enhanced"], pp["scale"]

    # 喷口映射到缩放后坐标
    nozzle = (int(nozzle_adapted[0] * scale), int(nozzle_adapted[1] * scale))

    # 多方法测量
    results = [
        measure_near_hough(enhanced, nozzle),
        measure_gradient_profile(enhanced, nozzle),
        measure_width_regression(enhanced, nozzle),
        measure_multi_band_gradient(enhanced, nozzle),
        measure_texture(enhanced, nozzle),
    ]

    # 融合
    fusion = fuse_results(results)
    final_angle = fusion.get("final_angle")
    confidence = fusion.get("confidence", 0)

    # 可视化 (在原始喷雾图上画)
    vis_img = None
    if final_angle:
        vis_img = draw_cone_angle(spray_img, nozzle_adapted, final_angle, fusion)

    return {
        "cone_angle": final_angle,
        "confidence": confidence,
        "nozzle_used": nozzle_adapted,
        "individual": {r["method"]: r.get("angle") for r in results},
        "visualization": vis_img,
        "fusion": fusion,
    }


# ═══════════════════════════════════════════════════════════════
# Streamlit UI
# ═══════════════════════════════════════════════════════════════
def main():
    st.title("🔬 喷雾锥角测量系统")
    st.caption("基于计算机视觉的喷雾锥角自动测量 · 无需训练数据 · 可解释算法")

    tab1, tab2 = st.tabs(["📐 机位标定", "🎯 角度测量"])

    # ─── Tab 1: 机位标定 ───
    with tab1:
        st.header("机位标定")
        st.info("上传该机位的**无喷雾背景图**，并点击标定喷口位置。标定后可重复使用。")

        col1, col2 = st.columns([2, 1])

        with col2:
            cal_name = st.text_input("机位名称", value="默认机位",
                                     help="用于区分不同机位的标定")

            # 已有标定
            cals = load_calibrations()
            if cals:
                st.subheader("已保存的标定")
                for name, info in cals.items():
                    nz = info["nozzle_pos"]
                    st.success(f"**{name}** — 喷口({nz[0]}, {nz[1]})")

        with col1:
            bg_file = st.file_uploader("上传无喷雾背景图", type=["jpg", "jpeg", "png", "bmp"],
                                       key="bg_upload")

            if bg_file:
                file_bytes = np.frombuffer(bg_file.read(), dtype=np.uint8)
                bg_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w = bg_image.shape[:2]

                st.write(f"图像尺寸: {w}×{h}")

                # 显示图像供用户点击标定
                bg_rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

                # 用slider指定喷口 (streamlit没有原生点击坐标)
                st.subheader("标定喷口位置")
                st.caption("拖动滑块指定喷口的 X、Y 坐标")

                nozzle_x = st.slider("喷口 X", 0, w-1, w//2, key="nz_x")
                nozzle_y = st.slider("喷口 Y", 0, h-1, h//3, key="nz_y")

                # 在图上画标记
                vis = bg_rgb.copy()
                cv2.circle(vis, (nozzle_x, nozzle_y), max(8, w//100), (255, 0, 0), -1)
                cv2.circle(vis, (nozzle_x, nozzle_y), max(12, w//70), (255, 0, 0), 3)
                # 十字线
                cross_len = max(20, w//30)
                cv2.line(vis, (nozzle_x-cross_len, nozzle_y), (nozzle_x+cross_len, nozzle_y), (255, 0, 0), 2)
                cv2.line(vis, (nozzle_x, nozzle_y-cross_len), (nozzle_x, nozzle_y+cross_len), (255, 0, 0), 2)

                st.image(vis, caption=f"喷口标定位置: ({nozzle_x}, {nozzle_y})", use_container_width=True)

                # 保存按钮
                if st.button("💾 保存标定", type="primary"):
                    # 保存背景图
                    bg_save_path = os.path.join(CALIBRATION_DIR, f"{cal_name}_bg.jpg")
                    cv2.imwrite(bg_save_path, bg_image)
                    # 保存标定信息
                    save_calibration(cal_name, (nozzle_x, nozzle_y), bg_save_path, (h, w))
                    st.success(f"✅ 标定已保存: **{cal_name}** — 喷口({nozzle_x}, {nozzle_y}), 分辨率{w}×{h}")
                    st.rerun()

    # ─── Tab 2: 角度测量 ───
    with tab2:
        st.header("角度测量")

        # 选择机位
        cals = load_calibrations()
        if not cals:
            st.warning("⚠️ 请先在「机位标定」页面完成至少一个机位的标定")
            return

        selected_cal = st.selectbox("选择机位", list(cals.keys()))
        cal_info = cals[selected_cal]
        nozzle_pos = tuple(cal_info["nozzle_pos"])
        bg_path = cal_info["background_path"]
        orig_size = tuple(cal_info["original_size"])  # (h, w)

        st.caption(f"机位: {selected_cal} | 喷口: {nozzle_pos} | 标定分辨率: {orig_size[1]}×{orig_size[0]}")

        # 上传喷雾图
        spray_files = st.file_uploader("上传喷雾图像 (可多张)",
                                       type=["jpg", "jpeg", "png", "bmp"],
                                       accept_multiple_files=True,
                                       key="spray_upload")

        if spray_files:
            # 加载背景图
            bg_img = cv2.imread(bg_path) if os.path.exists(bg_path) else None

            for spray_file in spray_files:
                st.divider()
                file_bytes = np.frombuffer(spray_file.read(), dtype=np.uint8)
                spray_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if spray_image is None:
                    st.error(f"无法读取: {spray_file.name}")
                    continue

                h_s, w_s = spray_image.shape[:2]
                st.subheader(f"📷 {spray_file.name} ({w_s}×{h_s})")

                # 测量
                with st.spinner("测量中..."):
                    result = measure_with_visualization(
                        spray_image, bg_img, nozzle_pos, orig_size
                    )

                # 显示结果
                if result["cone_angle"]:
                    col_a, col_b = st.columns([3, 1])

                    with col_a:
                        vis_rgb = cv2.cvtColor(result["visualization"], cv2.COLOR_BGR2RGB)
                        st.image(vis_rgb, caption=f"锥角: {result['cone_angle']:.1f}°",
                                use_container_width=True)

                    with col_b:
                        st.metric("锥角", f"{result['cone_angle']:.1f}°")
                        st.metric("置信度", f"{result['confidence']:.0%}")
                        st.caption(f"喷口: {result['nozzle_used']}")

                        # 各方法详情
                        st.write("**各方法结果:**")
                        for method, angle in result["individual"].items():
                            if angle:
                                st.write(f"- {method}: {angle:.1f}°")
                            else:
                                st.write(f"- {method}: N/A")

                        # 投影误差
                        with st.expander("投影误差提示"):
                            st.code(projection_warning(result["cone_angle"]))
                else:
                    st.error("❌ 测量失败，请检查图像质量或标定是否正确")


if __name__ == "__main__":
    main()

