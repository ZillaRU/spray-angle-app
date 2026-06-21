#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
喷雾锥角测量系统 - 单文件精简版
Spray Cone Angle Measurement - Single File Implementation

核心策略 (v3):
1. 聚焦喷口近区 (3%-30%) — 该区域边缘最锐利, 接近直线
2. 多方法融合: 近区Hough + 梯度剖面 + 宽度回归 + 多带梯度 + 纹理法
3. 纹理法(局部方差)解决极低对比度场景
4. 投影误差提示

使用方法:
    python spray_angle.py <image_path> [--nozzle x,y] [--background bg.jpg]
    python spray_angle.py <image_path> --bg background.jpg   # 有参考底图时推荐
    python spray_angle.py --batch <dir>

注意: --nozzle 坐标为原始图像坐标系 (缩放前)
"""

import cv2
import numpy as np
from scipy import ndimage
import os, sys, json, argparse
from datetime import datetime
from typing import Tuple, Optional, Dict, List


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "max_image_dimension": 1200,
    "near_start_ratio": 0.03,
    "near_end_ratio": 0.30,
    "boundary_angle_range": 85.0,
    "clahe_clip_limit": 3.0,
    "consistency_threshold": 20.0,
    "weights": {
        "near_hough": 0.20,
        "gradient_profile": 0.25,
        "width_regression": 0.15,
        "near_gradient_multi": 0.20,
        "texture": 0.20,
    },
}


# ═══════════════════════════════════════════════════════════════
# 预处理
# ═══════════════════════════════════════════════════════════════
def preprocess(image: np.ndarray) -> Dict:
    """图像预处理: 缩放、灰度化、CLAHE增强、背景减除"""
    h, w = image.shape[:2]
    scale = 1.0
    max_dim = CONFIG["max_image_dimension"]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=CONFIG["clahe_clip_limit"], tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    bg = cv2.GaussianBlur(denoised, (51, 51), 0)
    bg_sub = cv2.normalize(cv2.absdiff(denoised, bg), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {"original": image, "gray": gray, "enhanced": enhanced,
            "bg_subtracted": bg_sub, "scale": scale}


# ═══════════════════════════════════════════════════════════════
# 喷口检测
# ═══════════════════════════════════════════════════════════════
def detect_nozzle(gray: np.ndarray, enhanced: np.ndarray) -> Tuple[Tuple[int, int], float]:
    """
    自动检测喷口位置 — 通过测量质量反选
    
    核心原理:
      正确的喷口位置 → 多种角度测量方法结果一致 + 宽度回归R²高
      对候选点网格进行"试测量", 选得分最高的点作为喷口
    
    返回: ((x, y), confidence_score)
    """
    h, w = gray.shape
    
    # === 粗搜索: 9×12网格 ===
    x_cands = np.linspace(w*0.25, w*0.75, 9).astype(int)
    y_cands = np.linspace(h*0.05, h*0.60, 12).astype(int)
    
    best_score = -1
    best_nozzle = (w//2, h//4)
    
    for nx in x_cands:
        for ny in y_cands:
            score = _eval_nozzle_candidate(enhanced, (int(nx), int(ny)))
            if score > best_score:
                best_score = score
                best_nozzle = (int(nx), int(ny))
    
    # === 精细搜索: 最佳点附近 ===
    bx, by = best_nozzle
    step = max(5, min(15, int(h*0.015)))
    for nx in range(max(10, bx-60), min(w-10, bx+61), step):
        for ny in range(max(5, by-60), min(h-10, by+61), step):
            score = _eval_nozzle_candidate(enhanced, (nx, ny))
            if score > best_score:
                best_score = score
                best_nozzle = (nx, ny)
    
    return best_nozzle, best_score


def _eval_nozzle_candidate(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> float:
    """评估候选喷口: 用测量一致性打分"""
    r_w = measure_width_regression(enhanced, nozzle)
    r_m = measure_multi_band_gradient(enhanced, nozzle)
    
    angles, confs = [], []
    for r in [r_w, r_m]:
        a = r.get('angle')
        if a and 15 < a < 155:
            angles.append(a)
            confs.append(r.get('confidence', 0))
    
    if len(angles) < 2:
        return 0.0
    
    std = np.std(angles)
    consistency = max(0, 1 - std/20)
    mean_conf = np.mean(confs)
    return consistency * mean_conf


# ═══════════════════════════════════════════════════════════════
# 辅助: RANSAC 边界线拟合
# ═══════════════════════════════════════════════════════════════
def _fit_boundary_line(points: np.ndarray, nozzle: Tuple[int, int]):
    """RANSAC-like鲁棒直线拟合, 返回 (angle_from_vertical, line_coords)"""
    if len(points) < 3:
        return None, None
    nx, ny = nozzle
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    try:
        dy = y - ny
        dx = x - nx
        if np.std(dy) < 1e-6:
            return None, None

        best_slope, best_inliers = None, 0
        for _ in range(min(50, len(points) * 5)):
            idx = np.random.choice(len(points), 2, replace=False)
            ddy = dy[idx[1]] - dy[idx[0]]
            if abs(ddy) < 1e-6:
                continue
            slope = (dx[idx[1]] - dx[idx[0]]) / ddy
            predicted_dx = slope * dy
            inliers = np.sum(np.abs(dx - predicted_dx) < 8)
            if inliers > best_inliers:
                best_inliers, best_slope = inliers, slope

        if best_slope is None:
            best_slope = np.polyfit(dy, dx, 1)[0]

        angle = np.degrees(np.arctan(best_slope))
        y_min, y_max = int(np.min(y)), int(np.max(y))
        line = (int(nx + best_slope*(y_min-ny)), y_min,
                int(nx + best_slope*(y_max-ny)), y_max)
        return angle, line
    except:
        return None, None


# ═══════════════════════════════════════════════════════════════
# 角度测量方法
# ═══════════════════════════════════════════════════════════════
def measure_near_hough(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> Dict:
    """方法1: 近区Hough直线检测"""
    h, w = enhanced.shape
    nx, ny = nozzle
    spray_len = h - ny
    roi_s = ny + int(spray_len * CONFIG["near_start_ratio"])
    roi_e = min(ny + int(spray_len * CONFIG["near_end_ratio"]), h - 5)
    if roi_e - roi_s < 30:
        return {"angle": None, "confidence": 0, "method": "near_hough"}

    roi_gray = enhanced[roi_s:roi_e, :]
    med = np.median(roi_gray)
    roi_edges = cv2.Canny(roi_gray, int(max(10, 0.5*med)), int(min(255, 1.5*med)))
    roi_edges = cv2.dilate(roi_edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)

    min_len = max(15, (roi_e - roi_s) // 4)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, max(20, min_len//2),
                            minLineLength=min_len, maxLineGap=8)
    if lines is None or len(lines) < 2:
        return {"angle": None, "confidence": 0, "method": "near_hough"}

    processed = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1 += roi_s; y2 += roi_s
        if y1 > y2: x1,y1,x2,y2 = x2,y2,x1,y1
        if y2 - y1 < 10: continue
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        slope_xy = (x2-x1)/(y2-y1+1e-10)
        x_at_nz = x1 + slope_xy*(ny-y1)
        conv_dist = abs(x_at_nz - nx)
        if conv_dist > w*0.15: continue
        angle = np.degrees(np.arctan2(x2-x1, y2-y1))
        if abs(angle) > 75: continue
        side = "right" if (x1+x2)/2 > nx else "left"
        processed.append({"angle": angle, "length": length, "side": side, "conv": conv_dist})

    if len(processed) < 2:
        return {"angle": None, "confidence": 0, "method": "near_hough"}

    left = [l for l in processed if l["side"] == "left"]
    right = [l for l in processed if l["side"] == "right"]
    if not left or not right:
        processed.sort(key=lambda l: l["angle"])
        left, right = [processed[0]], [processed[-1]]

    def wavg(lst):
        w = np.array([l["length"]/(l["conv"]+10) for l in lst])
        a = np.array([l["angle"] for l in lst])
        return float(np.average(a, weights=w)) if w.sum() > 0 else float(np.mean(a))

    la, ra = wavg(left), wavg(right)
    cone = abs(ra - la)
    n = len(left) + len(right)
    best_left = max(left, key=lambda l: l["length"])
    best_right = max(right, key=lambda l: l["length"])
    avg_len = (best_left["length"] + best_right["length"]) / 2
    roi_h = roi_e - roi_s
    conf = min(1.0, (n/6) * (avg_len / roi_h))
    if cone < 10 or cone > 160: conf *= 0.5
    return {"angle": cone, "confidence": conf, "method": "near_hough",
            "details": f"L={la:.1f}° R={ra:.1f}° lines={n}"}


def measure_gradient_profile(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> Dict:
    """方法2: 近区梯度方向剖面 (341角度采样)"""
    h, w = enhanced.shape
    nx, ny = nozzle
    grad_mag = np.sqrt(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)**2 +
    
