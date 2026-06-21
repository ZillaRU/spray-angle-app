#!/usr/bin/env python3
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
                       cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)**2)

    ar = CONFIG["boundary_angle_range"]
    num_a = 341
    angles = np.linspace(-ar, ar, num_a)
    max_r = min(h-ny-10, w//2) * 0.85
    radii = np.linspace(max(10, max_r*CONFIG["near_start_ratio"]),
                        max_r*CONFIG["near_end_ratio"], 30)
    scores = np.zeros(num_a)

    for i, a in enumerate(angles):
        rad = np.radians(90 - a)
        dx, dy = np.cos(rad), np.sin(rad)
        t, c = 0.0, 0
        for r in radii:
            x, y = int(nx+r*dx), int(ny+r*dy)
            if 0 <= x < w and 0 <= y < h:
                t += grad_mag[y, x]; c += 1
        if c > 0: scores[i] = t/c

    scores = ndimage.uniform_filter1d(scores, 5)
    mid = num_a // 2
    ce = int(5.0 / ar * mid)
    ls, rs = scores[:mid-ce], scores[mid+ce:]
    if len(ls) == 0 or len(rs) == 0:
        return {"angle": None, "confidence": 0, "method": "gradient_profile"}

    la = angles[np.argmax(ls)]
    ra = angles[np.argmax(rs) + mid + ce]
    cone = ra - la
    mean_s = np.mean(scores)
    snr = min(ls.max(), rs.max()) / (mean_s + 1e-10)
    return {"angle": cone, "confidence": min(1.0, snr/2.5), "method": "gradient_profile",
            "details": f"L={la:.1f}° R={ra:.1f}° SNR={snr:.1f}"}


def measure_width_regression(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> Dict:
    """方法3: 近区宽度回归 (含RANSAC边界线)"""
    h, w = enhanced.shape
    nx, ny = nozzle
    spray_len = h - ny
    s_start = ny + int(spray_len * CONFIG["near_start_ratio"])
    s_end = min(ny + int(spray_len * CONFIG["near_end_ratio"]), h - 5)
    positions = np.linspace(s_start, s_end, 20).astype(int)

    distances, widths, left_xs, right_xs = [], [], [], []
    for sy in positions:
        if sy >= h: continue
        profile = enhanced[sy, :].astype(np.float64)
        margin = max(20, w//10)
        bg = (np.mean(profile[:margin]) + np.mean(profile[w-margin:])) / 2
        center = np.mean(profile[max(0,nx-15):min(w,nx+15)])
        contrast = abs(center - bg)
        if contrast < 3: continue
        bright = center > bg
        thresh = bg + contrast*0.20 if bright else bg - contrast*0.20

        lx, rx = None, None
        for x in range(min(nx,w-1), -1, -1):
            if (bright and profile[x] < thresh) or (not bright and profile[x] > thresh):
                lx = x; break
        for x in range(max(nx,0), w):
            if (bright and profile[x] < thresh) or (not bright and profile[x] > thresh):
                rx = x; break
        if lx is not None and rx is not None and rx-lx > 5:
            distances.append(sy - ny)
            widths.append(rx - lx)
            left_xs.append(lx)
            right_xs.append(rx)

    if len(distances) < 5:
        return {"angle": None, "confidence": 0, "method": "width_regression"}

    D = np.array(distances, dtype=np.float64)
    W = np.array(widths, dtype=np.float64)
    lxs = np.array(left_xs, dtype=np.float64)
    rxs = np.array(right_xs, dtype=np.float64)

    # 方法A: width = k*d 回归
    k = np.sum(D*W) / np.sum(D**2)
    cone_a = 2 * np.degrees(np.arctan(k/2))
    pred = k*D
    ss_res = np.sum((W-pred)**2)
    ss_tot = np.sum((W-np.mean(W))**2) + 1e-10
    r2 = 1 - ss_res/ss_tot

    # 方法B: 分别拟合左右边界线 (RANSAC)
    scan_ys = positions[:len(lxs)]
    left_pts = np.column_stack([lxs, scan_ys])
    right_pts = np.column_stack([rxs, scan_ys])
    la, _ = _fit_boundary_line(left_pts, nozzle)
    ra, _ = _fit_boundary_line(right_pts, nozzle)
    cone_b = abs(ra - la) if la is not None and ra is not None else None

    # 取均值
    if cone_b is not None:
        cone = (cone_a + cone_b) / 2
    else:
        cone = cone_a

    conf = min(1.0, max(0, r2) * (len(D)/10))
    if cone < 10 or cone > 160: conf *= 0.5
    return {"angle": cone, "confidence": conf, "method": "width_regression",
            "details": f"k={k:.3f} R²={r2:.3f} A={cone_a:.1f}° B={cone_b:.1f}° n={len(D)}" if cone_b else f"k={k:.3f} R²={r2:.3f} A={cone_a:.1f}° n={len(D)}"}


def measure_multi_band_gradient(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> Dict:
    """方法4: 多半径带梯度 (取中位数)"""
    h, w = enhanced.shape
    nx, ny = nozzle
    grad_mag = np.sqrt(cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)**2 +
                       cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)**2)
    max_r = min(h-ny-10, w//2) * 0.85
    ar = CONFIG["boundary_angle_range"]
    num_a = 181
    angles = np.linspace(-ar, ar, num_a)
    mid = num_a // 2
    ce = int(5.0/ar*mid)

    bands = [(0.05,0.20), (0.10,0.30), (0.15,0.40), (0.20,0.50)]
    band_angles = []
    for rs, re in bands:
        radii = np.linspace(max(10, max_r*rs), max_r*re, 20)
        if radii[-1] - radii[0] < 10: continue
        scores = np.zeros(num_a)
        for i, a in enumerate(angles):
            rad = np.radians(90-a)
            dx, dy = np.cos(rad), np.sin(rad)
            t, c = 0.0, 0
            for r in radii:
                x, y = int(nx+r*dx), int(ny+r*dy)
                if 0 <= x < w and 0 <= y < h: t += grad_mag[y,x]; c += 1
            if c > 0: scores[i] = t/c
        ss = ndimage.uniform_filter1d(scores, 5)
        lss, rss = ss[:mid-ce], ss[mid+ce:]
        if len(lss) > 0 and len(rss) > 0:
            ba = angles[np.argmax(rss)+mid+ce] - angles[np.argmax(lss)]
            if 10 < ba < 160: band_angles.append(ba)

    if not band_angles:
        return {"angle": None, "confidence": 0, "method": "near_gradient_multi"}
    cone = float(np.median(band_angles))
    std = float(np.std(band_angles)) if len(band_angles) > 1 else 15.0
    return {"angle": cone, "confidence": min(1.0, max(0.2, 1-std/20)),
            "method": "near_gradient_multi",
            "details": f"median={cone:.1f}° std={std:.1f}° n={len(band_angles)}"}


def measure_texture(enhanced: np.ndarray, nozzle: Tuple[int, int]) -> Dict:
    """方法5: 纹理法 (局部方差, 低对比度利器)"""
    h, w = enhanced.shape
    nx, ny = nozzle
    gf = enhanced.astype(np.float64)
    ks = 5
    mean_l = cv2.blur(gf, (ks,ks))
    tex = np.sqrt(np.clip(cv2.blur(gf**2, (ks,ks)) - mean_l**2, 0, None))

    near_y = min(ny+25, h-1)
    spray_t = tex[near_y, max(0,nx-30):min(w,nx+30)].mean()
    bg_t = (tex[near_y, :max(30,w//10)].mean() + tex[near_y, min(w-30,w*9//10):].mean()) / 2
    tc = spray_t / (bg_t + 1e-10)
    if tc < 2.0:
        return {"angle": None, "confidence": 0, "method": "texture",
                "details": f"contrast={tc:.1f}x (too low)"}

    max_d = min(60, int((h-ny)*0.10))
    instant_angles = []
    for d in range(12, max_d, 2):
        sy = ny + d
        if sy >= h: break
        row = tex[sy, :]
        peak = np.max(row[max(0,nx-60):min(w,nx+60)])
        if peak < 0.3: continue
        above = np.where(row > peak*0.5)[0]
        if len(above) < 3: continue
        sw = above[-1] - above[0]
        if sw < 5 or sw > w*0.95: continue
        a = 2*np.degrees(np.arctan(sw/(2.0*d)))
        if 10 < a < 170: instant_angles.append(a)

    if len(instant_angles) < 3:
        return {"angle": None, "confidence": 0, "method": "texture"}
    cone = float(np.median(instant_angles))
    std = float(np.std(instant_angles))
    conf = max(0.2, 1-std/15) * min(1.0, tc/10)
    return {"angle": cone, "confidence": conf, "method": "texture",
            "details": f"{cone:.1f}°±{std:.1f}° contrast={tc:.1f}x n={len(instant_angles)}"}


# ═══════════════════════════════════════════════════════════════
# 融合
# ═══════════════════════════════════════════════════════════════
def fuse_results(results: List[Dict]) -> Dict:
    """多方法结果融合: 一致性检查 + 加权平均"""
    valid = [r for r in results if r.get("angle") is not None and r["angle"] > 0]
    if not valid:
        return {"final_angle": None, "confidence": 0}
    if len(valid) == 1:
        return {"final_angle": valid[0]["angle"], "confidence": valid[0].get("confidence",0.5)*0.7,
                "individual": {valid[0]["method"]: valid[0]["angle"]}}

    angles = np.array([r["angle"] for r in valid])
    confs = np.array([r.get("confidence", 0.5) for r in valid])
    methods = [r["method"] for r in valid]

    # 一致性: 去除偏离中值过大的
    median = np.median(angles)
    mask = np.abs(angles - median) < CONFIG["consistency_threshold"]
    if mask.sum() == 0:
        best = np.argmax(confs)
        return {"final_angle": float(angles[best]), "confidence": float(confs[best])*0.5,
                "individual": {m: float(a) for m,a in zip(methods, angles)}}

    ca, cc, cm = angles[mask], confs[mask], [m for m,ok in zip(methods, mask) if ok]
    weights = np.array([CONFIG["weights"].get(m, 0.25) for m in cm]) * cc
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()
    final = float(np.sum(ca * weights))
    std = float(np.std(ca))
    overall_conf = 0.4*np.mean(cc) + 0.3*(mask.sum()/len(valid)) + 0.3*max(0, 1-std/10)

    return {"final_angle": final, "confidence": float(overall_conf),
            "std": std, "consistent_methods": cm,
            "individual": {m: float(a) for m, a in zip(methods, angles)}}


# ═══════════════════════════════════════════════════════════════
# 投影误差
# ═══════════════════════════════════════════════════════════════
def projection_warning(angle: float) -> str:
    """相机俯仰投影误差提示"""
    if angle is None: return ""
    half = np.radians(angle/2)
    lines = ["\n  ⚠️  投影误差提示 (相机俯仰角→实际角度):"]
    for tilt in [10, 20, 30]:
        real = 2*np.degrees(np.arctan(np.tan(half)*np.cos(np.radians(tilt))))
        lines.append(f"      俯仰{tilt}°: 实际≈{real:.1f}° (误差~{abs(angle-real)/angle*100:.0f}%)")
    lines.append("      公式: tan(α_real/2) = tan(α_measured/2) × cos(θ_tilt)")
    lines.append("      建议: 相机光轴尽量垂直于喷雾中轴线")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 背景作差 (有参考底图时使用)
# ═══════════════════════════════════════════════════════════════
def _subtract_background(image: np.ndarray, bg_path: str, verbose: bool = True) -> np.ndarray:
    """
    用无喷雾的同机位参考图做作差, 得到纯喷雾图像
    
    优势:
    - 完全消除背景纹理/设备/反光干扰
    - 喷雾信号对比度提升数倍~数十倍
    - 喷口检测和角度测量精度大幅提高
    - 适合工业场景: 相机固定, 只需拍一张空背景
    
    处理逻辑:
    1. 对齐尺寸
    2. 绝对值差分 |spray - background|
    3. 归一化到 [0, 255]
    4. 返回增强后的差分图像 (作为新的输入图像)
    """
    bg = cv2.imread(bg_path)
    if bg is None:
        if verbose: print(f"  ⚠️  无法读取背景图: {bg_path}, 跳过作差")
        return image
    
    # 对齐尺寸
    h, w = image.shape[:2]
    bh, bw = bg.shape[:2]
    if (bh, bw) != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
    
    # 差分 (用浮点避免溢出)
    img_f = image.astype(np.float32)
    bg_f = bg.astype(np.float32)
    diff = np.abs(img_f - bg_f)
    
    # 通道合并 (取max通道差异, 保留所有颜色通道的喷雾信息)
    if len(diff.shape) == 3:
        diff_gray = np.max(diff, axis=2)
    else:
        diff_gray = diff
    
    # 归一化
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 轻度去噪 (消除传感器噪声残留)
    diff_norm = cv2.GaussianBlur(diff_norm, (3, 3), 0)
    
    # 转回3通道 (保持后续pipeline兼容)
    result = cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR)
    
    if verbose:
        spray_signal = np.mean(diff_norm[diff_norm > 20]) if np.sum(diff_norm > 20) > 0 else 0
        coverage = np.sum(diff_norm > 20) / diff_norm.size * 100
        print(f"  ✓ 背景作差完成: 信号强度={spray_signal:.0f}, 喷雾覆盖={coverage:.1f}%")
    
    return result


# ═══════════════════════════════════════════════════════════════
# 主测量流程
# ═══════════════════════════════════════════════════════════════
def measure(image_path: str, nozzle_pos: Optional[Tuple[int,int]] = None,
            background_path: Optional[str] = None,
            verbose: bool = True) -> Dict:
    """
    测量喷雾锥角 - 主入口
    
    参数:
        image_path: 图像路径
        nozzle_pos: 喷口(x,y)原图坐标(会随图像一起缩放), None则自动检测
        background_path: 无喷雾背景图路径(同机位), 提供后用作差法大幅提升精度
        verbose: 打印详细信息
    返回:
        {"cone_angle": float, "confidence": float, ...}
    """
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Cannot read: {image_path}"}
    name = os.path.basename(image_path)
    if verbose: print(f"\n{'='*60}\n Processing: {name}\n{'='*60}")

    # 背景作差预处理
    if background_path:
        image = _subtract_background(image, background_path, verbose)

    # 预处理
    pp = preprocess(image)
    gray, enhanced, scale = pp["gray"], pp["enhanced"], pp["scale"]

    # 喷口 (缩放坐标)
    if nozzle_pos:
        nozzle = (int(nozzle_pos[0]*scale), int(nozzle_pos[1]*scale))
        nozzle_conf = 1.0
        if verbose: print(f"  Nozzle: {nozzle} (manual, scale={scale:.3f})")
    else:
        nozzle, nozzle_conf = detect_nozzle(gray, enhanced)
        if verbose:
            print(f"  Nozzle: {nozzle} (auto, confidence={nozzle_conf:.2f})")
            if nozzle_conf < 0.5:
                print(f"  ⚠️  喷口检测置信度低, 建议手动指定 --nozzle x,y")

    # 多方法测量 (全部基于 enhanced 灰度图)
    if verbose: print(f"  Near zone: {CONFIG['near_start_ratio']*100:.0f}%-{CONFIG['near_end_ratio']*100:.0f}%")
    results = [
        measure_near_hough(enhanced, nozzle),
        measure_gradient_profile(enhanced, nozzle),
        measure_width_regression(enhanced, nozzle),
        measure_multi_band_gradient(enhanced, nozzle),
        measure_texture(enhanced, nozzle),
    ]
    if verbose:
        for r in results:
            a = f"{r['angle']:.1f}°" if r['angle'] else "N/A"
            print(f"    {r['method']:<20}: {a}  {r.get('details','')}")

    # 融合
    fusion = fuse_results(results)
    final = fusion.get("final_angle")
    conf = fusion.get("confidence", 0)

    if verbose:
        if final:
            print(f"\n  ★ Result: {final:.1f}° (confidence: {conf:.0%})")
            print(projection_warning(final))
        else:
            print(f"\n  ✗ Measurement failed")

    return {"image": name, "cone_angle": final, "confidence": conf,
            "nozzle": nozzle, "scale": scale, "details": fusion,
            "timestamp": str(datetime.now())}


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="喷雾锥角测量 (单文件版)")
    parser.add_argument("image", nargs="?", help="图像路径")
    parser.add_argument("--batch", type=str, help="批量处理目录")
    parser.add_argument("--nozzle", type=str, help="喷口位置 x,y (原图坐标)")
    parser.add_argument("--background", "--bg", type=str, help="无喷雾背景参考图 (同机位)")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    nozzle = None
    if args.nozzle:
        parts = args.nozzle.split(",")
        nozzle = (int(parts[0]), int(parts[1]))

    bg = args.background

    if args.batch:
        import glob
        files = sorted(glob.glob(os.path.join(args.batch, "*.jpg")) +
                      glob.glob(os.path.join(args.batch, "*.png")))
        all_r = []
        for f in files:
            r = measure(f, nozzle, background_path=bg, verbose=not args.quiet)
            all_r.append(r)
            if r.get("cone_angle"):
                print(f"  → {r['cone_angle']:.1f}°\n")
        # 保存汇总
        out = os.path.join(args.batch, "batch_results.json")
        with open(out, "w") as fp:
            json.dump(all_r, fp, indent=2, default=str)
        print(f"\nResults saved to: {out}")
    elif args.image:
        r = measure(args.image, nozzle, background_path=bg, verbose=not args.quiet)
        if r.get("cone_angle"):
            print(f"\n{'='*40}\n  Cone Angle: {r['cone_angle']:.1f}°\n  Confidence: {r['confidence']:.0%}\n{'='*40}")
    else:
        parser.print_help()
