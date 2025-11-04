from typing import Dict
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    x = current['step'] - current['activated_steps'][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current['step'] == (current['num_steps'] - 1):
        cache_dic['cache'][-1][current['layer']][current['module']] = {}



def interpolate_features(tensor_list, target_T, prevprev_tensor=None, stage_ratio=0.0):
    """
    GPU 전용 Linear Interpolation (양 끝점 고정)
    tensor_list: list of [B, C, H, W] tensors on GPU
    target_T: 원하는 길이
    """
    if len(tensor_list) == 0:
        raise ValueError("tensor_list is empty.")

    device = tensor_list[0].device
    orig_T = len(tensor_list)
    stacked = torch.stack(tensor_list, dim=0)  # [T, B, C, H, W]
    orig_shape = stacked.shape[1:]             # [B, C, H, W]

    # flatten for interpolation
    flattened = stacked.reshape(orig_T, -1)    # [T, D]

    # original and target positions
    orig_x = torch.linspace(0, 1, orig_T, device=device)
    target_x = torch.linspace(0, 1, target_T, device=device)

    # find indices for interpolation (exclude exact 0 and 1)
    idx = torch.clamp(torch.searchsorted(orig_x, target_x) - 1, 0, orig_T - 2)
    x0 = orig_x[idx]
    x1 = orig_x[idx + 1]
    f0 = flattened[idx]       # [target_T, D]
    f1 = flattened[idx + 1]

    # linear interpolation
    dx = (target_x - x0) / (x1 - x0 + 1e-10)
    interp = f0 + dx.unsqueeze(1) * (f1 - f0)

    # reshape back
    interp_tensor = interp.reshape(target_T, *orig_shape)

    # === enforce exact endpoints ===
    interp_tensor[0]  = stacked[0]
    interp_tensor[-1] = stacked[-1]

    return interp_tensor



## 그냥 인접 step이랑 feature 변화량
def get_interval(prev_features, curr_features, prev_interval, 
                 high_th=0.7, low_th=0.5, 
                 min_interval=3, max_interval=6):
    """
    Adaptive interval scheduling rule.
    Feature 변화량에 따라 다음 coarse interval 길이를 동적으로 조정함.
    """
    if prev_features is None or curr_features is None:
        return prev_interval  # 첫 segment에서는 그대로 유지
    
    #변화량 계산 (attention + MLP)
    diff_attn = torch.mean(torch.abs(curr_features["attn"] - prev_features["attn"])).item()
    diff_mlp  = torch.mean(torch.abs(curr_features["mlp"]  - prev_features["mlp"])).item()
    diff_mean = (diff_attn + diff_mlp) / 2

    # 2️⃣ Adaptive rule
    if diff_mean > high_th:
        new_interval = max(prev_interval - 1, min_interval)
        status = f" High dynamic ({diff_mean:.4f}) -> {new_interval}"
    elif diff_mean < low_th:
        new_interval = min(prev_interval + 1, max_interval)
        status = f" Stable ({diff_mean:.4f}) -> {new_interval}"
    else:
        new_interval = prev_interval
        status = f"Moderate ({diff_mean:.4f}) = {new_interval}"

    print(f"[get_interval] {status}")
    return new_interval


## Curvature로 
def get_interval_by_feature_curv(F_prevprev, F_prev, F_curr, prev_interval,
                                 high_th=1, low_th=0.5,
                                 min_interval=3, max_interval=6):
    """
    Adaptive coarse-step interval scheduling based on feature curvature.
    
    입력:
        F_prevprev: feature at t-2Δ (tensor)
        F_prev:     feature at t-Δ (tensor)
        F_curr:     feature at t   (tensor)
        prev_interval: 이전 coarse interval 길이
    규칙:
        curvature = mean(|(F_t - F_{t-Δ}) - (F_{t-Δ} - F_{t-2Δ})|)
        curvature ↑  → 변화 급격 → interval 감소
        curvature ↓  → 안정적 → interval 증가
    """

    # 초반부
    if F_prevprev is None or F_prev is None or F_curr is None:
        return prev_interval

    # curvature 계산
    curv_attn = torch.mean(torch.abs(
        (F_curr["attn"] - F_prev["attn"]) - (F_prev["attn"] - F_prevprev["attn"])
    )).item()

    curv_mlp = torch.mean(torch.abs(
        (F_curr["mlp"] - F_prev["mlp"]) - (F_prev["mlp"] - F_prevprev["mlp"])
    )).item()

    curvature = (curv_attn)

    # adaptive
    if curvature > high_th:
        new_interval = max(prev_interval - 1, min_interval)
        status = f"⚠️ High curvature ({curvature:.4f}) → Decrease interval → {new_interval}"
    elif curvature < low_th:
        new_interval = min(prev_interval + 1, max_interval)
        status = f"✅ Stable ({curvature:.4f}) → Increase interval → {new_interval}"
    else:
        new_interval = prev_interval
        status = f"~ Moderate ({curvature:.4f}) → Keep interval = {new_interval}"

    print(f"[get_interval_by_feature_curv] {status}")
    return new_interval



# import torch

# def interpolate_features(tensor_list, target_T, prevprev_tensor=None, stage_ratio=0.0, method="linear"):
#     """
#     GPU 전용 Feature Interpolation (선형 / Hermite / Catmull-Rom / PCHIP)
#     tensor_list: list of [B, C, H, W] tensors on GPU
#     target_T: 원하는 길이
#     method: 'linear' | 'hermite' | 'catmull_rom' | 'pchip'
#     """
#     if len(tensor_list) < 2:
#         raise ValueError("Need at least 2 tensors for interpolation.")
    
#     device = tensor_list[0].device
#     orig_T = len(tensor_list)
#     stacked = torch.stack(tensor_list, dim=0)  # [T, B, C, H, W]
#     orig_shape = stacked.shape[1:]
#     flattened = stacked.reshape(orig_T, -1)   # [T, D]
    
#     # 원본 및 타깃 위치
#     orig_x = torch.linspace(0, 1, orig_T, device=device)
#     target_x = torch.linspace(0, 1, target_T, device=device)
    
#     # 각 타깃의 구간 인덱스
#     idx = torch.clamp(torch.searchsorted(orig_x, target_x) - 1, 0, orig_T - 2)
#     x0, x1 = orig_x[idx], orig_x[idx + 1]
#     f0, f1 = flattened[idx], flattened[idx + 1]
#     s = (target_x - x0) / (x1 - x0 + 1e-10)  # 정규화된 보간 비율

#     # =====================================================
#     # 1️⃣ LINEAR
#     # =====================================================
#     if method == "linear":
#         interp = f0 + s.unsqueeze(1) * (f1 - f0)

#     # =====================================================
#     # 2️⃣ HERMITE (Cubic Hermite with shared slope)
#     # =====================================================
#     elif method == "hermite":
#         m = f1 - f0  # slope (secant)
#         s2, s3 = s**2, s**3
#         h00 = 2*s3 - 3*s2 + 1
#         h10 = s3 - 2*s2 + s
#         h01 = -2*s3 + 3*s2
#         h11 = s3 - s2
#         interp = (h00.unsqueeze(1)*f0 +
#                   h10.unsqueeze(1)*m +
#                   h01.unsqueeze(1)*f1 +
#                   h11.unsqueeze(1)*m)

#     # =====================================================
#     # 3️⃣ CATMULL-ROM (C¹ continuous cubic spline)
#     # =====================================================
#     elif method == "catmull_rom":
#         f_prev = flattened[torch.clamp(idx - 1, 0, orig_T - 1)]
#         f_next = flattened[torch.clamp(idx + 2, 0, orig_T - 1)]
#         m0 = 0.5 * (f1 - f_prev)
#         m1 = 0.5 * (f_next - f0)
#         s2, s3 = s**2, s**3
#         h00 = 2*s3 - 3*s2 + 1
#         h10 = s3 - 2*s2 + s
#         h01 = -2*s3 + 3*s2
#         h11 = s3 - s2
#         interp = (h00.unsqueeze(1)*f0 +
#                   h10.unsqueeze(1)*m0 +
#                   h01.unsqueeze(1)*f1 +
#                   h11.unsqueeze(1)*m1)

#     # =====================================================
#     # 4️⃣ PCHIP (Monotone-preserving Hermite cubic)
#     # =====================================================
#     elif method == "pchip":
#         # 인접 구간 slope (delta)
#         delta = (flattened[1:] - flattened[:-1]) / (orig_x[1:] - orig_x[:-1]).unsqueeze(1)
#         m = torch.zeros_like(flattened)
#         m[1:-1] = (delta[:-1] + delta[1:]) / 2

#         # 단조성 보정 (overshoot 방지)
#         mask = (delta[:-1] * delta[1:]) <= 0
#         m[1:-1][mask] = 0.0

#         m0 = m[idx]
#         m1 = m[idx + 1]

#         s2, s3 = s**2, s**3
#         h00 = 2*s3 - 3*s2 + 1
#         h10 = s3 - 2*s2 + s
#         h01 = -2*s3 + 3*s2
#         h11 = s3 - s2
#         interp = (h00.unsqueeze(1)*f0 +
#                   h10.unsqueeze(1)*(x1 - x0).unsqueeze(1)*m0 +
#                   h01.unsqueeze(1)*f1 +
#                   h11.unsqueeze(1)*(x1 - x0).unsqueeze(1)*m1)

#     else:
#         raise ValueError(f"Unknown method: {method}")

#     interp_tensor = interp.reshape(target_T, *orig_shape)
#     return interp_tensor


# import torch

# def interpolate_two_points(tensor_list, target_T):
#     """
#     Hermite-like cubic interpolation using two points and estimated derivatives.
#     tensor_list: list of 2 tensors [B, C, H, W] on GPU
#     target_T: 원하는 길이
#     """
#     if len(tensor_list) != 2:
#         raise ValueError("tensor_list must contain exactly 2 tensors for this method.")
    
#     device = tensor_list[0].device
#     y0, y1 = tensor_list
#     # shape = [B, C, H, W]
#     orig_shape = y0.shape
    
#     # estimate derivatives (simple finite difference)
#     dy0 = y1 - y0
#     dy1 = y1 - y0  # 두 점뿐이므로 양쪽 slope 동일하게 사용

#     # param t in [0,1]
#     t = torch.linspace(0, 1, target_T, device=device).view(target_T, *([1]* (y0.ndim)))
    
#     # cubic Hermite interpolation formula
#     h00 = 2*t**3 - 3*t**2 + 1
#     h10 = t**3 - 2*t**2 + t
#     h01 = -2*t**3 + 3*t**2
#     h11 = t**3 - t**2
    
#     interp = h00 * y0 + h10 * dy0 + h01 * y1 + h11 * dy1
    
#     return interp


# import torch

# def interpolate_features_1(tensor_list, num_layers=28, target_T=40):
#     """
#     Segment-wise linear interpolation with exact total length = target_T.
#     Each coarse interval expands to variable number of fine steps, with remainders balanced.
#     """
#     if len(tensor_list) == 0:
#         raise ValueError("tensor_list is empty.")
#     # ✅ 모든 텐서를 CPU로 옮김
#     tensor_list = [t.detach().to('cpu') for t in tensor_list]

#     B, T, D = tensor_list[0].shape
#     num_total = len(tensor_list)
#     num_steps = num_total // num_layers
#     num_segments = num_steps - 1

#     base_points = (target_T - num_steps) // num_segments
#     remainder = (target_T - num_steps) % num_segments
#     print(f"[Interp DEBUG] num_steps={num_steps}, base_points={base_points}, remainder={remainder}, num_segments={num_segments}")

#     # reshape [num_steps, num_layers, B, T, D]
#     x = torch.stack(tensor_list, dim=0).view(num_steps, num_layers, B, T, D)
#     fine_feats = []

#     total_points = 0
#     for step in range(num_segments):
#         f0, f1 = x[step], x[step + 1]

#         # 항상 첫 coarse step 추가
#         if step == 0:
#             fine_feats.append(f0)
#             total_points += 1

#         # 구간별 보간 step 수 계산
#         n_points = base_points + (1 if step < remainder else 0)
#         for k in range(1, n_points + 1):
#             alpha = k / (n_points + 1)
#             interp = f0 * (1 - alpha) + f1 * alpha
#             fine_feats.append(interp)
#             total_points += 1

#         # 마지막 coarse step 추가
#         fine_feats.append(f1)
#         total_points += 1

#     fine_feats = torch.stack(fine_feats, dim=0)
#     print(f"[Interp DEBUG] total_points after loop = {total_points}")
#     print(f"[Interp DEBUG] fine_feats.shape[0] = {fine_feats.shape[0]} (target_T={target_T})")

#     assert fine_feats.shape[0] == target_T, f"fine_steps={fine_feats.shape[0]} != target_T={target_T}"
#     fine_feats = torch.flip(fine_feats, dims=[0])
#     return fine_feats



# def interpolate_features(tensor_list, target_T, prevprev_tensor=None, stage_ratio=0.0):
#     """
#     통합형 GPU Feature Interpolation (Adaptive Linear + Curvature Correction)
#     ------------------------------------------------------------------------
#     tensor_list: [prev, curr] feature list (각각 [B, C, H, W] 텐서)
#     prevprev_tensor: 이전 구간의 anchor feature (없을 수 있음)
#     target_T: 생성할 총 step 수 (예: interval + 1)
#     stage_ratio: 전체 coarse step 중 현재 위치 (0~1), 후반부일수록 보정 강화
#     """
#     if len(tensor_list) != 2:
#         raise ValueError("tensor_list must contain exactly [prev, curr]")

#     device = tensor_list[0].device
#     f0, f1 = [t.reshape(t.shape[0], -1) for t in tensor_list]  # [B, D]
#     orig_shape = tensor_list[0].shape
#     target_x = torch.linspace(0, 1, target_T, device=device)

#     # (1) 기본 선형 보간
#     interps = []
#     for s_val in target_x:
#         s = s_val.item()
#         interp = (1 - s) * f0 + s * f1

#         # (2) prevprev_tensor가 있을 경우 curvature correction
#         # if prevprev_tensor is not None and stage_ratio > 1:
#         #     f_prevprev = prevprev_tensor.reshape(prevprev_tensor.shape[0], -1)
#         #     # 이전 변화율과 현재 변화율을 모두 고려
#         #     grad_prev = f0 - f_prevprev
#         #     grad_curr = f1 - f0
#         #     curvature = grad_curr - grad_prev

#         #     β = stage_ratio * s * (1 - s)  # 중앙부에서 가장 강함
#         #     interp = interp + β * curvature  # 곡률 보정

#         interps.append(interp)

#     interps = torch.stack(interps, dim=0).reshape(target_T, *orig_shape)
#     return interps