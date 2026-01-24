from tqdm import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from PIL import Image
import math
import torch
import os
import cv2
import numpy as np
from typing import Optional, Union


# ============================================================
# Delta-map computation utilities
# ============================================================

def compute_delta_map(Vt_src: torch.Tensor, Vt_tar: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized per-pixel L2 distance map between source and target velocity fields.

    Args:
        Vt_src: (B, H*W, C) source velocity tensor
        Vt_tar: (B, H*W, C) target velocity tensor

    Returns:
        delta_norm: (B, H, W) normalized delta map in [0, 1]
    """
    assert Vt_src.shape == Vt_tar.shape, "Vt_src and Vt_tar must have the same shape"

    batch_size, num_pixels, channels = Vt_src.shape
    latent_size = int(math.isqrt(num_pixels))
    assert latent_size * latent_size == num_pixels, (
        f"num_pixels ({num_pixels}) must be a perfect square"
    )

    # (B, H*W, C) -> (B, C, H, W)
    Vt_src_4d = (
        Vt_src.view(batch_size, latent_size, latent_size, channels)
        .permute(0, 3, 1, 2)
    )
    Vt_tar_4d = (
        Vt_tar.view(batch_size, latent_size, latent_size, channels)
        .permute(0, 3, 1, 2)
    )

    # L2 norm across channel dimension
    delta = torch.norm(Vt_tar_4d - Vt_src_4d, p=2, dim=1)

    delta_min = delta.view(batch_size, -1).min(dim=1)[0].view(batch_size, 1, 1)
    delta_max = delta.view(batch_size, -1).max(dim=1)[0].view(batch_size, 1, 1)

    delta_norm = (delta - delta_min) / (delta_max - delta_min + 1e-8)
    return delta_norm


# ============================================================
# Flow-matching / FLUX-related helpers
# ============================================================

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process used in flow-matching.

    Args:
        scheduler: diffusion scheduler
        sample: clean latent
        timestep: current timestep
        noise: noise tensor

    Returns:
        Scaled noisy latent
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """
    Compute mu shift for FLUX scheduler based on image sequence length.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def calc_v_flux(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: Optional[torch.Tensor],
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute velocity prediction V_t using FLUX transformer.
    """
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


# ============================================================
# Region growing based mask generation
# ============================================================

def bfs_expand(seeds, patch_means, patch_mask, threshold, neighbors):
    """
    BFS-based region growing over patch grid with dynamic mean update.
    """
    from collections import deque

    H, W = patch_means.shape

    for sy, sx in seeds:
        if patch_mask[sy, sx] == 1:
            continue

        sum_val = patch_means[sy, sx]
        count = 1
        region_mean = sum_val / count

        queue = deque([(sy, sx)])
        patch_mask[sy, sx] = 1

        while queue:
            y, x = queue.popleft()
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and patch_mask[ny, nx] == 0:
                    if abs(patch_means[ny, nx] - region_mean) < threshold:
                        patch_mask[ny, nx] = 1
                        queue.append((ny, nx))

                        sum_val += patch_means[ny, nx]
                        count += 1
                        region_mean = sum_val / count

    return patch_mask


def region_growing_segmentation(
    heatmap: np.ndarray,
    patch_size: int = 1,
    threshold: float = 0.1,
    fix_margin: float = 0.0,
    fix_adjacent_only: bool = False,
) -> np.ndarray:
    """
    Region growing segmentation on a single-channel heatmap.

    Returns:
        Binary pixel-level mask.
    """
    if heatmap.max() <= 1.0:
        heatmap = (heatmap * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(heatmap, (3, 3), 0)
    h, w = blurred.shape

    ph = pw = patch_size
    H, W = h // ph, w // pw
    patch_means = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            patch = blurred[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            patch_means[i, j] = float(np.mean(patch))

    global_mean = float(np.mean(blurred))
    global_std = float(np.std(blurred))
    high_patches = np.argwhere(patch_means > global_mean + 0.5 * global_std)

    if len(high_patches) == 0:
        max_y, max_x = np.unravel_index(np.argmax(blurred), blurred.shape)
        seeds = [(max_y // ph, max_x // pw)]
    else:
        seeds = [tuple(p) for p in high_patches]

    patch_mask = np.zeros((H, W), dtype=np.uint8)

    neigh8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)]
    patch_mask = bfs_expand(seeds, patch_means, patch_mask, threshold, neigh8)

    neigh4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_seeds = np.argwhere(patch_mask == 1)
    patch_mask = bfs_expand(new_seeds, patch_means, patch_mask, threshold, neigh4)

    region_vals = patch_means[patch_mask > 0]
    if region_vals.size > 0:
        region_mean = float(np.mean(region_vals))
        if fix_adjacent_only:
            kernel = np.ones((3, 3), np.uint8)
            neighbor_ring = cv2.dilate(patch_mask, kernel, iterations=1)
            eligible = (neighbor_ring > 0) & (patch_mask == 0)
            to_add = eligible & (patch_means > region_mean + fix_margin)
        else:
            to_add = (patch_mask == 0) & (patch_means > region_mean + fix_margin)

        patch_mask[to_add] = 1

    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            if patch_mask[i, j] == 1:
                mask[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = 1

    norm_heatmap = heatmap.astype(np.float32) / 255.0
    mask[norm_heatmap < 0.2] = 0

    kernel_px = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_px)
    mask = cv2.erode(mask, kernel_px, iterations=1)
    mask = cv2.dilate(mask, kernel_px, iterations=3)

    return mask


def get_mask(cumulative_avg_attn: torch.Tensor, channels: int = 64) -> torch.Tensor:
    """
    Convert accumulated attention / delta map into a broadcastable latent mask.
    """
    attn_np = cumulative_avg_attn.squeeze(0).to(torch.float32).cpu().numpy()
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    mask = region_growing_segmentation(attn_np)

    mask = mask[np.newaxis, :, :]
    M = mask.reshape(1, -1)
    M = torch.from_numpy(M).float()
    M = (M - M.min()) / (M.max() - M.min() + 1e-8)

    M = M.unsqueeze(-1).expand(-1, -1, channels)
    return M


# ============================================================
# Main FusionEdit pipeline
# ============================================================

@torch.no_grad()
def FusionEdit(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    T_steps: int = 28,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):
    """
    Fusion-based editing via FLUX flow matching.
    """
    device = x_src.device

    orig_height = x_src.shape[2] * pipe.vae_scale_factor // 2
    orig_width = x_src.shape[3] * pipe.vae_scale_factor // 2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src,
    )

    x_src_packed = pipe._pack_latents(
        x_src,
        x_src.shape[0],
        num_channels_latents,
        x_src.shape[2],
        x_src.shape[3],
    )

    latent_tar_image_ids = latent_src_image_ids

    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )

    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    pipe._num_timesteps = len(timesteps)

    # Encode prompts
    src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # --------------------------------------------------------
    # Phase 1: Delta-map estimation
    # --------------------------------------------------------

    zt_edit_1 = x_src_packed.clone()
    delta_map_sum = None
    num_iterations = 10

    t = timesteps[5]
    scheduler._init_step_index(t)
    t_i = scheduler.sigmas[scheduler.step_index]

    for _ in range(num_iterations):
        fwd_noise = torch.randn_like(x_src_packed)
        zt_src_1 = (1 - t_i) * x_src_packed + t_i * fwd_noise
        zt_tar_1 = zt_edit_1 + zt_src_1 - x_src_packed

        Vt_src_1 = calc_v_flux(
            pipe, zt_src_1,
            src_prompt_embeds, src_pooled_prompt_embeds,
            src_guidance, src_text_ids,
            latent_src_image_ids, t,
        )

        Vt_tar_1 = calc_v_flux(
            pipe, zt_tar_1,
            tar_prompt_embeds, tar_pooled_prompt_embeds,
            src_guidance, tar_text_ids,
            latent_tar_image_ids, t,
        )

        delta_map = compute_delta_map(Vt_src_1, Vt_tar_1)
        if delta_map_sum is None:
            delta_map_sum = torch.zeros_like(delta_map)

        delta_map_sum += delta_map

    delta_map_avg = delta_map_sum / num_iterations
    tar_mask_1 = get_mask(delta_map_avg).to(device)

    # --------------------------------------------------------
    # Phase 2: Guided ODE editing
    # --------------------------------------------------------

    for i, t in tqdm(enumerate(timesteps)):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) else t_i

        if T_steps - i > n_min:
            fwd_noise = torch.randn_like(x_src_packed)
            zt_src_1 = (1 - t_i) * x_src_packed + t_i * fwd_noise
            zt_tar_1 = zt_edit_1 + zt_src_1 - x_src_packed

            Vt_src_1 = calc_v_flux(
                pipe, zt_src_1,
                src_prompt_embeds, src_pooled_prompt_embeds,
                src_guidance, src_text_ids,
                latent_src_image_ids, t,
            )

            Vt_tar_1 = calc_v_flux(
                pipe, zt_tar_1,
                tar_prompt_embeds, tar_pooled_prompt_embeds,
                tar_guidance, tar_text_ids,
                latent_tar_image_ids, t,
            )

            V_delta = Vt_tar_1 - Vt_src_1

            zt_edit_1 = tar_mask_1 * zt_edit_1 + (1 - tar_mask_1) * x_src_packed
            zt_edit_1 = zt_edit_1.to(torch.float32)
            zt_edit_1 = zt_edit_1 + (t_im1 - t_i) * V_delta
            zt_edit_1 = zt_edit_1.to(V_delta.dtype)

    out = pipe._unpack_latents(
        zt_edit_1,
        orig_height,
        orig_width,
        pipe.vae_scale_factor,
    )

    torch.cuda.empty_cache()
    return out
