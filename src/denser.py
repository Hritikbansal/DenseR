# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Divergence-Enhanced Nuanced Supervision for Effective Reasoning (DENSER)

Per-token weighting for GRPO that uses intermediate hidden states to focus
gradient on tokens that are (1) semantically distinct from the opposite
reward class and (2) unique within their own reward class.

Reference: see `compute_denser_weights` for the main entry point.
"""

import torch
import torch.nn.functional as F


def _batched_windowed_divergence(
    query_hs: torch.Tensor,
    ref_hs: torch.Tensor,
    mask_q: torch.Tensor,
    masks_ref: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    Per-token divergence of one query sequence from N reference sequences,
    computed in a single batched operation.

    For each query token, finds the max cosine similarity to any reference
    token within a proportional alignment window, averaged across all
    valid reference sequences.  Divergence = 1 - mean_similarity.

    Args:
        query_hs:   (T_q, D) hidden states for the query.
        ref_hs:     (N, T_r, D) hidden states for N reference sequences.
        mask_q:     (T_q,) binary mask for the query.
        masks_ref:  (N, T_r) binary masks for references.
        window_size: half-width of the proportional alignment window.

    Returns:
        (T_q,) divergence values in [0, 1]; higher = more different.
    """
    T_q, D = query_hs.shape
    N, T_r, _ = ref_hs.shape
    device = query_hs.device

    # Filter out references with no valid tokens
    valid_refs = masks_ref.sum(dim=1) > 0  # (N,)
    if valid_refs.sum() == 0:
        return torch.ones(T_q, device=device) * mask_q.float()

    ref_hs = ref_hs[valid_refs]        # (N', T_r, D)
    masks_ref = masks_ref[valid_refs]   # (N', T_r)
    N_valid = ref_hs.shape[0]

    # Normalise for cosine similarity
    q_norm = F.normalize(query_hs, dim=-1)   # (T_q, D)
    r_norm = F.normalize(ref_hs, dim=-1)     # (N', T_r, D)

    # Batched similarity: one einsum replaces N separate matmuls
    # (T_q, D) x (N', D, T_r) -> (N', T_q, T_r)
    sim_matrix = torch.einsum("qd,nrd->nqr", q_norm, r_norm)

    # Build proportional-window masks per reference
    T_q_valid = mask_q.sum().clamp(min=1).float()
    T_r_valids = masks_ref.sum(dim=1).clamp(min=1).float()  # (N',)

    q_idx = torch.arange(T_q, device=device, dtype=torch.float32)  # (T_q,)
    r_idx = torch.arange(T_r, device=device)                       # (T_r,)

    # Proportional center for each (ref, query_pos): (N', T_q)
    centers = (q_idx.unsqueeze(0) * T_r_valids.unsqueeze(1) / T_q_valid).long()

    # Distance from center: (N', T_q, T_r)
    dist = (r_idx.view(1, 1, T_r) - centers.unsqueeze(2)).abs()

    window_mask = (dist <= window_size) & masks_ref.bool().unsqueeze(1)  # (N', T_q, T_r)

    # Apply mask; max over ref positions
    sim_matrix = sim_matrix.masked_fill(~window_mask, -float("inf"))
    max_sims = sim_matrix.max(dim=2).values.clamp(min=0.0)  # (N', T_q)
    max_sims = max_sims * mask_q.float().unsqueeze(0)        # zero padded query positions

    # Average across references, then divergence
    avg_max_sim = max_sims.mean(dim=0)                       # (T_q,)
    divergence = (1.0 - avg_max_sim) * mask_q.float()
    return divergence


def compute_denser_weights(
    hidden_states: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    num_generations: int,
    window_size: int = 5,
    alpha_cross: float = 1.0,
    alpha_within: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-token DENSER weights for every rollout in the batch.

    Each rollout's token weight is the weighted sum of:
      * **cross-class divergence** -- how different this token's hidden state
        is from all rollouts of the *opposite* advantage sign.
      * **within-class uniqueness** -- how different it is from rollouts of
        the *same* advantage sign.

    ``combined = alpha_cross * cross_div + alpha_within * within_uniq``

    After mean-1 normalisation, the weights are blended with uniform weights:

    ``final = (1 - beta) * 1.0 + beta * normalised_combined``

    ``beta`` controls how much DENSER influence is applied:
      * 0.0 = fully uniform (vanilla GRPO)
      * 1.0 = full DENSER (default, preserves existing behaviour)
      * 0.1 = gentle nudge from DENSER

    Note: since ``alpha_cross / alpha_within`` ratio is what matters (the
    individual magnitudes cancel in normalisation), ``beta`` is the only
    knob that controls the *strength* of DENSER's redistribution.

    Args:
        hidden_states:    (B, T, D) completion hidden states.
        advantages:       (B,) per-rollout advantages.
        completion_mask:  (B, T) binary mask (1 = real token).
        num_generations:  G -- number of rollouts per prompt.
        window_size:      half-width of the proportional alignment window.
        alpha_cross:      scaling factor for cross-class divergence
                          (0 = ignore cross-class signal, higher = more weight).
        alpha_within:     scaling factor for within-class uniqueness
                          (0 = ignore uniqueness, higher = more diversity-seeking).
        beta:             blending strength between uniform (0) and DENSER (1).

    Returns:
        Tuple of:
          * ``(B, T)`` tensor of per-token weights.
          * ``(B,)`` tensor of per-completion mean cross-class divergence.
          * ``(B,)`` tensor of per-completion mean within-class uniqueness.
    """
    B, T, D = hidden_states.shape
    device = hidden_states.device
    weights = torch.ones(B, T, device=device)
    avg_cross_div = torch.zeros(B, device=device)
    avg_within_uniq = torch.zeros(B, device=device)

    num_groups = B // num_generations

    for g in range(num_groups):
        s = g * num_generations
        e = s + num_generations

        g_hs = hidden_states[s:e]          # (G, T, D)
        g_adv = advantages[s:e]            # (G,)
        g_mask = completion_mask[s:e]       # (G, T)

        pos_idx = (g_adv >= 0).nonzero(as_tuple=True)[0]
        neg_idx = (g_adv < 0).nonzero(as_tuple=True)[0]

        # No cross-class signal -> keep uniform weights
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        # ----------------------------------------------------------
        # Cross-class contrast exists → standard DENSER weighting
        # ----------------------------------------------------------
        pos_hs, pos_mask = g_hs[pos_idx], g_mask[pos_idx]
        neg_hs, neg_mask = g_hs[neg_idx], g_mask[neg_idx]

        for i in range(num_generations):
            is_pos = g_adv[i] >= 0

            # --- cross-class divergence ---
            opp_hs = neg_hs if is_pos else pos_hs
            opp_mask = neg_mask if is_pos else pos_mask
            cross_div = _batched_windowed_divergence(
                g_hs[i], opp_hs, g_mask[i], opp_mask, window_size
            )

            # --- within-class uniqueness (excluding self) ---
            if is_pos:
                same_keep = torch.ones(len(pos_idx), dtype=torch.bool, device=device)
                self_loc = (pos_idx == i).nonzero(as_tuple=True)[0]
                if len(self_loc) > 0:
                    same_keep[self_loc[0]] = False
                same_hs, same_mask = pos_hs[same_keep], pos_mask[same_keep]
            else:
                same_keep = torch.ones(len(neg_idx), dtype=torch.bool, device=device)
                self_loc = (neg_idx == i).nonzero(as_tuple=True)[0]
                if len(self_loc) > 0:
                    same_keep[self_loc[0]] = False
                same_hs, same_mask = neg_hs[same_keep], neg_mask[same_keep]

            if same_hs.shape[0] > 0:
                within_uniq = _batched_windowed_divergence(
                    g_hs[i], same_hs, g_mask[i], same_mask, window_size
                )
            else:
                within_uniq = torch.ones(T, device=device)

            # Diagnostics
            valid = g_mask[i].bool()
            if valid.any():
                avg_cross_div[s + i] = cross_div[valid].mean()
                avg_within_uniq[s + i] = within_uniq[valid].mean()

            combined = alpha_cross * cross_div + alpha_within * within_uniq

            # Normalise to mean-1 over valid tokens so loss scale is unchanged
            if valid.any():
                combined = combined / combined[valid].mean().clamp(min=1e-8)

            # Blend toward uniform: beta=0 → all ones, beta=1 → full DENSER
            combined = (1.0 - beta) + beta * combined

            weights[s + i] = combined

    return weights, avg_cross_div, avg_within_uniq