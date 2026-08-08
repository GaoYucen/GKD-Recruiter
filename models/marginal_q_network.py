from __future__ import annotations

import torch
from torch import nn


class MarginalQNetwork(nn.Module):
    def __init__(self, pair_dim: int, state_dim: int, local_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(pair_dim + local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, pair_features, local_features, valid_mask=None):
        state_h = self.state_encoder(state)
        action_input = torch.cat([pair_features, local_features], dim=-1)
        action_h = self.action_encoder(action_input)
        state_expanded = state_h.unsqueeze(1).expand_as(action_h)
        fused = torch.cat(
            [
                state_expanded,
                action_h,
                state_expanded * action_h,
                torch.abs(state_expanded - action_h),
            ],
            dim=-1,
        )
        q = self.q_head(fused).squeeze(-1)
        if valid_mask is not None:
            mask = valid_mask.to(dtype=torch.bool, device=q.device)
            q = q.masked_fill(~mask, torch.finfo(q.dtype).min)
        return q


class CandidateAwareDuelingQNetwork(nn.Module):
    def __init__(self, pair_dim: int, state_dim: int, local_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(pair_dim + local_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state, pair_features, local_features, valid_mask=None):
        state_h = self.state_encoder(state)
        action_h = self.action_encoder(torch.cat([pair_features, local_features], dim=-1))
        if valid_mask is None:
            valid_mask = torch.ones(action_h.shape[:2], dtype=torch.bool, device=action_h.device)
        mask = valid_mask.to(dtype=torch.bool, device=action_h.device)
        mask_f = mask.unsqueeze(-1).to(action_h.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        masked_actions = action_h * mask_f
        context_mean = masked_actions.sum(dim=1, keepdim=True) / denom
        masked_for_max = action_h.masked_fill(~mask.unsqueeze(-1), torch.finfo(action_h.dtype).min)
        context_max = masked_for_max.max(dim=1, keepdim=True).values
        context_max = torch.where(torch.isfinite(context_max), context_max, torch.zeros_like(context_max))
        state_expanded = state_h.unsqueeze(1).expand_as(action_h)
        context_mean_expanded = context_mean.expand_as(action_h)
        context_max_expanded = context_max.expand_as(action_h)
        interaction = torch.cat(
            [
                state_expanded,
                action_h,
                context_mean_expanded,
                context_max_expanded,
                state_expanded * action_h,
                torch.abs(state_expanded - action_h),
            ],
            dim=-1,
        )
        advantage = self.advantage_head(interaction).squeeze(-1)
        value = self.value_head(torch.cat([state_h, context_mean.squeeze(1), context_max.squeeze(1)], dim=-1)).squeeze(-1)
        masked_advantage = advantage.masked_fill(~mask, 0.0)
        mean_advantage = masked_advantage.sum(dim=1, keepdim=True) / mask_f.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1.0)
        q = value.unsqueeze(1) + advantage - mean_advantage
        q = q.masked_fill(~mask, torch.finfo(q.dtype).min)
        return q