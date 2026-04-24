"""
Actor-only Orbit Wars model for Kaggle submission.

학습용 OrbitWarsPolicy(model.py)와 *동일* 파라미터 키를 갖는 actor 서브셋.
critic head 만 제외 — state_dict load 시 critic.* 키는 strict=False 로 unexpected 처리.

학습-제출 parity 보장:
  - 동일 config.yaml 읽음 (embed_dim, layers, NUM_HEADS, HISTORY, bins)
  - 동일 forward 경로 (Linear → Temporal → Local → Global → actor head)
  - PLANET_DIM=21 (submission_features 와 일치)

Inference only — torch.no_grad 는 caller 가 감싸도 되고 여기 내부에서도 안전.
"""

import os
import torch
import torch.nn as nn
import yaml

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_CFG_PATH) as f:
    CFG = yaml.safe_load(f)

M   = CFG["model"]
ENV = CFG["env"]

EMBED_DIM              = M["embed_dim"]
PLANET_TEMPORAL_LAYERS = M["planet_temporal_layers"]
FLEET_TEMPORAL_LAYERS  = M["fleet_temporal_layers"]
FLEET_TEMPORAL         = M["fleet_temporal"]
LOCAL_LAYERS           = M["local_layers"]
GLOBAL_LAYERS          = M["global_layers"]
NUM_HEADS              = M["num_heads"]
HISTORY                = M["temporal_window"]
MAX_PLANETS            = ENV["max_planets"]
MAX_FLEETS             = ENV["max_fleets"]
PLANET_DIM             = 21   # submission_features 와 동기화
FLEET_DIM              = 7

SHIPS_MULTIPLIER_BINS = tuple(M.get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
NUM_SHIPS_BINS        = len(SHIPS_MULTIPLIER_BINS)
ACTION_DIM            = 1 + NUM_SHIPS_BINS + MAX_PLANETS


def _make_transformer(layers):
    enc = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 4,
        dropout=0.1,
        batch_first=True,
    )
    return nn.TransformerEncoder(enc, num_layers=layers)


class OrbitWarsActor(nn.Module):
    """OrbitWarsPolicy 의 actor-only 버전.

    nn.Module key layout 이 학습 모델과 완전히 같도록 구성:
      planet_embed / fleet_embed
      planet_temporal_pos / fleet_temporal_pos
      planet_temporal_attn / fleet_temporal_attn(optional)
      local_attn / global_attn
      actor  (critic 은 제외)
    """

    def __init__(self):
        super().__init__()

        self.planet_embed = nn.Linear(PLANET_DIM, EMBED_DIM)
        self.fleet_embed  = nn.Linear(FLEET_DIM,  EMBED_DIM)

        self.planet_temporal_pos = nn.Embedding(HISTORY, EMBED_DIM)
        self.fleet_temporal_pos  = nn.Embedding(HISTORY, EMBED_DIM)

        self.planet_temporal_attn = _make_transformer(PLANET_TEMPORAL_LAYERS)
        self.fleet_temporal_attn  = _make_transformer(FLEET_TEMPORAL_LAYERS) if FLEET_TEMPORAL else None

        self.local_attn  = _make_transformer(LOCAL_LAYERS)
        self.global_attn = _make_transformer(GLOBAL_LAYERS)

        self.actor = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, ACTION_DIM),
        )

    def forward(self, obs_flat):
        """
        obs_flat: (B, HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM))
        returns action_logits: (B, MAX_PLANETS, ACTION_DIM)
        """
        B = obs_flat.shape[0]
        p_size = MAX_PLANETS * PLANET_DIM
        f_size = MAX_FLEETS  * FLEET_DIM

        obs   = obs_flat.view(B, HISTORY, p_size + f_size)
        p_raw = obs[:, :, :p_size].view(B, HISTORY, MAX_PLANETS, PLANET_DIM)
        f_raw = obs[:, :, p_size:].view(B, HISTORY, MAX_FLEETS,  FLEET_DIM)

        p_emb = self.planet_embed(p_raw)
        f_emb = self.fleet_embed(f_raw)

        t_idx = torch.arange(HISTORY, device=obs_flat.device)

        # Planet temporal
        p_pos = self.planet_temporal_pos(t_idx)
        p_t   = p_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_PLANETS, HISTORY, EMBED_DIM)
        p_t   = p_t + p_pos.unsqueeze(0)
        p_t   = self.planet_temporal_attn(p_t)
        p_t   = p_t[:, -1, :].view(B, MAX_PLANETS, EMBED_DIM)

        # Fleet temporal (optional)
        if FLEET_TEMPORAL:
            f_pos = self.fleet_temporal_pos(t_idx)
            f_t   = f_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_FLEETS, HISTORY, EMBED_DIM)
            f_t   = f_t + f_pos.unsqueeze(0)
            f_t   = self.fleet_temporal_attn(f_t)
            f_t   = f_t[:, -1, :].view(B, MAX_FLEETS, EMBED_DIM)
        else:
            f_t = f_emb[:, -1, :, :]

        # Local (fleet ↔ planet) + Global
        local_tokens = torch.cat([p_t, f_t], dim=1)
        local_out    = self.local_attn(local_tokens)
        p_local      = local_out[:, :MAX_PLANETS, :]
        global_out   = self.global_attn(p_local)

        return self.actor(global_out)
