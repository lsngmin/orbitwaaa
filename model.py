"""
Hierarchical Transformer Policy for Orbit Wars.

구조:
  1. Embedding       — 행성/fleet 토큰 → embed_dim
  2. Temporal Attn   — 과거 N턴 패턴 학습
  3. Local Attn      — fleet ↔ 행성 관계
  4. Global Attn     — 전체 전략
  5. Actor / Critic  — 행동 확률 + 상태 가치
"""

import math
import torch
import torch.nn as nn
import yaml

with open("config.yaml") as f:
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
MAX_PLANETS     = ENV["max_planets"]
MAX_FLEETS      = ENV["max_fleets"]
PLANET_DIM      = 13  # +2: ETA bin(near/mid) 분리
FLEET_DIM       = 7
ACTION_DIM      = MAX_PLANETS + 2  # 발사여부 + ships비율 + 타겟 logits


def make_transformer(layers):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 4,
        dropout=0.1,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class OrbitWarsPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # 입력 임베딩
        self.planet_embed = nn.Linear(PLANET_DIM, EMBED_DIM)
        self.fleet_embed  = nn.Linear(FLEET_DIM,  EMBED_DIM)

        # 위치 인코딩 — planet/fleet 분리
        self.planet_temporal_pos = nn.Embedding(HISTORY, EMBED_DIM)
        self.fleet_temporal_pos  = nn.Embedding(HISTORY, EMBED_DIM)

        # 1. Temporal Attention — planet/fleet 분리
        self.planet_temporal_attn = make_transformer(PLANET_TEMPORAL_LAYERS)
        # ablation B (fleet_temporal=false): fleet_temporal_attn 미사용, current-step만 통과
        self.fleet_temporal_attn  = make_transformer(FLEET_TEMPORAL_LAYERS) if FLEET_TEMPORAL else None

        # 2. Local Attention — fleet ↔ 행성 관계
        self.local_attn = make_transformer(LOCAL_LAYERS)

        # 3. Global Attention — 전체 전략
        self.global_attn = make_transformer(GLOBAL_LAYERS)

        # Actor head — 행성별 행동 출력
        self.actor = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, ACTION_DIM),
        )

        # Critic head — 상태 가치 출력
        self.critic = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, 1),
        )

    def forward(self, obs_flat):
        """
        obs_flat: (B, HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM))
        returns:
          action_logits: (B, MAX_PLANETS, ACTION_DIM)
          value:         (B, 1)
        """
        B = obs_flat.shape[0]
        p_size = MAX_PLANETS * PLANET_DIM
        f_size = MAX_FLEETS  * FLEET_DIM

        # obs 복원: (B, HISTORY, MAX_PLANETS, PLANET_DIM) + (B, HISTORY, MAX_FLEETS, FLEET_DIM)
        obs = obs_flat.view(B, HISTORY, p_size + f_size)
        p_raw = obs[:, :, :p_size].view(B, HISTORY, MAX_PLANETS, PLANET_DIM)
        f_raw = obs[:, :, p_size:].view(B, HISTORY, MAX_FLEETS,  FLEET_DIM)

        # --- 임베딩 ---
        p_emb = self.planet_embed(p_raw)  # (B, H, P, E)
        f_emb = self.fleet_embed(f_raw)   # (B, H, F, E)

        # --- 1. Temporal Attention ---
        t_idx = torch.arange(HISTORY, device=obs_flat.device)

        # planet: 저주파 상태 변화 (점유, 생산, 압박)
        p_pos = self.planet_temporal_pos(t_idx)
        p_t   = p_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_PLANETS, HISTORY, EMBED_DIM)
        p_t   = p_t + p_pos.unsqueeze(0)
        p_t   = self.planet_temporal_attn(p_t)
        p_t   = p_t[:, -1, :].view(B, MAX_PLANETS, EMBED_DIM)

        # fleet: ablation A=temporal encoding, ablation B=current-step only
        if FLEET_TEMPORAL:
            f_pos = self.fleet_temporal_pos(t_idx)
            f_t   = f_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_FLEETS, HISTORY, EMBED_DIM)
            f_t   = f_t + f_pos.unsqueeze(0)
            f_t   = self.fleet_temporal_attn(f_t)
            f_t   = f_t[:, -1, :].view(B, MAX_FLEETS, EMBED_DIM)
        else:
            # current-step fleet embedding만 사용 (마지막 턴)
            f_t = f_emb[:, -1, :, :]  # (B, F, E)

        # --- 2. Local Attention (fleet ↔ 행성) ---
        local_tokens = torch.cat([p_t, f_t], dim=1)  # (B, P+F, E)
        local_out    = self.local_attn(local_tokens)  # (B, P+F, E)
        p_local = local_out[:, :MAX_PLANETS, :]       # (B, P, E)

        # --- 3. Global Attention ---
        global_out = self.global_attn(p_local)        # (B, P, E)

        # --- Actor ---
        action_logits = self.actor(global_out)         # (B, P, ACTION_DIM)

        # --- Critic (전체 평균 풀링) ---
        value = self.critic(global_out.mean(dim=1))    # (B, 1)

        return action_logits, value

    def get_action_and_value(self, obs_flat):
        """PPO 학습용: 행동 샘플링 + log_prob + value."""
        action_logits, value = self.forward(obs_flat)

        # 발사 여부 (Bernoulli)
        launch_logits = action_logits[:, :, 0]
        launch_dist   = torch.distributions.Bernoulli(logits=launch_logits)
        launch        = launch_dist.sample()

        # ships 비율 (Beta distribution을 Normal로 근사)
        ships_mean    = torch.sigmoid(action_logits[:, :, 1])
        ships_dist    = torch.distributions.Normal(ships_mean, 0.1)
        ships_ratio   = ships_dist.sample().clamp(0.0, 1.0)

        # 타겟 선택 (Categorical)
        target_logits = action_logits[:, :, 2:]
        target_dist   = torch.distributions.Categorical(logits=target_logits)
        target        = target_dist.sample()

        log_prob = (
            launch_dist.log_prob(launch).sum(-1)
            + ships_dist.log_prob(ships_ratio).sum(-1)
            + target_dist.log_prob(target).sum(-1)
        )

        # action 합치기: (B, P, P+2)
        target_onehot = torch.zeros(*target_logits.shape, device=obs_flat.device)
        target_onehot.scatter_(-1, target.unsqueeze(-1), 1.0)
        action = torch.cat([
            launch.unsqueeze(-1),
            ships_ratio.unsqueeze(-1),
            target_onehot,
        ], dim=-1)

        return action, log_prob, value

    def evaluate_actions(self, obs_flat, actions):
        """PPO 업데이트용: 주어진 행동의 log_prob + entropy + value."""
        action_logits, value = self.forward(obs_flat)

        launch      = actions[:, :, 0]
        ships_ratio = actions[:, :, 1]
        target      = actions[:, :, 2:].argmax(dim=-1)

        launch_dist = torch.distributions.Bernoulli(logits=action_logits[:, :, 0])
        ships_mean  = torch.sigmoid(action_logits[:, :, 1])
        ships_dist  = torch.distributions.Normal(ships_mean, 0.1)
        target_dist = torch.distributions.Categorical(logits=action_logits[:, :, 2:])

        log_prob = (
            launch_dist.log_prob(launch).sum(-1)
            + ships_dist.log_prob(ships_ratio).sum(-1)
            + target_dist.log_prob(target).sum(-1)
        )

        entropy = (
            launch_dist.entropy().sum(-1)
            + ships_dist.entropy().sum(-1)
            + target_dist.entropy().sum(-1)
        )

        return log_prob, entropy, value
