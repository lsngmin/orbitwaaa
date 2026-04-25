"""
Hierarchical Transformer Policy for Orbit Wars.

구조:
  1. Embedding       — 행성/fleet 토큰 → embed_dim
  2. Temporal Attn   — 과거 N턴 패턴 학습
  3. Local Attn      — fleet ↔ 행성 관계
  4. Global Attn     — 전체 전략
  5. Actor / Critic  — 행동 확률 + 상태 가치
"""

import os
import math
import torch
import torch.nn as nn
import yaml

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path) as f:
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
PLANET_DIM      = 16  # 0-14: numeric features, 15: is_valid (1=real, 0=empty)
PLANET_FEAT_DIM = 15  # planet_embed 입력 dim (is_valid 제외)
FLEET_DIM       = 8   # 0-6: numeric features, 7: src_idx (-2=empty slot, -1=src lookup miss, ≥0=valid)
FLEET_FEAT_DIM  = 7   # fleet_embed 입력 dim

# ships head: required_ships 배수 Categorical (commit 2)
# decode: ships_to_send = min(int(required × multiplier), src.ships)
SHIPS_MULTIPLIER_BINS = tuple(M.get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
NUM_SHIPS_BINS        = len(SHIPS_MULTIPLIER_BINS)

# Action layout: [launch(1), ships_bin_onehot(K), target_onehot(P)]
ACTION_DIM      = 1 + NUM_SHIPS_BINS + MAX_PLANETS


def make_transformer(layers):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 4,
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def _safe_pad_mask(pad_mask):
    """src_key_padding_mask 안전화: 시퀀스 전체가 padding 인 경우 NaN 회피.

    nn.MultiheadAttention 는 모든 key 가 masked 면 softmax NaN. 빈 행성/슬롯은
    어차피 다운스트림에서 valid mask 로 무시되므로, all-masked 행은 mask 를
    전부 False 로 덮어 attention 출력을 0-input 그대로 통과시킨다.
    """
    fully = pad_mask.all(dim=-1, keepdim=True)
    return pad_mask & ~fully


class OrbitWarsPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # 입력 임베딩
        self.planet_embed = nn.Linear(PLANET_FEAT_DIM, EMBED_DIM)
        self.fleet_embed  = nn.Linear(FLEET_FEAT_DIM,  EMBED_DIM)

        # Gated source-planet fusion — fleet 가 자기 source planet 의 표현을 참조.
        # candidate = tanh(Wv [f_t ; src_t])
        # gate      = sigmoid(Wg [f_t ; src_t])
        # f_fused   = f_t + gate * candidate    (residual + gated update)
        self.fleet_source_value = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_source_gate  = nn.Linear(EMBED_DIM * 2, EMBED_DIM)

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

    def _fleet_source_fuse(self, f_t, p_t, fp_idx_raw):
        """Gated residual fusion — fleet 가 source planet 표현을 참조.

        invalid(-1)/out-of-range 는 valid mask 로 fusion 차단 → residual identity.

        Args:
          f_t        : (B, F, E)
          p_t        : (B, P, E)
          fp_idx_raw : (B, F) — float; 마지막 dim 그대로 (cast 전)
        Returns:
          (B, F, E)  — f_t + gate * tanh(Wv [f_t;src]) * valid
        """
        fp_idx      = fp_idx_raw.long()
        valid       = (fp_idx >= 0) & (fp_idx < MAX_PLANETS)
        fp_idx_safe = fp_idx.clamp(0, MAX_PLANETS - 1)
        gather_idx  = fp_idx_safe.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
        src_t       = p_t.gather(1, gather_idx)

        fused_in = torch.cat([f_t, src_t], dim=-1)
        gate     = torch.sigmoid(self.fleet_source_gate(fused_in))
        cand     = torch.tanh(self.fleet_source_value(fused_in))
        valid_f  = valid.unsqueeze(-1).to(f_t.dtype)
        return f_t + gate * cand * valid_f

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

        # 마지막 dim 은 sentinel — embed 입력에서 분리.
        #   planet: idx 15 = is_valid (1=real, 0=empty)
        #   fleet : idx 7  = from_planet_idx (-1=invalid)
        p_features = p_raw[..., :PLANET_FEAT_DIM]               # (B, H, P, 15)
        f_features = f_raw[..., :FLEET_FEAT_DIM]                # (B, H, F, 7)
        fp_idx_raw = f_raw[:, -1, :, -1]                        # (B, F) — 현재 step idx

        # --- Padding masks (sentinel 기반, 명시적) ---
        # fleet: -2 = empty slot (mask 대상). -1 = real fleet 인데 src lookup miss
        # (위치/속도/ships 다 valid → mask 안 함, fusion 만 차단됨)
        planet_pad_h = (p_raw[..., -1] == 0)                     # (B, H, P)
        fleet_pad_h  = (f_raw[..., -1] == -2)                    # (B, H, F)
        planet_pad_now = planet_pad_h[:, -1, :]                  # (B, P)
        fleet_pad_now  = fleet_pad_h[:, -1, :]                   # (B, F)

        # --- 임베딩 ---
        p_emb = self.planet_embed(p_features)  # (B, H, P, E)
        f_emb = self.fleet_embed(f_features)   # (B, H, F, E)

        # --- 1. Temporal Attention ---
        t_idx = torch.arange(HISTORY, device=obs_flat.device)

        # planet: 저주파 상태 변화 (점유, 생산, 압박)
        p_pos = self.planet_temporal_pos(t_idx)
        p_t   = p_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_PLANETS, HISTORY, EMBED_DIM)
        p_t   = p_t + p_pos.unsqueeze(0)
        p_pad_seq = planet_pad_h.permute(0, 2, 1).contiguous().view(B * MAX_PLANETS, HISTORY)
        p_t   = self.planet_temporal_attn(p_t, src_key_padding_mask=_safe_pad_mask(p_pad_seq))
        p_t   = p_t[:, -1, :].view(B, MAX_PLANETS, EMBED_DIM)

        # fleet: ablation A=temporal encoding, ablation B=current-step only
        if FLEET_TEMPORAL:
            f_pos = self.fleet_temporal_pos(t_idx)
            f_t   = f_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_FLEETS, HISTORY, EMBED_DIM)
            f_t   = f_t + f_pos.unsqueeze(0)
            f_pad_seq = fleet_pad_h.permute(0, 2, 1).contiguous().view(B * MAX_FLEETS, HISTORY)
            f_t   = self.fleet_temporal_attn(f_t, src_key_padding_mask=_safe_pad_mask(f_pad_seq))
            f_t   = f_t[:, -1, :].view(B, MAX_FLEETS, EMBED_DIM)
        else:
            # current-step fleet embedding만 사용 (마지막 턴)
            f_t = f_emb[:, -1, :, :]  # (B, F, E)

        # --- 1.5 Source planet fusion — fleet 가 자기 source planet 표현 참조 ---
        f_t = self._fleet_source_fuse(f_t, p_t, fp_idx_raw)

        # --- 2. Local Attention (fleet ↔ 행성) ---
        local_tokens = torch.cat([p_t, f_t], dim=1)               # (B, P+F, E)
        local_pad    = torch.cat([planet_pad_now, fleet_pad_now], dim=1)
        local_out    = self.local_attn(local_tokens, src_key_padding_mask=_safe_pad_mask(local_pad))
        p_local = local_out[:, :MAX_PLANETS, :]       # (B, P, E)

        # --- 3. Global Attention ---
        global_out = self.global_attn(p_local, src_key_padding_mask=_safe_pad_mask(planet_pad_now))

        # --- Actor ---
        action_logits = self.actor(global_out)         # (B, P, ACTION_DIM)

        # --- Critic (masked mean 풀링) ---
        # 빈 행성 토큰을 평균에서 배제 — 후반 P_alive ≪ MAX_PLANETS 일 때
        # 분모를 P 로 고정하면 V 가 인위적으로 줄어들어 advantage 가 부풀어 오름.
        valid_p     = (~planet_pad_now).to(global_out.dtype).unsqueeze(-1)   # (B, P, 1)
        valid_sum   = (global_out * valid_p).sum(dim=1)                       # (B, E)
        valid_count = valid_p.sum(dim=1).clamp(min=1.0)                       # (B, 1)
        value       = self.critic(valid_sum / valid_count)                    # (B, 1)

        return action_logits, value

    def _split_logits(self, action_logits):
        """action_logits: (B, P, ACTION_DIM) → (launch, ships_bin, target) 분리.

        Layout: [launch(1), ships_bin(K), target(P)]
        """
        launch_logits    = action_logits[..., 0]
        ships_bin_logits = action_logits[..., 1:1 + NUM_SHIPS_BINS]
        target_logits    = action_logits[..., 1 + NUM_SHIPS_BINS:]
        return launch_logits, ships_bin_logits, target_logits

    def get_action_and_value(self, obs_flat, launch_mask=None, target_mask=None,
                              ships_bin_mask=None):
        """PPO 학습용: 행동 샘플링 + log_prob + value.

        launch_mask: (B, P) bool — False 위치의 launch=1 확률을 0으로 강제.
        target_mask: (B, P, P) bool — False 위치의 target 선택 확률을 0으로 강제.
        ships_bin_mask: (B, P, K) bool — False 위치의 ships_bin 확률을 0으로 강제.
                        None이면 전체 허용 (기본값).
        launch_active로 ships/target log_prob 게이팅.
        """
        action_logits, value = self.forward(obs_flat)
        launch_logits, ships_bin_logits, target_logits = self._split_logits(action_logits)

        # 발사 여부 (Bernoulli) — mask 적용
        if launch_mask is not None:
            launch_logits = launch_logits.masked_fill(~launch_mask, -1e9)
        launch_dist   = torch.distributions.Bernoulli(logits=launch_logits)
        launch        = launch_dist.sample()

        # ships 배수 (Categorical over K bins) — mask 적용
        if ships_bin_mask is not None:
            ships_bin_logits = ships_bin_logits.masked_fill(~ships_bin_mask, -1e9)
        ships_dist    = torch.distributions.Categorical(logits=ships_bin_logits)
        ships_bin     = ships_dist.sample()          # (B, P) long

        # 타겟 선택 (Categorical) — mask 적용
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask, -1e9)
        target_dist   = torch.distributions.Categorical(logits=target_logits)
        target        = target_dist.sample()

        lp_launch = launch_dist.log_prob(launch).sum(-1)
        lp_ships  = (ships_dist.log_prob(ships_bin) * launch).sum(-1)
        lp_target = (target_dist.log_prob(target) * launch).sum(-1)
        log_prob  = lp_launch + lp_ships + lp_target

        # action 합치기: (B, P, 1 + K + P)
        ships_onehot  = torch.zeros(*ships_bin_logits.shape, device=obs_flat.device)
        ships_onehot.scatter_(-1, ships_bin.unsqueeze(-1), 1.0)
        target_onehot = torch.zeros(*target_logits.shape, device=obs_flat.device)
        target_onehot.scatter_(-1, target.unsqueeze(-1), 1.0)
        action = torch.cat([
            launch.unsqueeze(-1),
            ships_onehot,
            target_onehot,
        ], dim=-1)

        lp_heads = torch.stack([lp_launch, lp_ships, lp_target], dim=-1)
        return action, log_prob, value, lp_heads

    def evaluate_actions(self, obs_flat, actions, launch_mask=None, target_mask=None,
                          ships_bin_mask=None):
        """PPO 업데이트용: 주어진 행동의 log_prob + entropy + value.

        rollout 시 사용한 것과 동일한 mask를 전달해야 importance ratio가 유효함.
        """
        action_logits, value = self.forward(obs_flat)
        launch_logits, ships_bin_logits, target_logits = self._split_logits(action_logits)

        launch    = actions[..., 0]
        ships_bin = actions[..., 1:1 + NUM_SHIPS_BINS].argmax(dim=-1)
        target    = actions[..., 1 + NUM_SHIPS_BINS:].argmax(dim=-1)

        if launch_mask is not None:
            launch_logits = launch_logits.masked_fill(~launch_mask, -1e9)
        launch_dist = torch.distributions.Bernoulli(logits=launch_logits)

        if ships_bin_mask is not None:
            ships_bin_logits = ships_bin_logits.masked_fill(~ships_bin_mask, -1e9)
        ships_dist  = torch.distributions.Categorical(logits=ships_bin_logits)

        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask, -1e9)
        target_dist = torch.distributions.Categorical(logits=target_logits)

        lp_launch = launch_dist.log_prob(launch).sum(-1)
        lp_ships  = (ships_dist.log_prob(ships_bin) * launch).sum(-1)
        lp_target = (target_dist.log_prob(target) * launch).sum(-1)
        log_prob  = lp_launch + lp_ships + lp_target

        ent_launch = launch_dist.entropy().sum(-1)
        ent_ships  = (ships_dist.entropy() * launch).sum(-1)
        ent_target = (target_dist.entropy() * launch).sum(-1)
        entropy    = ent_launch + ent_ships + ent_target

        lp_heads = torch.stack([lp_launch, lp_ships, lp_target], dim=-1)
        return log_prob, entropy, value, ent_launch, ent_ships, ent_target, lp_heads
