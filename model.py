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
LOCAL_LAYERS           = M["local_layers"]
GLOBAL_LAYERS          = M["global_layers"]
NUM_HEADS              = M["num_heads"]
HISTORY                = M["temporal_window"]
MAX_PLANETS     = ENV["max_planets"]
MAX_FLEETS      = ENV["max_fleets"]
PLANET_DIM      = 16  # 0-14: numeric features, 15: is_valid (1=real, 0=empty)
PLANET_FEAT_DIM = 15  # planet_embed 입력 dim (is_valid 제외)
FLEET_DIM       = 9   # 0-6: numeric, 7: src_idx, 8: dst_idx
                      # idx 7,8 sentinel: -2=empty slot, -1=lookup miss (real fleet), ≥0=valid
FLEET_FEAT_DIM  = 7   # fleet_embed 입력 dim

# ships head: amount mode 선택. "multiplier" (Phase A) / "surplus" (legacy).
#   multiplier: ships = ceil(required × multiplier_k)
#   surplus   : ships = required + bin × max(0, src - required)
AMOUNT_MODE        = M.get("amount_mode", "multiplier")
SHIPS_SURPLUS_BINS = tuple(M.get("ships_surplus_bins", [0.0, 0.33, 0.66, 1.0]))
SHIPS_MULTIPLIERS  = tuple(M.get("ships_multipliers",  [1.00, 1.05, 1.12, 1.20]))
SHIPS_BINS         = SHIPS_MULTIPLIERS if AMOUNT_MODE == "multiplier" else SHIPS_SURPLUS_BINS
NUM_SHIPS_BINS     = len(SHIPS_BINS)
# 단일 5-way action: idx 0 = skip (발사 안 함), idx 1..K = bin k-1 (발사 + 함선량)
# launch + ships_bin 를 합쳐 정책이 source 별 "발사 보류" 를 직접 sample.
NUM_ACTIONS        = 1 + NUM_SHIPS_BINS

# ── Cost feature dims ────────────────────────────────────────────────────────
# Pair-level (target head): [
#   0: required/src_ships          (linear clip)  — capacity 한계 지표
#   1: required/dst_ships          (linear clip)
#   2: eta_norm                    (linear clip)
#   3: proj_target_ships/src_ships (linear clip)
#   4: log1p(required/src.production)              — 몇 턴어치 생산량을 태우나 (경제 비용)
#   5: log1p(required/(src.production × eta_turns))— 도착 전까지 회복 가능성
#   6: eta_advantage_norm                          — race signal:
#        clip((enemy_nearest_eta_to_dst - my_eta) / 30, -1, 1)
#        > 0: 내가 더 빠름, < 0: 적이 더 빠름
#   7: turn_norm                                   — phase-awareness:
#        current_ep_step / max_episode_steps (0..1)
#        bias_mlp 가 implicit 하게 phase 별 feature 가중 학습
# ]
TARGET_PAIR_FEAT_DIM = 8
# Per-bin (amount head): [
#   0: candidate_ships/src_ships                    (linear clip)
#   1: remaining_after_send/src_ships               (linear clip)
#   2: candidate_ships/required                     (linear clip)
#   3: eta_norm                                     (linear clip)
#   4: log1p(candidate_ships/src.production)        — 후보가 몇 턴어치 생산량
#   5: log1p(candidate_ships/(src.production × eta_turns)) — 후보 도착 전 회복 가능성
#   6: eta_advantage_norm  (pair-level mirror; per-amount ETA 변동은 무시)
#   7: turn_norm           (phase)
# ]
AMOUNT_BIN_FEAT_DIM  = 8

# Action layout: [action_5way_onehot(NUM_ACTIONS), target_onehot(P)]
#   action_5way: 0=skip, 1..K=bin (k-1)
ACTION_DIM      = NUM_ACTIONS + MAX_PLANETS


def make_transformer(layers):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 2,   # 표준 ×4 대신 ×2 — 작은 모델 + 짧은 H=20 에 충분, 파라미터 ↓
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def make_decoder(layers):
    """Cross-attn decoder: planet query 가 [P+F] memory 를 읽음.

    한 층 안에 (planet self-attn) + (planet→[P+F] cross-attn) + FFN.
    """
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 2,
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=layers)


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

        # Gated planet fusion — fleet 가 자기 source/destination planet 표현을 참조.
        # candidate = tanh(Wv [f_t ; planet_t])
        # gate      = sigmoid(Wg [f_t ; planet_t])
        # f_fused   = f_t + gate * candidate    (residual + gated update)
        # Source: 출발지 행성 컨텍스트 (출발지 약화 신호 등).
        # Destination: ray-cast 첫 충돌 행성 컨텍스트 (목적지 방어/위협 직접 참조).
        self.fleet_source_value = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_source_gate  = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_dest_value   = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_dest_gate    = nn.Linear(EMBED_DIM * 2, EMBED_DIM)

        # 위치 인코딩 — planet/fleet 분리
        self.planet_temporal_pos = nn.Embedding(HISTORY, EMBED_DIM)
        self.fleet_temporal_pos  = nn.Embedding(HISTORY, EMBED_DIM)

        # 1. Temporal Attention — planet/fleet 분리
        self.planet_temporal_attn = make_transformer(PLANET_TEMPORAL_LAYERS)
        self.fleet_temporal_attn  = make_transformer(FLEET_TEMPORAL_LAYERS)

        # 1.5 Attention pool — H턴 시퀀스를 학습 가능한 query 로 weighted-sum.
        # 이전: p_t[:, -1, :] (마지막 위치 anchor) → 마지막 턴 input 의 noise 에 민감.
        # 이후: 학습된 query 가 "어떤 턴이 결정적인지" 자율적 가중. 마지막 턴이 transient
        # (fleet 모두 도착 등) 일 때 자동으로 과거 턴 가중치 ↑.
        self.planet_pool_query = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        self.planet_pool_attn  = nn.MultiheadAttention(
            EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True
        )
        self.fleet_pool_query = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        self.fleet_pool_attn  = nn.MultiheadAttention(
            EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True
        )

        # 2. Local Attention — fleet ↔ 행성 관계
        self.local_attn = make_transformer(LOCAL_LAYERS)

        # 3. Global Attention — planet query 가 [P+F] memory 를 cross-attn 으로 읽음.
        # 기존 P-only self-attn 의 정보 병목 (함대가 local 후 사라짐) 해소:
        # local 4층으로 P+F 관계 깊게 섞고, global decoder 가 마지막에 P 가 함대까지
        # 한 번 더 참조해 의사결정 표현 만듦.
        self.global_attn = make_decoder(GLOBAL_LAYERS)

        # Actor heads — sequential factorization (Step 4: target pair-aware):
        #   target_i  ~ Categorical(q(H_i) · k(H_j) / sqrt(E))              # (B, P, P)
        #   amount_i  ~ Categorical(amount_pair_head(H_i, H_target_i))      # 5-way
        # target/amount 모두 (src, dst) 를 명시적으로 점수화 — Step 3 의 비대칭
        #   target: f(H_i)_j   ← P-way classifier (index prior 만 학습 가능)
        #   amount: g(H_i, H_j) ← pair-aware
        # 을 해소. target_logit[i,j] = score(source i, destination j) 로 dst 의 표현
        # 을 직접 본다. Q/K 분리 — source/destination 방향성 고려.
        self.target_q = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.target_k = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.amount_pair_head = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, NUM_ACTIONS),
        )

        # ── Phase A: explicit cost-feature bias channels ─────────────────────
        # rationale: surplus-fraction plateau (gen 32, 40-gen) 는 모델이 required 를
        # 명시적으로 모르는 상태에서 src/dst 의 토큰 표현만으로 "얼마 보낼지" 를
        # 학습해야 했기 때문. required/src_ships, eta 같은 명시 지표를 head 에
        # 직접 주면 학습 신호가 농축됨.
        #
        # Step 4 parity: alpha = nn.Parameter(zeros(1)) 으로 초기화 → 학습 시작
        # 시점에 bias 출력이 정확히 0. 즉 t_logits = q·k/√E 그대로, a_logits
        # = amount_pair_head(...) 그대로. Step 4 verification 가드 통과.
        self.target_bias_mlp = nn.Sequential(
            nn.Linear(TARGET_PAIR_FEAT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # target_bias_alpha: 0.1 init (1차 5gen 진단 결과 0 init 은 P=44 way softmax
        # gradient dilution 으로 ±0.014 진동에서 못 자람 → q·k/√E logit 에 묻힘.
        # 0.1 으로 강제 시동하면 bias_mlp 가 받는 reward gradient 가 즉시 chain rule
        # 통해 커짐. 자연스레 자라는 amount_bin_alpha 와 비대칭 해소.
        self.target_bias_alpha = nn.Parameter(torch.full((1,), 0.1))
        # amount: per-bin scalar bias (skip(idx 0) 은 candidate 가 없으므로 0 유지).
        # amount_bin_alpha 는 5gen 동안 양수 단조 자람 (-0.0018 → +0.0031) — 일하는
        # 중이라 0 init 유지. target 쪽만 시동 거는 게 ablation 깨끗.
        self.amount_bin_bias_mlp = nn.Sequential(
            nn.Linear(AMOUNT_BIN_FEAT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.amount_bin_alpha = nn.Parameter(torch.zeros(1))

        # Critic head — 상태 가치 출력
        self.critic = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, 1),
        )

    def _gated_planet_fuse(self, f_t, p_t, planet_pad_now, idx_raw,
                            value_layer, gate_layer):
        """Gated residual fusion — fleet 가 지정 planet idx 의 표현을 참조.

        invalid 슬롯은 fusion 차단 (residual identity). source/destination 공유 logic,
        호출자가 (value_layer, gate_layer) 로 분리.

        valid 정의 (Hole A — 강건화):
          - idx ∈ [0, MAX_PLANETS) — 범위
          - planet_pad_now[idx] == False — gather 대상 슬롯이 padded 가 아님.
          현재 흐름에선 id_to_idx 가 padded slot 으로 안 향하므로 (3) 은 redundant
          하지만, encoder 변경 시 silent leak 방지용 1차 회귀 가드.

        compute optimization (Hole D):
          invalid 슬롯도 통과시키면 B*F 의 ~95% (대다수 empty) 가 무의미 matmul.
          valid 슬롯만 골라 W_g/W_v 통과 → scatter back. valid 가 비면 자연스럽게
          (0,2E) → (0,E) 로 흘러 추가 분기 없음.

        Args:
          f_t            : (B, F, E)
          p_t            : (B, P, E)
          planet_pad_now : (B, P) bool — 현재 step 의 planet padding mask
          idx_raw        : (B, F) — float; gather pointer (cast 전)
        Returns:
          (B, F, E) — f_t + (valid 슬롯에만) gate * tanh(Wv [f_t; planet])
        """
        B, F, E    = f_t.shape
        idx        = idx_raw.long()
        idx_safe   = idx.clamp(0, MAX_PLANETS - 1)
        # gathered_pad: 각 fleet 슬롯이 가리키는 planet 슬롯의 padding 여부
        gathered_pad = planet_pad_now.gather(1, idx_safe)                    # (B, F)
        valid        = (idx >= 0) & (idx < MAX_PLANETS) & ~gathered_pad      # (B, F)

        gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, E)
        planet_t   = p_t.gather(1, gather_idx)                                # (B, F, E)

        # subset valid 슬롯만 matmul (compute 절감)
        flat_valid = valid.flatten()                                          # (B*F,)
        f_flat     = f_t.reshape(-1, E)                                       # (B*F, E)
        src_flat   = planet_t.reshape(-1, E)                                  # (B*F, E)

        f_sub      = f_flat[flat_valid]                                       # (N, E)
        src_sub    = src_flat[flat_valid]                                     # (N, E)

        fused_in   = torch.cat([f_sub, src_sub], dim=-1)                      # (N, 2E)
        gate       = torch.sigmoid(gate_layer(fused_in))                      # (N, E)
        cand       = torch.tanh(value_layer(fused_in))                        # (N, E)
        update_sub = gate * cand                                              # (N, E)

        update_flat = torch.zeros_like(f_flat)
        update_flat[flat_valid] = update_sub
        return f_t + update_flat.view(B, F, E)

    def forward(self, obs_flat):
        """
        obs_flat: (B, HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM))
        returns:
          src_token: (B, MAX_PLANETS, EMBED_DIM) — encoder output (= global_out)
          value:     (B, 1)
        head 적용은 get_action_and_value / evaluate_actions 가 담당 (target → amount
        sequential conditional sampling 때문에 forward 단에서 amount logits 를 미리
        못 계산함 — target 샘플이 필요).
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
        #   fleet : idx 7 = src_idx, idx 8 = dst_idx
        p_features = p_raw[..., :PLANET_FEAT_DIM]               # (B, H, P, 15)
        f_features = f_raw[..., :FLEET_FEAT_DIM]                # (B, H, F, 7)
        fp_idx_raw = f_raw[:, -1, :, 7]                          # (B, F) — src_idx
        dp_idx_raw = f_raw[:, -1, :, 8]                          # (B, F) — dst_idx

        # --- Padding masks (sentinel 기반, 명시적) ---
        # fleet: idx 7,8 모두 -2 = empty slot. -1 = real fleet w/ lookup miss
        # (위치/속도/ships 다 valid → mask 안 함, fusion 만 차단됨)
        planet_pad_h = (p_raw[..., -1] == 0)                     # (B, H, P)
        fleet_pad_h  = (f_raw[..., 7] == -2)                     # (B, H, F)
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
        # attention pool: 학습된 query 가 H턴을 weighted-sum (마지막 위치 anchor 대체)
        p_q   = self.planet_pool_query.expand(B * MAX_PLANETS, 1, EMBED_DIM)
        p_t, _ = self.planet_pool_attn(p_q, p_t, p_t,
                                        key_padding_mask=_safe_pad_mask(p_pad_seq))
        p_t   = p_t.squeeze(1).view(B, MAX_PLANETS, EMBED_DIM)

        # fleet: planet 과 동일한 temporal self-attn → pool 경로
        f_pos = self.fleet_temporal_pos(t_idx)
        f_t   = f_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_FLEETS, HISTORY, EMBED_DIM)
        f_t   = f_t + f_pos.unsqueeze(0)
        f_pad_seq = fleet_pad_h.permute(0, 2, 1).contiguous().view(B * MAX_FLEETS, HISTORY)
        f_t   = self.fleet_temporal_attn(f_t, src_key_padding_mask=_safe_pad_mask(f_pad_seq))
        f_q   = self.fleet_pool_query.expand(B * MAX_FLEETS, 1, EMBED_DIM)
        f_t, _ = self.fleet_pool_attn(f_q, f_t, f_t,
                                       key_padding_mask=_safe_pad_mask(f_pad_seq))
        f_t   = f_t.squeeze(1).view(B, MAX_FLEETS, EMBED_DIM)

        # --- 1.5 Planet fusion — fleet 가 source/destination planet 표현 참조 ---
        f_t = self._gated_planet_fuse(f_t, p_t, planet_pad_now, fp_idx_raw,
                                       self.fleet_source_value, self.fleet_source_gate)
        f_t = self._gated_planet_fuse(f_t, p_t, planet_pad_now, dp_idx_raw,
                                       self.fleet_dest_value,   self.fleet_dest_gate)

        # --- 2. Local Attention (fleet ↔ 행성) ---
        local_tokens = torch.cat([p_t, f_t], dim=1)               # (B, P+F, E)
        local_pad    = torch.cat([planet_pad_now, fleet_pad_now], dim=1)
        local_out    = self.local_attn(local_tokens, src_key_padding_mask=_safe_pad_mask(local_pad))
        p_query = local_out[:, :MAX_PLANETS, :]                   # (B, P, E) — decoder query

        # --- 3. Global Attention (cross-attn: P → [P+F]) ---
        # tgt    = planet local output (B, P, E)
        # memory = full local output  (B, P+F, E)  — 함대 정보 유지
        global_out = self.global_attn(
            tgt=p_query,
            memory=local_out,
            tgt_key_padding_mask=_safe_pad_mask(planet_pad_now),
            memory_key_padding_mask=_safe_pad_mask(local_pad),
        )

        # --- Critic (masked mean 풀링) ---
        # 빈 행성 토큰을 평균에서 배제 — 후반 P_alive ≪ MAX_PLANETS 일 때
        # 분모를 P 로 고정하면 V 가 인위적으로 줄어들어 advantage 가 부풀어 오름.
        valid_p     = (~planet_pad_now).to(global_out.dtype).unsqueeze(-1)   # (B, P, 1)
        valid_sum   = (global_out * valid_p).sum(dim=1)                       # (B, E)
        valid_count = valid_p.sum(dim=1).clamp(min=1.0)                       # (B, 1)
        value       = self.critic(valid_sum / valid_count)                    # (B, 1)

        return global_out, value

    def _gather_amount_features(self, amount_features, target_idx):
        """Shape-detect helper: full (B,P,P,K,F) → gathered (B,P,K,F) by target_idx.

        4D 입력은 이미 gather 된 것으로 보고 pass-through.
        None 은 None.

        Args:
          amount_features: (B, P, P, K, F) or (B, P, K, F) or None
          target_idx     : (B, P) long
        """
        if amount_features is None:
            return None
        if amount_features.dim() == 4:
            return amount_features
        # 5D: gather by chosen target
        B, P, _, K, F = amount_features.shape
        idx_safe = target_idx.clamp(0, P - 1)
        gather_idx = idx_safe.view(B, P, 1, 1, 1).expand(-1, -1, 1, K, F)
        return amount_features.gather(2, gather_idx).squeeze(2)             # (B, P, K, F)

    def _target_logits(self, src_token, pair_features=None):
        """Pair-aware target head (Step 4 + Phase A cost bias).

        Step 4:    t_logits[i,j] = q(H_i)·k(H_j)/sqrt(E)
        Phase A:   t_logits[i,j] += α · target_bias_mlp(pair_features[i,j])

        pair_features=None 또는 α=0 (init) 이면 정확히 Step 4 결과와 동일 →
        Step-4 체크포인트 transfer 시 보존된 q·k 표현을 망치지 않는다.

        Args:
          src_token    : (B, P, E)
          pair_features: (B, P, P, TARGET_PAIR_FEAT_DIM) or None
                         (required/src, required/dst, eta_norm, proj_target/src)
        Returns:
          (B, P, P)
        """
        q = self.target_q(src_token)                                    # (B, P, E)
        k = self.target_k(src_token)                                    # (B, P, E)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (EMBED_DIM ** 0.5)
        if pair_features is not None:
            bias = self.target_bias_mlp(pair_features).squeeze(-1)      # (B, P, P)
            logits = logits + self.target_bias_alpha * bias
        return logits

    def _amount_logits(self, src_token, target_idx, amount_features=None):
        """Pair-wise amount head + per-bin cost bias (Phase A).

        base:    a_logits[i, k] = amount_pair_head([H_i, H_target_i])[k]
        Phase A: a_logits[i, 1+k] += α · amount_bin_bias_mlp(amount_features[i, k])
                 (skip(idx 0) 은 candidate 가 없으므로 bias 없음 — 0 유지)

        amount_features=None 또는 α=0 (init) → 정확히 base 결과 유지.

        Args:
          src_token      : (B, P, E)
          target_idx     : (B, P) long — 각 source 가 가리키는 target index
          amount_features: (B, P, NUM_SHIPS_BINS, AMOUNT_BIN_FEAT_DIM) or None
                           각 (src, chosen_dst, bin_k) 에 대한 candidate 지표.
        Returns:
          (B, P, NUM_ACTIONS)  — idx 0 = skip
        """
        B, P, E = src_token.shape
        idx_safe   = target_idx.clamp(0, P - 1)
        gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, E)
        dst_token  = src_token.gather(1, gather_idx)                    # (B, P, E)
        pair_input = torch.cat([src_token, dst_token], dim=-1)          # (B, P, 2E)
        logits     = self.amount_pair_head(pair_input)                  # (B, P, NUM_ACTIONS)
        if amount_features is not None:
            # bias_per_bin: (B, P, NUM_SHIPS_BINS)
            bias_per_bin = self.amount_bin_bias_mlp(amount_features).squeeze(-1)
            # skip dim 앞에 0 prepend → (B, P, NUM_ACTIONS)
            zero_skip = torch.zeros_like(bias_per_bin[..., :1])
            bias_full = torch.cat([zero_skip, bias_per_bin], dim=-1)
            logits = logits + self.amount_bin_alpha * bias_full
        return logits

    def get_action_and_value(self, obs_flat, action_mask=None, target_mask=None,
                              target_pair_features=None, amount_features=None):
        """PPO 학습용: target → amount sequential conditional sampling.

        Sampling order (Step 4 + Phase A cost bias):
          1. target_i ~ Categorical(q·k/√E + α_t · target_bias_mlp(pair_feats))     # masked
          2. amount_i ~ Categorical(amount_pair_head + α_a · amount_bin_bias_mlp)   # masked

        action_mask         : (B, P, NUM_ACTIONS) bool
        target_mask         : (B, P, P) bool
        target_pair_features: (B, P, P, TARGET_PAIR_FEAT_DIM) or None
        amount_features     : (B, P, NUM_SHIPS_BINS, AMOUNT_BIN_FEAT_DIM) or None

        log_prob factorization (skip 흡수 방식):
          log p(action) = log p(amount | src, target) + is_launch · log p(target | src)

        Returns: action, log_prob, value, lp_heads (action layout = [a_onehot(K), t_onehot(P)])
        """
        src_token, value = self.forward(obs_flat)

        # Step 1: target sample
        t_logits = self._target_logits(src_token, target_pair_features)  # (B, P, P)
        if target_mask is not None:
            t_logits = t_logits.masked_fill(~target_mask, -1e9)
        t_dist = torch.distributions.Categorical(logits=t_logits)
        target = t_dist.sample()                                         # (B, P) long

        # Step 2: amount sample (conditional on target).
        # amount_features 는 (B, P, P, K, F) 인 경우 chosen target 으로 gather,
        # 또는 호출자가 미리 (B, P, K, F) 로 만들어 와도 됨. 여기서는 전자/후자 둘 다 수용.
        a_features = self._gather_amount_features(amount_features, target)
        a_logits = self._amount_logits(src_token, target, a_features)
        if action_mask is not None:
            a_logits = a_logits.masked_fill(~action_mask, -1e9)
        a_dist = torch.distributions.Categorical(logits=a_logits)
        a_idx  = a_dist.sample()                                         # (B, P) long, 0=skip

        is_launch = (a_idx > 0).float()
        lp_action = a_dist.log_prob(a_idx).sum(-1)
        lp_target = (t_dist.log_prob(target) * is_launch).sum(-1)
        log_prob  = lp_action + lp_target

        # action 합치기: (B, P, NUM_ACTIONS + P) — env_wrapper / decode 와 호환.
        a_onehot = torch.zeros(*a_logits.shape, device=obs_flat.device)
        a_onehot.scatter_(-1, a_idx.unsqueeze(-1), 1.0)
        t_onehot = torch.zeros(*t_logits.shape, device=obs_flat.device)
        t_onehot.scatter_(-1, target.unsqueeze(-1), 1.0)
        action = torch.cat([a_onehot, t_onehot], dim=-1)

        lp_heads = torch.stack([lp_action, lp_target], dim=-1)
        return action, log_prob, value, lp_heads

    def evaluate_actions(self, obs_flat, actions, action_mask=None, target_mask=None,
                          target_pair_features=None, amount_features=None):
        """PPO 업데이트용: 저장된 (target, amount) 의 log_prob + entropy + value.

        rollout 시와 동일 마스크 + 동일 cost feature 사용해야 importance ratio 유효.
        amount_features 는 rollout 에서 이미 gather 된 (B, P, K, F) 를 받는다
        (full (B,P,P,K,F) 도 받지만 _gather_amount_features 가 알아서 처리).
        """
        src_token, value = self.forward(obs_flat)

        a_idx  = actions[..., :NUM_ACTIONS].argmax(dim=-1)               # (B, P)
        target = actions[..., NUM_ACTIONS:].argmax(dim=-1)               # (B, P)

        t_logits = self._target_logits(src_token, target_pair_features)
        if target_mask is not None:
            t_logits = t_logits.masked_fill(~target_mask, -1e9)
        t_dist = torch.distributions.Categorical(logits=t_logits)

        a_features = self._gather_amount_features(amount_features, target)
        a_logits   = self._amount_logits(src_token, target, a_features)  # 저장된 target 으로 pair
        if action_mask is not None:
            a_logits = a_logits.masked_fill(~action_mask, -1e9)
        a_dist = torch.distributions.Categorical(logits=a_logits)

        is_launch = (a_idx > 0).float()
        lp_action = a_dist.log_prob(a_idx).sum(-1)
        lp_target = (t_dist.log_prob(target) * is_launch).sum(-1)
        log_prob  = lp_action + lp_target

        ent_action = a_dist.entropy().sum(-1)
        ent_target = (t_dist.entropy() * is_launch).sum(-1)
        entropy    = ent_action + ent_target

        lp_heads = torch.stack([lp_action, lp_target], dim=-1)
        return log_prob, entropy, value, ent_action, ent_target, lp_heads
