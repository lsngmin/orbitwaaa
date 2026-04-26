"""
Actor-only Orbit Wars model for Kaggle submission.

학습용 OrbitWarsPolicy(model.py)와 *동일* 파라미터 키를 갖는 actor 서브셋.
critic head 만 제외 — state_dict load 시 critic.* 키는 strict=False 로 unexpected 처리.

학습-제출 parity 보장:
  - 동일 config.yaml 읽음 (embed_dim, layers, NUM_HEADS, HISTORY, bins)
  - 동일 forward 경로 (Linear → Temporal → Local → Global → actor head)
  - PLANET_DIM=15 (submission_features 와 일치)

Inference only — torch.no_grad 는 caller 가 감싸도 되고 여기 내부에서도 안전.
"""

import os
import torch
import torch.nn as nn
import yaml

# config.yaml 위치:
#   - Kaggle 번들 (flat layout): submission_actor.py 옆
#   - repo 소스 (submission/ 하위): repo root (한 단계 위)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "config.yaml"), os.path.join(_HERE, "..", "config.yaml")):
    if os.path.exists(_p):
        _CFG_PATH = os.path.abspath(_p)
        break
else:
    raise FileNotFoundError(f"config.yaml not found near {_HERE}")
with open(_CFG_PATH) as f:
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
MAX_PLANETS            = ENV["max_planets"]
MAX_FLEETS             = ENV["max_fleets"]
PLANET_DIM             = 16   # 0-14: numeric features, 15: is_valid (submission_features 동기화)
PLANET_FEAT_DIM        = 15
FLEET_DIM              = 9   # 0-6: numeric, 7: src_idx, 8: dst_idx
                              # idx 7,8 sentinel: -2=empty slot, -1=lookup miss (real fleet), ≥0=valid
FLEET_FEAT_DIM         = 7

SHIPS_SURPLUS_BINS = tuple(M.get("ships_surplus_bins", [0.0, 0.33, 0.66, 1.0]))
NUM_SHIPS_BINS     = len(SHIPS_SURPLUS_BINS)
# 5-way amount action: idx 0 = skip, 1..K = bin (k-1). model.py 와 동기.
NUM_ACTIONS        = 1 + NUM_SHIPS_BINS
# Action layout (env_wrapper / decode 호환): [a_onehot(NUM_ACTIONS), t_onehot(P)]
ACTION_DIM         = NUM_ACTIONS + MAX_PLANETS


def _make_transformer(layers):
    enc = nn.TransformerEncoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 2,   # model.py 와 동기 (표준 ×4 → ×2)
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerEncoder(enc, num_layers=layers)


def _make_decoder(layers):
    """model.make_decoder 미러 — planet query → [P+F] memory cross-attn."""
    dec = nn.TransformerDecoderLayer(
        d_model=EMBED_DIM,
        nhead=NUM_HEADS,
        dim_feedforward=EMBED_DIM * 2,
        dropout=0.0,
        batch_first=True,
    )
    return nn.TransformerDecoder(dec, num_layers=layers)


def _safe_pad_mask(pad_mask):
    """model._safe_pad_mask 미러 — all-masked 행 NaN 회피."""
    fully = pad_mask.all(dim=-1, keepdim=True)
    return pad_mask & ~fully


class OrbitWarsActor(nn.Module):
    """OrbitWarsPolicy 의 actor-only 버전.

    nn.Module key layout 이 학습 모델과 완전히 같도록 구성:
      planet_embed / fleet_embed
      planet_temporal_pos / fleet_temporal_pos
      planet_temporal_attn / fleet_temporal_attn
      local_attn / global_attn
      actor  (critic 은 제외)
    """

    def __init__(self):
        super().__init__()

        self.planet_embed = nn.Linear(PLANET_FEAT_DIM, EMBED_DIM)
        self.fleet_embed  = nn.Linear(FLEET_FEAT_DIM,  EMBED_DIM)

        # Gated planet fusion — source + destination (model.py 와 키 동일)
        self.fleet_source_value = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_source_gate  = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_dest_value   = nn.Linear(EMBED_DIM * 2, EMBED_DIM)
        self.fleet_dest_gate    = nn.Linear(EMBED_DIM * 2, EMBED_DIM)

        self.planet_temporal_pos = nn.Embedding(HISTORY, EMBED_DIM)
        self.fleet_temporal_pos  = nn.Embedding(HISTORY, EMBED_DIM)

        self.planet_temporal_attn = _make_transformer(PLANET_TEMPORAL_LAYERS)
        self.fleet_temporal_attn  = _make_transformer(FLEET_TEMPORAL_LAYERS)

        # Attention pool — H턴 시퀀스를 학습된 query 로 weighted-sum (model.py 와 동기)
        self.planet_pool_query = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        self.planet_pool_attn  = nn.MultiheadAttention(
            EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True
        )
        self.fleet_pool_query = nn.Parameter(torch.randn(1, 1, EMBED_DIM) * 0.02)
        self.fleet_pool_attn  = nn.MultiheadAttention(
            EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True
        )

        self.local_attn  = _make_transformer(LOCAL_LAYERS)
        # Global: cross-attn decoder (Q=P, KV=P+F) — model.py 와 동기
        self.global_attn = _make_decoder(GLOBAL_LAYERS)

        # Step 3 sequential heads — model.py 와 키 동일.
        #   target_i  ~ Categorical(target_head(src_i))
        #   amount_i  ~ Categorical(amount_pair_head(src_i, dst_target_i))
        self.target_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, MAX_PLANETS),
        )
        self.amount_pair_head = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, EMBED_DIM),
            nn.ReLU(),
            nn.Linear(EMBED_DIM, NUM_ACTIONS),
        )

    def amount_logits(self, src_token, target_idx):
        """Pair-wise amount head — model.py._amount_logits 와 동일.

        src_token : (B, P, E) — forward 출력
        target_idx: (B, P) long
        returns   : (B, P, NUM_ACTIONS) — 5-way (idx 0 = skip)
        """
        B, P, E = src_token.shape
        idx_safe   = target_idx.clamp(0, P - 1)
        gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, E)
        dst_token  = src_token.gather(1, gather_idx)
        pair_input = torch.cat([src_token, dst_token], dim=-1)
        return self.amount_pair_head(pair_input)

    def forward(self, obs_flat):
        """
        obs_flat: (B, HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM))
        returns src_token: (B, MAX_PLANETS, EMBED_DIM)

        Heads (target_head / amount_pair_head) 는 caller (main.py) 가 직접 호출.
        sequential 샘플링이 필요해서 forward 단에서 amount logits 를 미리 못 만듦.
        """
        B = obs_flat.shape[0]
        p_size = MAX_PLANETS * PLANET_DIM
        f_size = MAX_FLEETS  * FLEET_DIM

        obs   = obs_flat.view(B, HISTORY, p_size + f_size)
        p_raw = obs[:, :, :p_size].view(B, HISTORY, MAX_PLANETS, PLANET_DIM)
        f_raw = obs[:, :, p_size:].view(B, HISTORY, MAX_FLEETS,  FLEET_DIM)

        p_features = p_raw[..., :PLANET_FEAT_DIM]
        f_features = f_raw[..., :FLEET_FEAT_DIM]
        fp_idx_raw = f_raw[:, -1, :, 7]    # src_idx
        dp_idx_raw = f_raw[:, -1, :, 8]    # dst_idx

        # Padding masks (sentinel 기반, model.py 와 동일)
        # fleet: idx 7,8 모두 -2 = empty slot. -1 = real fleet w/ lookup miss (mask 안 함)
        planet_pad_h   = (p_raw[..., -1] == 0)
        fleet_pad_h    = (f_raw[..., 7] == -2)
        planet_pad_now = planet_pad_h[:, -1, :]
        fleet_pad_now  = fleet_pad_h[:, -1, :]

        p_emb = self.planet_embed(p_features)
        f_emb = self.fleet_embed(f_features)

        t_idx = torch.arange(HISTORY, device=obs_flat.device)

        # Planet temporal + attention pool
        p_pos = self.planet_temporal_pos(t_idx)
        p_t   = p_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_PLANETS, HISTORY, EMBED_DIM)
        p_t   = p_t + p_pos.unsqueeze(0)
        p_pad_seq = planet_pad_h.permute(0, 2, 1).contiguous().view(B * MAX_PLANETS, HISTORY)
        p_t   = self.planet_temporal_attn(p_t, src_key_padding_mask=_safe_pad_mask(p_pad_seq))
        p_q   = self.planet_pool_query.expand(B * MAX_PLANETS, 1, EMBED_DIM)
        p_t, _ = self.planet_pool_attn(p_q, p_t, p_t,
                                        key_padding_mask=_safe_pad_mask(p_pad_seq))
        p_t   = p_t.squeeze(1).view(B, MAX_PLANETS, EMBED_DIM)

        # Fleet temporal + attention pool
        f_pos = self.fleet_temporal_pos(t_idx)
        f_t   = f_emb.permute(0, 2, 1, 3).contiguous().view(B * MAX_FLEETS, HISTORY, EMBED_DIM)
        f_t   = f_t + f_pos.unsqueeze(0)
        f_pad_seq = fleet_pad_h.permute(0, 2, 1).contiguous().view(B * MAX_FLEETS, HISTORY)
        f_t   = self.fleet_temporal_attn(f_t, src_key_padding_mask=_safe_pad_mask(f_pad_seq))
        f_q   = self.fleet_pool_query.expand(B * MAX_FLEETS, 1, EMBED_DIM)
        f_t, _ = self.fleet_pool_attn(f_q, f_t, f_t,
                                       key_padding_mask=_safe_pad_mask(f_pad_seq))
        f_t   = f_t.squeeze(1).view(B, MAX_FLEETS, EMBED_DIM)

        # Planet fusion — source + destination (model.py._gated_planet_fuse 와 동일)
        # Hole A: planet_pad_now 도 valid 검증에 포함 (padded slot lookup 차단)
        # Hole D: valid 슬롯만 matmul → compute 절감
        def _fuse(f_t, idx_raw, value_layer, gate_layer):
            B, F, E    = f_t.shape
            idx        = idx_raw.long()
            idx_safe   = idx.clamp(0, MAX_PLANETS - 1)
            gathered_pad = planet_pad_now.gather(1, idx_safe)
            valid      = (idx >= 0) & (idx < MAX_PLANETS) & ~gathered_pad

            gather_idx = idx_safe.unsqueeze(-1).expand(-1, -1, E)
            planet_t   = p_t.gather(1, gather_idx)

            flat_valid = valid.flatten()
            f_flat     = f_t.reshape(-1, E)
            src_flat   = planet_t.reshape(-1, E)
            f_sub      = f_flat[flat_valid]
            src_sub    = src_flat[flat_valid]

            fused_in   = torch.cat([f_sub, src_sub], dim=-1)
            gate       = torch.sigmoid(gate_layer(fused_in))
            cand       = torch.tanh(value_layer(fused_in))
            update_sub = gate * cand

            update_flat = torch.zeros_like(f_flat)
            update_flat[flat_valid] = update_sub
            return f_t + update_flat.view(B, F, E)

        f_t = _fuse(f_t, fp_idx_raw, self.fleet_source_value, self.fleet_source_gate)
        f_t = _fuse(f_t, dp_idx_raw, self.fleet_dest_value,   self.fleet_dest_gate)

        # Local (fleet ↔ planet) + Global cross-attn
        local_tokens = torch.cat([p_t, f_t], dim=1)
        local_pad    = torch.cat([planet_pad_now, fleet_pad_now], dim=1)
        local_out    = self.local_attn(local_tokens, src_key_padding_mask=_safe_pad_mask(local_pad))
        p_query      = local_out[:, :MAX_PLANETS, :]                      # decoder query (B, P, E)
        global_out   = self.global_attn(
            tgt=p_query,
            memory=local_out,                                              # KV = P+F (함대 정보 유지)
            tgt_key_padding_mask=_safe_pad_mask(planet_pad_now),
            memory_key_padding_mask=_safe_pad_mask(local_pad),
        )

        return global_out
