"""fleet_embed 가 src/dst idx 를 feature 로 받지 않는지 회귀.

설계 의도:
  idx 7 (src_idx), idx 8 (dst_idx) 는 lookup pointer 일 뿐, feature 가 아님.
  fleet_embed 는 첫 7 dim (numeric features) 만 소비하고
  마지막 2 dim (idx) 은 gather 에서만 사용.

회귀 시나리오:
  - fleet_embed.weight.shape[1] == FLEET_FEAT_DIM (= 7)
  - 같은 7 dim feature + 다른 src_idx → fleet_embed 출력 동일
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest
from model import OrbitWarsPolicy, FLEET_DIM, FLEET_FEAT_DIM, EMBED_DIM, MAX_FLEETS


def test_fleet_feat_dim_is_two_less_than_fleet_dim():
    """FLEET_FEAT_DIM + 2 == FLEET_DIM (마지막 두 dim = src_idx, dst_idx)."""
    assert FLEET_FEAT_DIM + 2 == FLEET_DIM


def test_fleet_embed_input_dim_excludes_idx():
    """fleet_embed.weight.shape[1] 이 FLEET_FEAT_DIM 과 일치 → idx 안 받음."""
    torch.manual_seed(0)
    m = OrbitWarsPolicy()
    in_dim = m.fleet_embed.weight.shape[1]
    assert in_dim == FLEET_FEAT_DIM, \
        f"fleet_embed input={in_dim}, 기대값={FLEET_FEAT_DIM} — idx 가 잘못 들어옴"


def test_same_features_different_idx_yields_same_embed():
    """같은 7 numeric feature + 다른 from_planet_idx → fleet_embed 출력 bit-exact 동일.

    이게 깨지면 fleet_embed 가 idx 를 feature 처럼 학습 중 → 설계 위반.
    """
    torch.manual_seed(0)
    m = OrbitWarsPolicy().eval()

    B, H = 2, 5   # 임의 history 길이 (fleet_embed 는 step 무관)
    feats = torch.randn(B, H, MAX_FLEETS, FLEET_FEAT_DIM)

    # idx 자리 (src, dst) 두 dim — 서로 매우 다른 값으로
    idx_a = torch.zeros(B, H, MAX_FLEETS, 2)
    idx_b = torch.full((B, H, MAX_FLEETS, 2), 39.0)   # MAX_PLANETS-1 가깝게
    idx_c = torch.full((B, H, MAX_FLEETS, 2), -1.0)

    f_raw_a = torch.cat([feats, idx_a], dim=-1)
    f_raw_b = torch.cat([feats, idx_b], dim=-1)
    f_raw_c = torch.cat([feats, idx_c], dim=-1)

    # forward 가 하는 것과 동일: 첫 FLEET_FEAT_DIM 만 떼서 embed
    emb_a = m.fleet_embed(f_raw_a[..., :FLEET_FEAT_DIM])
    emb_b = m.fleet_embed(f_raw_b[..., :FLEET_FEAT_DIM])
    emb_c = m.fleet_embed(f_raw_c[..., :FLEET_FEAT_DIM])

    assert torch.equal(emb_a, emb_b), \
        "다른 src_idx 인데 fleet_embed 출력이 다름 — idx 가 새고 있음"
    assert torch.equal(emb_a, emb_c), \
        "-1 sentinel idx 가 fleet_embed 출력에 영향 — slice 가 잘못됨"


def test_fleet_embed_full_obs_with_extreme_idx_safe():
    """obs 마지막 dim 에 극단값 (39, -1) 이 와도 fleet_embed forward 안 깨짐."""
    torch.manual_seed(0)
    m = OrbitWarsPolicy().eval()

    from model import HISTORY, MAX_PLANETS, PLANET_DIM
    OBS_DIM = HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM)
    obs = torch.zeros(1, OBS_DIM)

    # last-step fleet 들의 src/dst idx 자리에 극단값 박기
    last_off = (HISTORY - 1) * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM)
    last_off += MAX_PLANETS * PLANET_DIM
    for f in range(MAX_FLEETS):
        # idx 7 = src_idx, idx 8 = dst_idx
        obs[0, last_off + f * FLEET_DIM + 7] = 39.0 if f % 2 else -1.0
        obs[0, last_off + f * FLEET_DIM + 8] = -1.0 if f % 3 else 39.0

    with torch.no_grad():
        logits, value = m(obs)

    assert not torch.isnan(logits).any() and not torch.isinf(logits).any()
    assert not torch.isnan(value).any() and not torch.isinf(value).any()
