"""Encoder obs layout 회귀 — 모듈 간 상수 동기화 + flatten/restore.

drift 가 잦은 곳:
  - submission_features / env_wrapper / model / submission_actor 의 FLEET_DIM
  - FLEET_FEAT_DIM (= FLEET_DIM - 1)

여기서 잡히면 build_submission 단계 전에 알 수 있음.

스코프 외 (의도적 제외):
  numpy_model 은 본 sync 회귀에 포함하지 않는다.
  - PLANET_DIM 이 학습 모델과 의도적으로 어긋난 상태 ("무시" 결정 — 별도 트랙)
  - numpy_model 은 build_submission 흐름의 inference fallback 으로,
    여기서 같이 검증하면 무관한 항목이 빨갛게 떠 fusion 회귀 신호를 가린다.
  numpy_model 의 layout 동기화는 별도 build_submission 검증에서 다룬다.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import pytest

import env_wrapper as ew
import submission_features as sf
import submission_actor as sa
import model as mdl


def test_fleet_dim_synced_across_modules():
    """FLEET_DIM 이 4개 모듈에서 동일."""
    assert ew.FLEET_DIM == sf.FLEET_DIM == sa.FLEET_DIM == mdl.FLEET_DIM == 8


def test_fleet_feat_dim_synced():
    """FLEET_FEAT_DIM 도 동기 (FLEET_DIM - 1)."""
    assert ew.FLEET_FEAT_DIM == sf.FLEET_FEAT_DIM == sa.FLEET_FEAT_DIM == mdl.FLEET_FEAT_DIM == 7


def test_planet_dim_synced_across_modules():
    """PLANET_DIM 도 함께 회귀 — drift 방지."""
    assert ew.PLANET_DIM == sf.PLANET_DIM == sa.PLANET_DIM == mdl.PLANET_DIM


def test_max_planets_fleets_synced():
    """MAX_PLANETS / MAX_FLEETS / HISTORY 동기."""
    assert ew.MAX_PLANETS == sf.MAX_PLANETS == sa.MAX_PLANETS == mdl.MAX_PLANETS
    assert ew.MAX_FLEETS  == sf.MAX_FLEETS  == sa.MAX_FLEETS  == mdl.MAX_FLEETS
    assert ew.HISTORY     == sf.HISTORY     == sa.HISTORY     == mdl.HISTORY


def test_obs_dim_formula():
    """OBS_DIM = HISTORY * (P*PD + F*FD)."""
    expected = mdl.HISTORY * (mdl.MAX_PLANETS * mdl.PLANET_DIM
                               + mdl.MAX_FLEETS * mdl.FLEET_DIM)
    obs = torch.zeros(1, expected)
    m = mdl.OrbitWarsPolicy().eval()
    with torch.no_grad():
        logits, value = m(obs)
    assert logits.shape == (1, mdl.MAX_PLANETS, mdl.ACTION_DIM)
    assert value.shape  == (1, 1)


def test_flatten_restore_round_trip():
    """encode → flatten → unflatten 후에도 idx 가 last dim 에 보존."""
    P, F, H = mdl.MAX_PLANETS, mdl.MAX_FLEETS, mdl.HISTORY
    PD, FD  = mdl.PLANET_DIM, mdl.FLEET_DIM

    p_hist = np.random.randn(H, P, PD).astype(np.float32)
    f_hist = np.random.randn(H, F, FD).astype(np.float32)
    # idx 자리에 의도적 값
    f_hist[H - 1, 0, -1] = 7.0
    f_hist[H - 1, 5, -1] = -1.0

    flat = np.concatenate([p_hist.flatten(), f_hist.flatten()])
    obs  = torch.from_numpy(flat).unsqueeze(0)

    # model forward 와 동일한 reshape 로직 검증
    p_size = P * PD
    f_size = F * FD
    obs_v = obs.view(1, H, p_size + f_size)
    f_raw = obs_v[:, :, p_size:].view(1, H, F, FD)

    assert f_raw[0, H - 1, 0, -1].item() == pytest.approx(7.0)
    assert f_raw[0, H - 1, 5, -1].item() == pytest.approx(-1.0)


def test_submission_actor_loads_state_from_policy():
    """OrbitWarsPolicy state_dict → OrbitWarsActor (critic 제외 strict=False) load 무결.

    학습/제출 parity 회귀 — fleet_source_* 키도 양쪽에 존재해야 매칭.
    """
    torch.manual_seed(0)
    policy = mdl.OrbitWarsPolicy()
    actor  = sa.OrbitWarsActor()

    sd = policy.state_dict()
    res = actor.load_state_dict(sd, strict=False)

    # critic.* 외엔 unexpected 가 없어야 함
    unexpected_non_critic = [k for k in res.unexpected_keys if not k.startswith("critic.")]
    assert not unexpected_non_critic, \
        f"actor 에 없는 키: {unexpected_non_critic}"
    # actor 가 missing 키를 갖고 있으면 학습 가중치가 빠진 것
    assert not res.missing_keys, f"actor 가 요구하는 missing 키: {res.missing_keys}"
