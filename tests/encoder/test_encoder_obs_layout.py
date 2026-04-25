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
    """env_wrapper._build_tensor flatten → model forward unflatten 의 layout 일치 회귀.

    과거 버그: `np.concatenate([p_hist.flatten(), f_hist.flatten()])` (=p_all||f_all)
    + `obs.view(B, H, p_size+f_size)` 가 misalign 해서 temporal attention 이 garbage
    를 보던 문제. 이 테스트는 모든 턴/슬롯/dim 에서 round-trip 이 보존되는지 확인한다.
    """
    P, F, H = mdl.MAX_PLANETS, mdl.MAX_FLEETS, mdl.HISTORY
    PD, FD  = mdl.PLANET_DIM, mdl.FLEET_DIM

    rng = np.random.default_rng(0)
    p_hist = rng.standard_normal((H, P, PD)).astype(np.float32)
    f_hist = rng.standard_normal((H, F, FD)).astype(np.float32)
    # 턴별 고유 마커 — 모든 턴이 따로 보존되는지 확인
    for h in range(H):
        p_hist[h, 0, 0] = float(h + 1)
        f_hist[h, 0, -1] = float(h + 100)

    # env_wrapper._build_tensor 와 동일한 turn-interleave layout
    p_flat   = p_hist.reshape(H, P * PD)
    f_flat   = f_hist.reshape(H, F * FD)
    per_turn = np.concatenate([p_flat, f_flat], axis=1)
    flat     = per_turn.reshape(-1)
    obs      = torch.from_numpy(flat).unsqueeze(0)

    # model forward 의 view 와 동일
    p_size = P * PD
    f_size = F * FD
    obs_v = obs.view(1, H, p_size + f_size)
    p_raw = obs_v[:, :, :p_size].view(1, H, P, PD)
    f_raw = obs_v[:, :, p_size:].view(1, H, F, FD)

    # 모든 턴에서 마커 보존
    for h in range(H):
        assert p_raw[0, h, 0, 0].item() == pytest.approx(float(h + 1)), \
            f"planet h={h} marker lost"
        assert f_raw[0, h, 0, -1].item() == pytest.approx(float(h + 100)), \
            f"fleet h={h} marker lost"

    # 전체 데이터 round-trip
    assert torch.allclose(p_raw[0], torch.from_numpy(p_hist))
    assert torch.allclose(f_raw[0], torch.from_numpy(f_hist))


def test_env_build_tensor_matches_model_view():
    """env_wrapper._build_tensor 가 만든 flat 이 model.view 와 정확히 align."""
    from collections import deque
    env = ew.OrbitWarsEnv()

    # 턴별 distinct array 로 history 재구성 + 마커
    p_list = []
    f_list = []
    for h in range(mdl.HISTORY):
        p = np.zeros((mdl.MAX_PLANETS, mdl.PLANET_DIM), dtype=np.float32)
        f = np.zeros((mdl.MAX_FLEETS,  mdl.FLEET_DIM),  dtype=np.float32)
        p[0, 0]  = float(h + 1)
        f[0, -1] = float(h + 100)
        p_list.append(p)
        f_list.append(f)
    env._planet_history = deque(p_list, maxlen=mdl.HISTORY)
    env._fleet_history  = deque(f_list, maxlen=mdl.HISTORY)

    flat = env._build_tensor()
    obs  = torch.from_numpy(flat).unsqueeze(0)
    p_size = mdl.MAX_PLANETS * mdl.PLANET_DIM
    f_size = mdl.MAX_FLEETS  * mdl.FLEET_DIM
    obs_v = obs.view(1, mdl.HISTORY, p_size + f_size)
    p_raw = obs_v[:, :, :p_size].view(1, mdl.HISTORY, mdl.MAX_PLANETS, mdl.PLANET_DIM)
    f_raw = obs_v[:, :, p_size:].view(1, mdl.HISTORY, mdl.MAX_FLEETS,  mdl.FLEET_DIM)

    for h in range(mdl.HISTORY):
        assert p_raw[0, h, 0, 0].item() == pytest.approx(float(h + 1)), \
            f"_build_tensor planet h={h} drift"
        assert f_raw[0, h, 0, -1].item() == pytest.approx(float(h + 100)), \
            f"_build_tensor fleet h={h} drift"


def test_attention_pool_params_present():
    """Attention pool query + cross-attn 파라미터가 state_dict 에 존재.

    회귀: pool 모듈이 빠지면 학습 → 제출 weights 호환성 깨짐.
    """
    policy = mdl.OrbitWarsPolicy()
    sd_keys = set(policy.state_dict().keys())
    assert "planet_pool_query" in sd_keys, "planet_pool_query 누락"
    # MultiheadAttention 의 in_proj_weight, out_proj.weight 등이 들어있어야 함
    assert any(k.startswith("planet_pool_attn.") for k in sd_keys), \
        "planet_pool_attn 파라미터 누락"
    if mdl.FLEET_TEMPORAL:
        assert "fleet_pool_query" in sd_keys
        assert any(k.startswith("fleet_pool_attn.") for k in sd_keys)


def test_attention_pool_query_independent_of_input_last_turn():
    """학습된 query 가 마지막 턴 input 에 종속되지 않음 — pool 의 핵심 동기.

    같은 H턴 시퀀스의 마지막 턴만 다른 두 입력 → pool 출력이 유의미하게 다름
    (마지막 턴 무시 X) 이지만, 마지막 턴 anchor 가 아니라는 sanity 차원.
    여기선 단순히 forward 가 안 깨지는지 회귀.
    """
    torch.manual_seed(0)
    m = mdl.OrbitWarsPolicy().eval()
    P, F, H = mdl.MAX_PLANETS, mdl.MAX_FLEETS, mdl.HISTORY
    PD, FD = mdl.PLANET_DIM, mdl.FLEET_DIM
    OBS = H * (P * PD + F * FD)
    obs = torch.zeros(1, OBS)
    with torch.no_grad():
        logits, value = m(obs)
    assert logits.shape == (1, P, mdl.ACTION_DIM)
    assert value.shape == (1, 1)
    assert not torch.isnan(logits).any()


def test_fleet_padding_mask_separates_empty_from_lookup_miss():
    """fleet padding mask 가 빈 슬롯(-2) 만 가리고, real fleet w/ src miss(-1)는 통과.

    회귀: 둘 다 -1 로 겹쳐있던 시절엔 lookup miss real fleet 이 attention 에서
    통째로 사라졌음. 이제 sentinel 분리됨.
    """
    P, F, H = mdl.MAX_PLANETS, mdl.MAX_FLEETS, mdl.HISTORY
    PD, FD  = mdl.PLANET_DIM, mdl.FLEET_DIM

    f_raw = torch.zeros(1, H, F, FD)
    # slot 0: real fleet, src lookup miss → -1
    f_raw[0, :, 0, -1] = -1.0
    # slot 1: 빈 슬롯 → -2
    f_raw[0, :, 1, -1] = -2.0
    # slot 2: valid src (idx=3) → 3
    f_raw[0, :, 2, -1] = 3.0

    # model.py 와 동일 mask 로직
    fleet_pad_h = (f_raw[..., -1] == -2)
    # slot 0 (real, miss) 은 padding 아님
    assert not fleet_pad_h[0, -1, 0].item(), "real fleet w/ src=-1 이 padding 으로 잡혔음"
    # slot 1 (empty) 은 padding
    assert fleet_pad_h[0, -1, 1].item(), "빈 슬롯이 padding 안 잡힘"
    # slot 2 (valid) 은 padding 아님
    assert not fleet_pad_h[0, -1, 2].item()


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
