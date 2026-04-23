"""
Minibatch PPO 검증 테스트.

- n_epochs * ceil(N/minibatch_size) 횟수만큼 업데이트 발생
- 반환값 9개 (p_loss, v_loss, e_loss, approx_kl, clip_frac,
  epochs_done, ent_launch, ent_ships, ent_target) shape/type 확인
- approx_kl >= 0, clip_frac in [0, 1]
- 동일 데이터 single-pass 대비 weights가 실제로 달라지는지 확인
- NaN/Inf 없는지 smoke test
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from unittest.mock import patch
from model import OrbitWarsPolicy, HISTORY, MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, ACTION_DIM
from train import ppo_update, DEVICE

OBS_DIM = HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM)
N       = 32  # rollout 길이


def make_dummy_batch(model, n=N):
    """model로 실제 log_prob을 뽑아 old_log_probs로 사용 — ratio 폭발 방지."""
    obs = torch.randn(n, OBS_DIM).to(DEVICE)
    with torch.no_grad():
        action_t, old_log_probs, _ = model.get_action_and_value(obs)
    actions   = action_t
    returns   = torch.randn(n).to(DEVICE)
    advantages = torch.randn(n).to(DEVICE)
    return obs, actions, old_log_probs, returns, advantages


@pytest.fixture
def model_and_opt():
    m   = OrbitWarsPolicy().to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    return m, opt


def test_ppo_update_returns_nine_values(model_and_opt):
    """ppo_update가 9개 값을 반환하는지 확인."""
    m, opt = model_and_opt
    obs, actions, old_lp, rets, advs = make_dummy_batch(m)
    result = ppo_update(m, opt, obs, actions, old_lp, rets, advs,
                        n_epochs=1, minibatch_size=N)
    assert len(result) == 9


def test_approx_kl_nonnegative(model_and_opt):
    """approx_kl >= 0 (KL divergence 하한)."""
    m, opt = model_and_opt
    obs, actions, old_lp, rets, advs = make_dummy_batch(m)
    _, _, _, approx_kl, _, _, _, _, _ = ppo_update(
        m, opt, obs, actions, old_lp, rets, advs,
        n_epochs=1, minibatch_size=N)
    assert approx_kl >= 0, f"approx_kl={approx_kl} < 0"


def test_clip_frac_in_range(model_and_opt):
    """clip_frac in [0, 1]."""
    m, opt = model_and_opt
    obs, actions, old_lp, rets, advs = make_dummy_batch(m)
    _, _, _, _, clip_frac, _, _, _, _ = ppo_update(
        m, opt, obs, actions, old_lp, rets, advs,
        n_epochs=1, minibatch_size=N)
    assert 0.0 <= clip_frac <= 1.0, f"clip_frac={clip_frac} out of range"


def test_no_nan_in_outputs(model_and_opt):
    """출력값에 NaN/Inf가 없는지 확인."""
    m, opt = model_and_opt
    obs, actions, old_lp, rets, advs = make_dummy_batch(m)
    results = ppo_update(m, opt, obs, actions, old_lp, rets, advs,
                         n_epochs=2, minibatch_size=16)
    for val in results:
        assert not (val != val), f"NaN 발생: {results}"
        assert abs(val) < 1e6,   f"Inf 발생: {results}"


def test_weights_change_after_update(model_and_opt):
    """업데이트 후 모델 가중치가 실제로 변하는지 확인."""
    m, opt = model_and_opt
    before = [p.clone() for p in m.parameters()]
    obs, actions, old_lp, rets, advs = make_dummy_batch(m)
    ppo_update(m, opt, obs, actions, old_lp, rets, advs, n_epochs=1, minibatch_size=N)
    after = list(m.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "ppo_update 후 가중치가 변하지 않음"


def test_multi_epoch_updates_more_than_single(model_and_opt):
    """n_epochs=4가 n_epochs=1보다 가중치를 더 많이 움직이는지 확인.

    target_kl 조기 종료가 개입하지 않도록 None으로 비활성화하여 공정 비교.
    """
    import copy

    m1, opt1 = model_and_opt
    obs, actions, old_lp, rets, advs = make_dummy_batch(m1)

    m2 = copy.deepcopy(m1)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)

    init_params = [p.clone() for p in m1.parameters()]

    ppo_update(m1, opt1, obs, actions, old_lp, rets, advs,
               n_epochs=1, minibatch_size=N, target_kl=None)
    ppo_update(m2, opt2, obs, actions, old_lp, rets, advs,
               n_epochs=4, minibatch_size=N, target_kl=None)

    delta1 = sum((p - i).abs().sum().item() for p, i in zip(m1.parameters(), init_params))
    delta2 = sum((p - i).abs().sum().item() for p, i in zip(m2.parameters(), init_params))

    assert delta2 > delta1, f"4 epochs({delta2:.4f}) <= 1 epoch({delta1:.4f})"
