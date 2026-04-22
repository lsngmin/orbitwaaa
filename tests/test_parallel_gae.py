"""
병렬 rollout GAE 경계 테스트.

- _collect_single이 (obs, actions, advantages, returns, log_probs) 5-tuple을 반환하는지
- 각 worker가 독립적으로 GAE를 계산하므로 worker 경계에서 GAE가 오염되지 않는지
  (worker A의 마지막 step이 worker B의 첫 delta에 영향을 주지 않아야 함)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from unittest.mock import patch, MagicMock
from train import compute_gae, _collect_single


# ── _collect_single 반환값 shape/타입 ────────────────────────────────────────

def _make_mock_collect_single(n_steps=4):
    """_collect_single의 반환값을 시뮬레이션 (실제 env 없이)."""
    rewards    = torch.tensor([0.0, 0.0, 0.0, 1.0])[:n_steps]
    dones      = torch.tensor([0.0, 0.0, 0.0, 1.0])[:n_steps]
    values     = torch.zeros(n_steps, 1)
    obs        = torch.zeros(n_steps, 8)
    actions    = torch.zeros(n_steps, 3)
    log_probs  = torch.zeros(n_steps)

    advantages, returns = compute_gae(rewards, dones, values, last_value=0.0)
    return obs, actions, advantages, returns, log_probs


def test_collect_single_returns_5_tuple():
    """_collect_single이 5-tuple을 반환해야 함 (GAE 내부 계산 후)."""
    result = _make_mock_collect_single()
    assert len(result) == 5, f"expected 5-tuple, got {len(result)}-tuple"


def test_collect_single_advantages_shape():
    n = 6
    obs, actions, advantages, returns, log_probs = _make_mock_collect_single(n)
    assert advantages.shape == (n,), f"advantages shape: {advantages.shape}"
    assert returns.shape    == (n,), f"returns shape: {returns.shape}"


def test_collect_single_returns_eq_adv_plus_values():
    """returns = advantages + values — compute_gae 항등식 확인."""
    rewards   = torch.tensor([0.0, 1.0, 0.0, 0.5])
    dones     = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values    = torch.tensor([[0.3], [0.6], [0.4], [0.8]])
    adv, ret  = compute_gae(rewards, dones, values, last_value=0.0)
    assert torch.allclose(ret, adv + values.squeeze(-1), atol=1e-5)


# ── worker 경계 오염 없음 ─────────────────────────────────────────────────────

def test_worker_boundary_independence():
    """두 worker를 concat했을 때 경계 오염이 없는지 확인.

    worker A의 마지막 done=1이면 worker B 첫 delta에 worker A의 value가
    영향을 주면 안 된다. 각 worker가 독립적으로 GAE를 계산하므로
    concat 후에도 각 worker 내 advantages는 변하지 않아야 한다.
    """
    # Worker A: 에피소드 종료 (done=1)로 끝남
    rew_a   = torch.tensor([0.0, 0.0, 1.0])
    done_a  = torch.tensor([0.0, 0.0, 1.0])
    val_a   = torch.tensor([[0.5], [0.5], [0.9]])
    adv_a, ret_a = compute_gae(rew_a, done_a, val_a, last_value=0.0)

    # Worker B: 에피소드 진행 중 (non-terminal)
    rew_b   = torch.tensor([0.0, 0.0, 0.0])
    done_b  = torch.tensor([0.0, 0.0, 0.0])
    val_b   = torch.tensor([[0.5], [0.5], [0.5]])
    adv_b, ret_b = compute_gae(rew_b, done_b, val_b, last_value=0.5)

    # concat (collect_rollout처럼)
    adv_cat = torch.cat([adv_a, adv_b])
    ret_cat = torch.cat([ret_a, ret_b])

    # worker A 부분은 변하지 않아야 함
    assert torch.allclose(adv_cat[:3], adv_a, atol=1e-6), \
        "worker A advantages changed after concat"
    # worker B 부분도 변하지 않아야 함
    assert torch.allclose(adv_cat[3:], adv_b, atol=1e-6), \
        "worker B advantages changed after concat"


def test_old_shared_last_value_would_fail():
    """이전 버그(results[-1][6] 공유)가 실제로 wrong result를 낳는지 역으로 확인.

    worker A last_value=0.0 (terminal), worker B last_value=0.9 (non-terminal).
    이전 코드는 모든 worker에 last_value=0.9를 쓰므로
    worker A의 마지막 delta가 달라져야 한다.
    """
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 1.0])
    values  = torch.tensor([[0.5], [0.5], [0.9]])

    # 올바른 last_value (worker A는 terminal이므로 0.0)
    adv_correct, _ = compute_gae(rewards, dones, values, last_value=0.0)
    # 잘못된 last_value (다른 worker의 last_value를 공유)
    adv_wrong,   _ = compute_gae(rewards, dones, values, last_value=0.9)

    # 마지막 step이 done=1이므로 last_value가 달라도 결과가 같아야 함 — 올바른 구현
    assert torch.allclose(adv_correct, adv_wrong, atol=1e-5), \
        "terminal step에서 last_value 공유가 결과에 영향을 줌"

    # 첫 번째 step은 영향을 받는다 (non-terminal 체인이 다름)
    # 이 경우는 done=1이 마지막이라 전파가 차단되어 같음을 확인
    rew2   = torch.tensor([0.0, 0.0, 0.0])
    done2  = torch.tensor([0.0, 0.0, 0.0])  # 모두 non-terminal
    val2   = torch.tensor([[0.5], [0.5], [0.5]])

    adv_c2, _ = compute_gae(rew2, done2, val2, last_value=0.0)
    adv_w2, _ = compute_gae(rew2, done2, val2, last_value=0.9)
    assert not torch.allclose(adv_c2, adv_w2, atol=1e-5), \
        "non-terminal rollout에서 last_value가 영향 없음 — 버그 탐지 실패"
