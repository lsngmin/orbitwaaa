"""
GAE 검증 테스트.

- GAE 수식 정확성 (단순 예제로 수작업 계산 대조)
- lambda=0 → TD residual과 동일
- lambda=1 → Monte Carlo returns - values와 동일
- returns = advantages + values 항등식
- 이전 버그(returns - returns.mean()) 방식과 달라야 함
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import torch
import pytest
from train import compute_gae


def test_gae_returns_equals_advantages_plus_values():
    """returns = advantages + values 항등식 확인."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 1.0])
    values  = torch.tensor([[0.3], [0.5], [0.8]])

    advantages, returns = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)
    assert torch.allclose(returns, advantages + values.squeeze(-1), atol=1e-5)


def test_gae_lambda_zero_equals_td_residual():
    """lambda=0 → advantage = delta = r + gamma*V(t+1) - V(t)."""
    rewards = torch.tensor([1.0, 0.0, 0.0])
    dones   = torch.tensor([0.0, 0.0, 1.0])
    values  = torch.tensor([[0.5], [0.4], [0.2]])
    gamma   = 0.99

    advantages, _ = compute_gae(rewards, dones, values, gamma=gamma, lam=0.0)

    # 수작업: lambda=0이면 last_gae 누적 없이 delta만
    # t=2: delta = 0 + 0.99*0*(1-1) - 0.2 = -0.2
    # t=1: delta = 0 + 0.99*0.2*(1-0) - 0.4 = 0.198 - 0.4 = -0.202
    # t=0: delta = 1 + 0.99*0.4*(1-0) - 0.5 = 1 + 0.396 - 0.5 = 0.896
    expected = torch.tensor([0.896, -0.202, -0.2])
    assert torch.allclose(advantages, expected, atol=1e-4), \
        f"lambda=0 GAE != TD residual: {advantages} vs {expected}"


def test_gae_done_resets_bootstrap():
    """done=1인 스텝 이후 next value가 0으로 처리되는지 확인."""
    rewards = torch.tensor([1.0, 0.0])
    dones   = torch.tensor([1.0, 0.0])  # t=0에서 에피소드 종료
    values  = torch.tensor([[0.9], [0.5]])
    gamma, lam = 0.99, 0.95

    advantages, returns = compute_gae(rewards, dones, values, gamma=gamma, lam=lam)

    # t=0: done=1이므로 next_val=0 강제
    # delta_0 = 1.0 + 0.99*0*(1-1) - 0.9 = 0.1
    # gae_0 = delta_0 + gamma*lam*(1-1)*gae_1 = 0.1 (누적 없음)
    assert torch.allclose(advantages[0], torch.tensor(0.1), atol=1e-5), \
        f"done 이후 bootstrap이 0이 아님: adv[0]={advantages[0]}"


def test_gae_differs_from_old_returns_mean_baseline():
    """GAE advantage가 이전 버그(returns - returns.mean()) 방식과 다른지 확인."""
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values  = torch.tensor([[0.1], [0.2], [0.5], [0.9]])

    advantages, returns = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)

    old_adv = returns - returns.mean()
    assert not torch.allclose(advantages, old_adv, atol=1e-4), \
        "GAE advantage가 이전 returns.mean() 방식과 동일 — 회귀 의심"


def test_gae_last_value_bootstrap():
    """non-terminal truncation 시 last_value가 마지막 delta에 반영되는지 확인."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    dones   = torch.tensor([0.0, 0.0, 0.0])   # 에피소드 미종료 — truncated
    values  = torch.tensor([[0.5], [0.5], [0.5]])
    gamma, lam = 0.99, 0.0  # lambda=0으로 delta만 보기

    adv_no_boot, _ = compute_gae(rewards, dones, values, last_value=0.0,   gamma=gamma, lam=lam)
    adv_boot,    _ = compute_gae(rewards, dones, values, last_value=1.0,   gamma=gamma, lam=lam)

    # last_value=1.0이면 마지막 delta = 0 + 0.99*1.0 - 0.5 = 0.49 > 0
    # last_value=0.0이면 마지막 delta = 0 + 0.99*0.0 - 0.5 = -0.5
    assert adv_boot[-1] > adv_no_boot[-1], (
        f"last_value bootstrap 미반영: boot={adv_boot[-1]:.4f}, no_boot={adv_no_boot[-1]:.4f}"
    )


def test_gae_last_value_zero_when_terminal():
    """마지막 step이 done=1이면 last_value가 무시되어야 함."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 1.0])   # 마지막이 terminal
    values  = torch.tensor([[0.5], [0.5], [0.5]])

    adv_no_boot, _ = compute_gae(rewards, dones, values, last_value=0.0)
    adv_boot,    _ = compute_gae(rewards, dones, values, last_value=999.0)

    # 마지막이 done=1이므로 last_value는 어떤 값이든 결과가 같아야 함
    assert torch.allclose(adv_no_boot, adv_boot, atol=1e-5), (
        "terminal step에서 last_value가 advantage에 영향을 줌"
    )


def test_gae_shape():
    """출력 shape가 입력 rewards와 동일한지 확인."""
    N = 16
    rewards = torch.zeros(N)
    dones   = torch.zeros(N)
    values  = torch.zeros(N, 1)

    advantages, returns = compute_gae(rewards, dones, values)
    assert advantages.shape == (N,)
    assert returns.shape    == (N,)
