"""
Full-episode collection 불변식 테스트.

새 _collect_single 구조에서의 핵심 불변식:
  - Buffer 마지막 step은 항상 done=True (episode-grained collection)
  - last_value = 0.0 고정 → bootstrap 없음 (terminal step이면 last_value 무관)
  - Multi-episode buffer에서 각 episode 경계마다 GAE reset
  - gamma=1.0 → length invariance (짧은/긴 에피소드의 terminal reward 동등)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from train import compute_gae


# ── 불변식 1: last_value irrelevance at terminal step ────────────────────────

def test_terminal_step_last_value_irrelevant():
    """마지막 step이 done=True이면 last_value는 GAE 결과에 영향 없음.

    Episode-grained collection에서 last_value=0.0 고정 안전성의 근거.
    (1 - dones[t]) 항이 bootstrap을 차단하므로 last_value 값 무관.
    """
    rewards = torch.tensor([0.0, 0.0, 5.0])      # terminal win reward
    dones   = torch.tensor([0.0, 0.0, 1.0])
    values  = torch.tensor([[0.5], [0.5], [0.9]])

    adv_zero, ret_zero   = compute_gae(rewards, dones, values, last_value=0.0)
    adv_other, ret_other = compute_gae(rewards, dones, values, last_value=99.0)

    assert torch.allclose(adv_zero, adv_other, atol=1e-6), \
        "terminal step이면 last_value 무관해야 함"
    assert torch.allclose(ret_zero, ret_other, atol=1e-6)


# ── 불변식 2: multi-episode buffer에서 경계 reset ────────────────────────────

def test_multi_episode_boundary_reset():
    """Multi-episode buffer: 에피소드 2의 reward가 에피소드 1에 새지 않음.

    에피소드 1: t=0,1,2 (done at t=2, reward +5)
    에피소드 2: t=3,4   (done at t=4, reward -5)

    각 에피소드를 독립적으로 계산한 결과와 buffer 전체 계산 결과가
    같은지 확인 (GAE reset이 done에서 제대로 작동하는지).
    """
    rewards = torch.tensor([0.0, 0.0, 5.0, 0.0, -5.0])
    dones   = torch.tensor([0.0, 0.0, 1.0, 0.0,  1.0])
    values  = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]])

    adv, _ = compute_gae(rewards, dones, values, last_value=0.0)

    # 에피소드별 독립 계산
    adv_ep1, _ = compute_gae(rewards[:3], dones[:3], values[:3], last_value=0.0)
    adv_ep2, _ = compute_gae(rewards[3:], dones[3:], values[3:], last_value=0.0)

    assert torch.allclose(adv[:3], adv_ep1, atol=1e-5), \
        "에피소드 1 advantages가 에피소드 2 reward에 오염됨"
    assert torch.allclose(adv[3:], adv_ep2, atol=1e-5), \
        "에피소드 2 advantages가 에피소드 1과 독립적이지 않음"


# ── 불변식 3: gamma=1.0 length invariance ────────────────────────────────────

def test_gamma_one_length_invariance():
    """gamma=1.0 + lam=1.0에서 짧은/긴 에피소드의 terminal reward가
    첫 step return에 동등한 크기로 반영됨.

    gamma<1이면 긴 에피소드의 terminal이 할인되어 초반 step에 약하게 전달됨.
    ELO/win-loss 관점에서 "게임 길이와 보상 크기는 무관해야 한다"는 요구사항.
    """
    # 짧은 에피소드 (5 step) terminal=+5
    rew_short  = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0])
    done_short = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    val_short  = torch.zeros(5, 1)

    # 긴 에피소드 (20 step) terminal=+5
    rew_long  = torch.zeros(20);  rew_long[-1]  = 5.0
    done_long = torch.zeros(20);  done_long[-1] = 1.0
    val_long  = torch.zeros(20, 1)

    # gamma=1.0, lam=1.0 (pure Monte Carlo return)
    _, ret_short = compute_gae(
        rew_short, done_short, val_short, last_value=0.0, gamma=1.0, lam=1.0,
    )
    _, ret_long = compute_gae(
        rew_long, done_long, val_long, last_value=0.0, gamma=1.0, lam=1.0,
    )

    assert torch.isclose(ret_short[0], ret_long[0], atol=1e-5), (
        f"gamma=1.0 length invariance 위반: "
        f"short={ret_short[0].item()}, long={ret_long[0].item()}"
    )
    assert torch.isclose(ret_short[0], torch.tensor(5.0), atol=1e-5), \
        "Monte Carlo return은 terminal reward 그대로여야 함"


# ── 불변식 4: gamma<1이면 length invariance 깨짐 (대조군) ────────────────────

def test_gamma_below_one_breaks_length_invariance():
    """gamma=0.99면 길이에 따라 return이 달라짐 — gamma=1.0 선택의 동기 검증."""
    rew_short  = torch.zeros(5);  rew_short[-1]  = 5.0
    done_short = torch.zeros(5);  done_short[-1] = 1.0
    val_short  = torch.zeros(5, 1)

    rew_long  = torch.zeros(100); rew_long[-1]  = 5.0
    done_long = torch.zeros(100); done_long[-1] = 1.0
    val_long  = torch.zeros(100, 1)

    _, ret_short = compute_gae(
        rew_short, done_short, val_short, last_value=0.0, gamma=0.99, lam=1.0,
    )
    _, ret_long = compute_gae(
        rew_long, done_long, val_long, last_value=0.0, gamma=0.99, lam=1.0,
    )

    # 긴 에피소드의 초반 return은 gamma^99 만큼 할인됨
    assert ret_long[0].item() < ret_short[0].item() * 0.5, (
        f"gamma<1이 길이에 따른 discount를 만들지 않음 — "
        f"short={ret_short[0].item()}, long={ret_long[0].item()}"
    )
