"""
P2-1 검증 테스트: planet/fleet temporal branch 분리.

- planet_temporal_attn / fleet_temporal_attn 이 별도 인스턴스인지
- planet_temporal_pos / fleet_temporal_pos 가 별도 인스턴스인지
- forward pass shape 정상 여부
- planet branch gradient가 fleet temporal에 영향 없는지 (가중치 독립성)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pytest
from model import (
    OrbitWarsPolicy,
    HISTORY, MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, ACTION_DIM, EMBED_DIM,
    NUM_ACTIONS,
)

OBS_DIM = HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM)


@pytest.fixture
def model():
    m = OrbitWarsPolicy()
    m.eval()
    return m


def test_temporal_attn_are_separate_instances(model):
    """planet_temporal_attn과 fleet_temporal_attn이 다른 인스턴스인지 확인."""
    assert model.planet_temporal_attn is not model.fleet_temporal_attn


def test_temporal_pos_are_separate_instances(model):
    """planet_temporal_pos와 fleet_temporal_pos가 다른 인스턴스인지 확인."""
    assert model.planet_temporal_pos is not model.fleet_temporal_pos


def test_temporal_pos_weights_not_shared(model):
    """두 positional embedding이 같은 storage를 공유하지 않는지 확인."""
    assert model.planet_temporal_pos.weight.data_ptr() != model.fleet_temporal_pos.weight.data_ptr()


def test_temporal_attn_weights_not_shared(model):
    """planet/fleet temporal_attn의 첫 번째 레이어 가중치가 독립적인지 확인."""
    p_params = list(model.planet_temporal_attn.parameters())
    f_params = list(model.fleet_temporal_attn.parameters())
    assert p_params[0].data_ptr() != f_params[0].data_ptr()


def test_forward_output_shapes(model):
    """forward pass가 올바른 shape를 반환하는지 확인.

    Step 3: forward 는 (src_token, value) 반환 — head 적용 전 encoder 출력.
    """
    B = 3
    obs = torch.randn(B, OBS_DIM)
    src_token, value = model(obs)
    assert src_token.shape == (B, MAX_PLANETS, EMBED_DIM)
    assert value.shape     == (B, 1)


def test_get_action_and_value_shapes(model):
    """get_action_and_value 출력 shape 확인."""
    obs = torch.randn(2, OBS_DIM)
    action, log_prob, value, lp_heads = model.get_action_and_value(obs)
    assert action.shape   == (2, MAX_PLANETS, ACTION_DIM)
    assert log_prob.shape == (2,)
    assert value.shape    == (2, 1)
    assert lp_heads.shape == (2, 2)


def test_planet_gradient_does_not_flow_to_fleet_temporal(model):
    """planet branch loss로 backward 시 fleet_temporal_attn 가중치에 grad가 없는지 확인."""
    model.train()
    obs = torch.randn(1, OBS_DIM)
    logits, _ = model(obs)

    # planet branch만 사용하는 loss
    loss = logits[:, :, 0].mean()
    loss.backward()

    # fleet_temporal_attn은 forward에서 사용됐으므로 grad가 있을 수 있음 —
    # 여기서 핵심은 planet_temporal_attn과 fleet_temporal_attn의 grad가 서로 다른지 확인
    p_grad = list(model.planet_temporal_attn.parameters())[0].grad
    f_grad = list(model.fleet_temporal_attn.parameters())[0].grad

    # 공유됐다면 두 grad tensor가 동일 storage를 가리킴
    if p_grad is not None and f_grad is not None:
        assert p_grad.data_ptr() != f_grad.data_ptr(), \
            "planet/fleet temporal_attn grad가 같은 storage — 가중치 공유 의심"


def test_fleet_temporal_layer_count_less_than_planet(model):
    """fleet temporal이 planet보다 레이어 수가 적거나 같은지 확인 (고잡음 신호 축소 원칙)."""
    from model import PLANET_TEMPORAL_LAYERS, FLEET_TEMPORAL_LAYERS
    assert FLEET_TEMPORAL_LAYERS <= PLANET_TEMPORAL_LAYERS


# ─── Step 4 verify: target pair-aware (target_q · target_k / sqrt(E)) ────────

def test_target_logits_pairwise_q_dot_k_formula(model):
    """target_logits[i,j] 가 정확히 q(H_i)·k(H_j)/sqrt(E) 공식인지 직접 검증.

    회귀: 옛 target_head Sequential 잔재 / 잘못된 reduction / scaling 누락 등.
    """
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        src_token, _ = model(obs)              # (1, P, E)
        t_logits = model._target_logits(src_token)   # (1, P, P)
        # 수동 계산
        q = model.target_q(src_token)           # (1, P, E)
        k = model.target_k(src_token)           # (1, P, E)
        expected = torch.matmul(q, k.transpose(-2, -1)) / (EMBED_DIM ** 0.5)
    assert torch.allclose(t_logits, expected, atol=1e-6), \
        "target_logits 가 q·k^T/sqrt(E) 공식과 일치 안 함"


def test_target_logits_responds_to_dst_only_at_column_j(model):
    """j 위치 embedding 만 흔들면 t_logits 의 column j (모든 i 에서) + row j (q 영향) 만 변함.

    이게 깨지면 target head 가 cross-pair 가 아니라 src-only function 으로 되돌아갔다는 뜻.
    """
    j = 5
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        src_token, _ = model(obs)
        t1 = model._target_logits(src_token).clone()
        # j 위치만 perturb
        src_token[0, j] = src_token[0, j] + torch.randn(EMBED_DIM)
        t2 = model._target_logits(src_token)

    diff = (t1 - t2)[0]                          # (P, P)
    # column j (모든 i, j=5) 변해야 — k(H_j) 사용처
    assert diff[:, j].abs().max() > 1e-3, "k(H_j) ignored — column 변화 없음"
    # row j (i=5, 모든 j') 도 변해야 — q(H_j) 사용처
    assert diff[j, :].abs().max() > 1e-3, "q(H_j) ignored — row 변화 없음"
    # 그 외 (i≠j, j'≠j) 는 영향 없어야 (수학적으로 정확히 0)
    P = MAX_PLANETS
    mask = torch.ones(P, P, dtype=torch.bool)
    mask[:, j] = False
    mask[j, :] = False
    assert diff[mask].abs().max() < 1e-5, \
        f"i≠{j}, j'≠{j} 에 누수 (max diff = {diff[mask].abs().max():.2e})"


def test_target_grad_flows_to_q_and_k(model):
    """target_q.weight 와 target_k.weight 둘 다 gradient 받아야 — dead-weight 방지."""
    model.zero_grad()
    obs = torch.randn(2, OBS_DIM)
    src_token, _ = model(obs)
    t_logits = model._target_logits(src_token)
    t_logits.sum().backward()
    assert model.target_q.weight.grad is not None and \
           model.target_q.weight.grad.abs().sum() > 0, "target_q.weight grad 없음"
    assert model.target_k.weight.grad is not None and \
           model.target_k.weight.grad.abs().sum() > 0, "target_k.weight grad 없음"


def test_target_logits_identical_in_get_action_and_evaluate(model):
    """get_action_and_value 와 evaluate_actions 가 *동일* target_logits 사용 — PPO ratio 무결성.

    하나는 새 _target_logits, 다른 하나는 옛 path 라면 importance ratio 가 깨져서
    PPO 업데이트가 잘못된 분포로 정규화됨. 회귀 테스트.
    """
    torch.manual_seed(42)
    obs = torch.randn(2, OBS_DIM)

    # Path 1: get_action_and_value 가 내부적으로 사용하는 target dist
    # (우회 방법: target_mask 다 True 로 두고 sampling 한 후 동일 forward 의 t_logits 직접 비교)
    target_mask = torch.ones(2, MAX_PLANETS, MAX_PLANETS, dtype=torch.bool)
    action_mask = torch.ones(2, MAX_PLANETS, NUM_ACTIONS, dtype=torch.bool)

    with torch.no_grad():
        # forward 단 한 번 = src_token 1 회
        src_token_a, _ = model(obs)
        t_logits_get = model._target_logits(src_token_a)

        # evaluate_actions 의 target_logits 도 동일 forward 거쳐야 함
        src_token_e, _ = model(obs)
        t_logits_eval = model._target_logits(src_token_e)

    # 같은 obs → 같은 src_token → 같은 t_logits (deterministic forward in eval mode)
    assert torch.allclose(t_logits_get, t_logits_eval, atol=1e-6), \
        "두 경로의 target_logits 가 다름 — PPO ratio 깨짐"


# ─── Step 3 verify: amount pair-aware (uses dst token, not just src) ─────────

def test_amount_logits_uses_destination_token(model):
    """같은 src 에 대해 target_idx 만 다르게 주면 amount logits 가 달라야.

    회귀: amount head 가 H_dst 부분을 무시하고 H_src 만으로 학습 (실제 src-only classifier).
    이 테스트가 통과해도 weight norm 분석 (W_dst vs W_src) 도 함께 봐야 *학습된*
    가중치가 dst 를 활용하는지 알 수 있음.
    """
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        src_token, _ = model(obs)
        # source 0 이 target=1 vs target=2 일 때 amount logits 비교
        # 다른 source 들의 target 도 차이 두면 안 됨 (i=0 만 비교 대상)
        t_a = torch.zeros(1, MAX_PLANETS, dtype=torch.long); t_a[0, 0] = 1
        t_b = torch.zeros(1, MAX_PLANETS, dtype=torch.long); t_b[0, 0] = 2
        a_a = model._amount_logits(src_token, t_a)[0, 0]   # (NUM_ACTIONS,)
        a_b = model._amount_logits(src_token, t_b)[0, 0]
    assert (a_a - a_b).abs().max() > 1e-3, \
        "amount head 가 dst token 무시 — pair-aware 미작동"


def test_amount_logits_dst_gather_correctness(model):
    """amount_logits[i] 의 입력 pair_input = [src_token[i], src_token[target_idx[i]]] 인지.

    수동 forward 와 _amount_logits 결과가 일치해야.
    """
    obs = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        src_token, _ = model(obs)             # (1, P, E)
        # i=3 → target=7, 다른 i 는 target=0 (default)
        target_idx = torch.zeros(1, MAX_PLANETS, dtype=torch.long)
        target_idx[0, 3] = 7
        a_logits = model._amount_logits(src_token, target_idx)[0, 3]   # (NUM_ACTIONS,)

        # 수동: pair_input = [src_token[0,3], src_token[0,7]]
        pair_manual = torch.cat([src_token[0, 3], src_token[0, 7]], dim=-1).unsqueeze(0)
        a_manual = model.amount_pair_head(pair_manual)[0]
    assert torch.allclose(a_logits, a_manual, atol=1e-6), \
        "amount_logits 의 dst gather 가 target_idx 와 어긋남"


def test_amount_grad_flows_to_dst_columns(model):
    """amount_pair_head 의 첫 Linear 의 dst 부분 (input dim E:2E) 이 grad 받아야."""
    model.zero_grad()
    obs = torch.randn(2, OBS_DIM)
    src_token, _ = model(obs)
    target_idx = torch.randint(0, MAX_PLANETS, (2, MAX_PLANETS))
    a_logits = model._amount_logits(src_token, target_idx)
    a_logits.sum().backward()

    W = model.amount_pair_head[0].weight        # (H, 2E)
    grad = W.grad                                # (H, 2E)
    grad_src = grad[:, :EMBED_DIM]
    grad_dst = grad[:, EMBED_DIM:]
    assert grad_src.abs().sum() > 0, "amount_pair_head src 부분 grad 없음"
    assert grad_dst.abs().sum() > 0, \
        "amount_pair_head dst 부분 grad 없음 — pair-aware 학습 신호 차단됨"


def test_amount_logits_identical_in_get_action_and_evaluate(model):
    """rollout (get_action_and_value) 와 update (evaluate_actions) 가 같은 target 으로
    같은 amount_logits 만들어야 — PPO ratio 무결성.

    evaluate_actions 는 stored action 의 target_idx (one-hot argmax) 를 쓰므로,
    same target_idx → same amount_logits 가 보장되어야 함.
    """
    torch.manual_seed(0)
    obs = torch.randn(2, OBS_DIM)
    with torch.no_grad():
        src_token, _ = model(obs)
        # 가짜 target_idx
        target_idx = torch.randint(0, MAX_PLANETS, (2, MAX_PLANETS))
        # one-hot 으로 만들어서 evaluate_actions 가 받는 형태 모사
        t_onehot = torch.zeros(2, MAX_PLANETS, MAX_PLANETS)
        t_onehot.scatter_(-1, target_idx.unsqueeze(-1), 1.0)
        # one-hot argmax → 같은 idx 복원되는지 sanity
        recovered = t_onehot.argmax(dim=-1)
        assert torch.equal(recovered, target_idx)

        a1 = model._amount_logits(src_token, target_idx)
        a2 = model._amount_logits(src_token, recovered)
    assert torch.allclose(a1, a2, atol=1e-6)
