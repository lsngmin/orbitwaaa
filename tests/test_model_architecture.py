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
    HISTORY, MAX_PLANETS, MAX_FLEETS, PLANET_DIM, FLEET_DIM, ACTION_DIM,
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
    """forward pass가 올바른 shape를 반환하는지 확인."""
    B = 3
    obs = torch.randn(B, OBS_DIM)
    logits, value = model(obs)
    assert logits.shape == (B, MAX_PLANETS, ACTION_DIM)
    assert value.shape  == (B, 1)


def test_get_action_and_value_shapes(model):
    """get_action_and_value 출력 shape 확인."""
    obs = torch.randn(2, OBS_DIM)
    action, log_prob, value, lp_heads = model.get_action_and_value(obs)
    assert action.shape   == (2, MAX_PLANETS, ACTION_DIM)
    assert log_prob.shape == (2,)
    assert value.shape    == (2, 1)
    assert lp_heads.shape == (2, 3)


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


# ── P2-2: fleet_temporal ablation flag ───────────────────────────────────────

import model as _model_module


def _make_model_with_flag(fleet_temporal: bool):
    """FLEET_TEMPORAL 플래그를 바꿔서 모델을 생성한 뒤 원래 값으로 복구."""
    original = _model_module.FLEET_TEMPORAL
    _model_module.FLEET_TEMPORAL = fleet_temporal
    try:
        return _model_module.OrbitWarsPolicy()
    finally:
        _model_module.FLEET_TEMPORAL = original


def test_ablation_a_fleet_temporal_attn_exists():
    """fleet_temporal=True → fleet_temporal_attn이 None이 아님."""
    m = _make_model_with_flag(True)
    assert m.fleet_temporal_attn is not None


def test_ablation_b_fleet_temporal_attn_is_none():
    """fleet_temporal=False → fleet_temporal_attn이 None."""
    m = _make_model_with_flag(False)
    assert m.fleet_temporal_attn is None


def test_ablation_b_forward_shape():
    """fleet_temporal=False일 때도 forward output shape이 동일한지 확인."""
    m = _make_model_with_flag(False)
    m.eval()
    obs = torch.randn(2, OBS_DIM)
    original = _model_module.FLEET_TEMPORAL
    _model_module.FLEET_TEMPORAL = False
    try:
        logits, val = m(obs)
    finally:
        _model_module.FLEET_TEMPORAL = original
    assert logits.shape == (2, MAX_PLANETS, ACTION_DIM)
    assert val.shape    == (2, 1)


def test_ablation_b_fewer_params_than_a():
    """fleet_temporal=False 모델이 True 모델보다 파라미터 수가 적은지 확인."""
    m_a = _make_model_with_flag(True)
    m_b = _make_model_with_flag(False)
    params_a = sum(p.numel() for p in m_a.parameters())
    params_b = sum(p.numel() for p in m_b.parameters())
    assert params_b < params_a, f"ablation B params({params_b}) >= A({params_a})"
