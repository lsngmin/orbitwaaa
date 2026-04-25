"""Gated residual fusion 동작 검증.

핵심 계약:
  - residual identity: 모든 fleet invalid → f_t 변화 없음
  - 부분 valid: invalid fleet 은 불변, valid fleet 만 변경
  - gate/value 모듈: 차원/존재 회귀
  - bounded output: gate ∈ [0,1], cand ∈ [-1,1] → 추가량 magnitude 제한

비고:
  fusion 은 model._fleet_source_fuse 로 분리되어 있어 unit-test 가능.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import pytest
from model import OrbitWarsPolicy, EMBED_DIM, MAX_PLANETS, MAX_FLEETS


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = OrbitWarsPolicy()
    m.eval()
    return m


# ── 모듈 회귀 ────────────────────────────────────────────────────────────────


def test_fusion_modules_exist_with_correct_dims(model):
    """fleet_source_gate / fleet_source_value 가 [2E → E] 매핑인지."""
    assert hasattr(model, "fleet_source_gate"),  "fleet_source_gate 누락"
    assert hasattr(model, "fleet_source_value"), "fleet_source_value 누락"

    gw = model.fleet_source_gate.weight
    vw = model.fleet_source_value.weight
    assert gw.shape == (EMBED_DIM, EMBED_DIM * 2), f"gate weight {gw.shape}"
    assert vw.shape == (EMBED_DIM, EMBED_DIM * 2), f"value weight {vw.shape}"


# ── 핵심: residual identity ──────────────────────────────────────────────────


def test_residual_identity_when_all_invalid(model):
    """모든 fleet invalid(-1) → f_t 가 정확히 그대로 보존."""
    torch.manual_seed(7)
    B = 4
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM)
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM)
    fp_idx = torch.full((B, MAX_FLEETS), -1.0)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    # bit-exact: residual identity (mask 가 fusion 항을 0 으로)
    assert torch.equal(out, f_t), \
        "residual identity 깨짐 — invalid mask 가 fusion 기여를 완전 차단해야 함"


def test_residual_identity_with_random_pt(model):
    """invalid mask 일 때 p_t 값이 어떻게 변해도 f_t 출력 영향 없음 (gather 의존성 차단)."""
    torch.manual_seed(11)
    B = 2
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM)
    fp_idx = torch.full((B, MAX_FLEETS), -1.0)

    # p_t 두 가지로 다르게 줘도 출력 동일해야 함
    p_t1 = torch.randn(B, MAX_PLANETS, EMBED_DIM)
    p_t2 = torch.randn(B, MAX_PLANETS, EMBED_DIM) * 100.0

    out1 = model._fleet_source_fuse(f_t, p_t1, fp_idx)
    out2 = model._fleet_source_fuse(f_t, p_t2, fp_idx)
    assert torch.equal(out1, out2), \
        "invalid mask 인데 p_t 변경이 출력에 새는 중 — mask 가 gather 결과를 막아야 함"


# ── 핵심: 부분 valid ─────────────────────────────────────────────────────────


def test_invalid_fleets_unchanged_when_others_valid(model):
    """일부 fleet 만 valid → invalid fleet 은 bit-exact 보존."""
    torch.manual_seed(13)
    B = 1
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM)
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM)

    fp_idx = torch.full((B, MAX_FLEETS), -1.0)
    valid_set = {2, 17, 33, 99}
    for f in valid_set:
        fp_idx[0, f] = float(f % MAX_PLANETS)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    for f in range(MAX_FLEETS):
        if f in valid_set:
            continue
        assert torch.equal(out[0, f], f_t[0, f]), \
            f"fleet {f} 는 invalid 인데 fusion 으로 변경됨"


def test_valid_fleets_change(model):
    """valid fleet 은 거의 확실히 변화 (gate/cand 가 0 일 확률 무시)."""
    torch.manual_seed(17)
    B = 2
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM)
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM)
    fp_idx = torch.zeros(B, MAX_FLEETS)   # 전부 valid (idx=0)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    diff = (out - f_t).abs().mean()
    assert diff > 1e-4, f"valid 인데 fusion 기여가 거의 0: {diff}"


# ── bounded output ──────────────────────────────────────────────────────────


def test_fusion_addition_magnitude_bounded(model):
    """추가량 |out - f_t| ≤ 1.0 per-element (gate ≤ 1, |cand| ≤ 1)."""
    torch.manual_seed(19)
    B = 3
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM) * 5.0   # 큰 값으로
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM) * 5.0
    fp_idx = torch.zeros(B, MAX_FLEETS)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    delta = (out - f_t).abs()
    assert delta.max().item() <= 1.0 + 1e-5, \
        f"추가량이 1.0 초과: {delta.max()} — gate/tanh 범위 깨짐"


# ── gradient flow ────────────────────────────────────────────────────────────


def test_gradient_flows_through_fusion(model):
    """fusion 출력 loss → gate/value Linear 까지 grad 흐름."""
    model.train()
    B = 1
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM, requires_grad=True)
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM, requires_grad=True)
    fp_idx = torch.zeros(B, MAX_FLEETS)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    out.sum().backward()

    assert model.fleet_source_gate.weight.grad is not None
    assert model.fleet_source_value.weight.grad is not None
    assert model.fleet_source_gate.weight.grad.abs().sum().item() > 0
    assert model.fleet_source_value.weight.grad.abs().sum().item() > 0
