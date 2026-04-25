"""Source planet gather 정확성 검증.

핵심 계약:
  - 유효 idx → src_t[b, f] == p_t[b, fp_idx[b, f]]
  - -1 sentinel → valid mask=False (fusion 기여 없음)
  - 범위 초과 (e.g. 999, -100) → clamp 후에도 valid mask=False
  - 배치/플릿 mixed (일부 valid, 일부 invalid) → per-fleet 독립 처리
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


def _setup_pf(B=2):
    torch.manual_seed(42)
    f_t = torch.randn(B, MAX_FLEETS, EMBED_DIM)
    p_t = torch.randn(B, MAX_PLANETS, EMBED_DIM)
    return p_t, f_t


def _patch_fusion_to_expose_gather(model):
    """fusion weight 를 고정해 fusion 출력에서 gather 결과를 역추적 가능하게.

    설정:
      gate.weight = 0,  gate.bias = +100      → sigmoid(100) ≈ 1
      value.weight = [0(E,E), I(E,E)], bias=0 → value([f;src]) = src
    결과:
      out = f_t + 1 * tanh(src) * valid = f_t + tanh(p_t[fp_idx]) * valid
    f_t = 0 으로 두면 out = tanh(p_t[fp_idx]) * valid 로 단순화 → axis/expand/clamp 버그 직접 검증.
    """
    E = EMBED_DIM
    with torch.no_grad():
        model.fleet_source_gate.weight.zero_()
        model.fleet_source_gate.bias.fill_(100.0)
        model.fleet_source_value.weight.zero_()
        model.fleet_source_value.weight[:, E:].copy_(torch.eye(E))
        model.fleet_source_value.bias.zero_()


def test_valid_idx_gathers_correct_source(model):
    """유효 idx → 실제 _fleet_source_fuse 출력이 tanh(p_t[fp_idx]) 와 일치.

    production gather 경로 (axis=1, expand to E, clamp) 를 직접 호출하여
    출력으로 검증 → 알고리즘 재구현 없이 production 의 실제 동작을 lock-down.
    """
    _patch_fusion_to_expose_gather(model)
    E = EMBED_DIM

    torch.manual_seed(42)
    B  = 2
    p_t = torch.randn(B, MAX_PLANETS, E)
    f_t = torch.zeros(B, MAX_FLEETS, E)

    # 다양한 batch/fleet/planet 조합 — 경계 (0, MAX_PLANETS-1) 포함
    valid_pairs = [
        (0, 0, 0), (0, 1, 7), (0, 2, 39), (0, 17, 23), (0, 99, 5),
        (1, 0, 38), (1, 5, 11), (1, 50, 0), (1, 99, 30),
    ]
    fp_idx = torch.full((B, MAX_FLEETS), -1.0)
    for b, f, p in valid_pairs:
        fp_idx[b, f] = float(p)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)

    # valid 위치: out == tanh(p_t[fp_idx])  (gather + axis 검증)
    for b, f, p in valid_pairs:
        expected = torch.tanh(p_t[b, p])
        assert torch.allclose(out[b, f], expected, atol=1e-4), \
            f"gather mismatch batch={b} fleet={f} idx={p}: " \
            f"out[:3]={out[b,f,:3].tolist()} expected[:3]={expected[:3].tolist()}"

    # invalid 위치 spot check: f_t=0 이었으므로 out 도 0
    valid_keys = {(b, f) for b, f, _ in valid_pairs}
    for b, f in [(0, 3), (0, 50), (1, 1), (1, 80)]:
        assert (b, f) not in valid_keys
        assert torch.allclose(out[b, f], torch.zeros(E), atol=1e-6), \
            f"invalid fleet ({b},{f}) 에서 mask 가 새고 있음"


def test_gather_with_swapped_batch_axis_would_fail(model):
    """배치 축이 바뀌면 결과가 달라야 함 — gather 가 dim=1 (planet axis) 인지 회귀.

    같은 fp_idx 를 두 배치에 다르게 넣으면 출력이 달라야 정상.
    (만약 axis 가 0 이거나 expand 가 잘못되면 batch 간 누설 발생)
    """
    _patch_fusion_to_expose_gather(model)
    E = EMBED_DIM

    torch.manual_seed(99)
    p_t = torch.randn(2, MAX_PLANETS, E)
    f_t = torch.zeros(2, MAX_FLEETS, E)

    # batch 0 fleet 0 → planet 5,  batch 1 fleet 0 → planet 30
    fp_idx = torch.full((2, MAX_FLEETS), -1.0)
    fp_idx[0, 0] = 5.0
    fp_idx[1, 0] = 30.0

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)

    # 각 배치는 자기 p_t 의 해당 idx 를 봐야 함 (cross-batch leak 없어야)
    assert torch.allclose(out[0, 0], torch.tanh(p_t[0, 5]),  atol=1e-4)
    assert torch.allclose(out[1, 0], torch.tanh(p_t[1, 30]), atol=1e-4)
    # 다른 배치 p_t 와 헷갈림 없는지
    assert not torch.allclose(out[0, 0], torch.tanh(p_t[1, 5]),  atol=1e-2)
    assert not torch.allclose(out[1, 0], torch.tanh(p_t[0, 30]), atol=1e-2)


def test_invalid_sentinel_blocks_fusion(model):
    """모든 idx=-1 → fusion 출력 == 입력 (residual identity)."""
    B = 2
    p_t, f_t = _setup_pf(B)
    fp_idx = torch.full((B, MAX_FLEETS), -1.0)

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    assert torch.equal(out, f_t), \
        "invalid(-1) sentinel 인데 f_t 가 변경됨 — valid mask 가 fusion 을 차단해야 함"


def test_out_of_range_idx_blocks_fusion(model):
    """범위 초과 idx (음/양 양쪽) → clamp 되더라도 fusion 차단."""
    B = 2
    p_t, f_t = _setup_pf(B)
    # MAX_PLANETS=40 기준 양/음 둘 다 out-of-range
    fp_idx = torch.tensor([
        [999.0] * MAX_FLEETS,           # +∞ 쪽
        [-100.0] * MAX_FLEETS,          # -∞ 쪽
    ])

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    assert torch.equal(out, f_t), \
        "out-of-range idx 인데 f_t 가 변경됨 — clamp 만으로는 안 되고 valid mask 가 차단해야 함"


def test_boundary_idx_treated_as_invalid(model):
    """idx == MAX_PLANETS (정확히 경계 밖) → invalid."""
    B = 1
    p_t, f_t = _setup_pf(B)
    fp_idx = torch.full((B, MAX_FLEETS), float(MAX_PLANETS))   # = MAX_PLANETS

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)
    assert torch.equal(out, f_t), \
        "idx == MAX_PLANETS 는 valid 범위 [0, MAX_PLANETS) 밖 — invalid 처리되어야 함"


def test_mixed_valid_invalid_per_fleet(model):
    """같은 배치 내 일부 fleet 은 valid, 일부는 invalid → 독립 처리."""
    B = 1
    p_t, f_t = _setup_pf(B)
    f_t_orig = f_t.clone()

    fp_idx = torch.full((B, MAX_FLEETS), -1.0)
    valid_fleets = [0, 5, 10, 50]
    fp_idx[0, 0]  = 7.0
    fp_idx[0, 5]  = 20.0
    fp_idx[0, 10] = 35.0
    fp_idx[0, 50] = 0.0

    out = model._fleet_source_fuse(f_t, p_t, fp_idx)

    # invalid 위치는 변화 없음
    invalid_mask = torch.ones(MAX_FLEETS, dtype=torch.bool)
    for f in valid_fleets:
        invalid_mask[f] = False
    assert torch.equal(out[0, invalid_mask], f_t_orig[0, invalid_mask]), \
        "invalid fleet 위치가 변경됨 — per-fleet mask 가 작동하지 않음"

    # valid 위치는 변화 *가능* (확률적, 모든 weight 가 0 이 아닌 한 거의 확실)
    diff = (out[0, valid_fleets] - f_t_orig[0, valid_fleets]).abs().sum()
    assert diff > 0, "valid fleet 인데 fusion 기여가 0 — gate/cand 가 동작 안 함"


def test_zero_idx_is_valid_not_sentinel(model):
    """idx == 0 은 유효한 행성이지 sentinel 이 아님 — 헷갈림 방지 회귀."""
    B = 1
    p_t, f_t = _setup_pf(B)

    # 모든 fleet 의 source = planet 0
    fp_idx = torch.zeros(B, MAX_FLEETS)
    out = model._fleet_source_fuse(f_t, p_t, fp_idx)

    # idx=0 valid → fusion 발생 → f_t 변화 있어야 함
    assert not torch.equal(out, f_t), \
        "idx=0 이 sentinel 처럼 무시됨 — 0 은 유효한 planet idx"
