"""LaunchCaptureTracker 와 events 단위 동작 검증.

검증:
- register_launches 가 metadata 누적
- update_step 이 prev_map vs curr_obs 비교로 capture 검출
- attribution: linked_launches 가 target_id 일치하는 window 내 launch 들
- window_turns 만료 처리
- reset_episode 가 state 비움
- 변동 없는 행성은 event 안 만듦
- player 인자가 attribution 자체에는 영향 없음 (component 가 prev/new owner 로 판단)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from reward import LaunchMetadata, CaptureEvent, LaunchCaptureTracker


# ── helpers ───────────────────────────────────────────────────────────────────

def _meta(turn, target_id, source_id=0, ships_sent=10):
    return LaunchMetadata(
        turn=turn, source_id=source_id, target_id=target_id,
        target_owner_at_launch=-1, target_kind="attack",
        ships_sent=ships_sent, eta_turns=3,
        req_over_src=0.5, req_over_prod=2.0,
        nearest_neutral_rank=1, target_prod=2.0,
    )


def _obs(planets):
    """planet tuple: (id, owner, x, y, radius, ships, production)."""
    return {"planets": planets}


# ── register / capture / attribution ──────────────────────────────────────────

def test_register_then_resolve_links_launch():
    t = LaunchCaptureTracker(window_turns=10)
    t.register_launches([_meta(turn=1, target_id=5)])
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 10, 3)]),
        player=0, turn=2,
    )
    assert len(events) == 1
    e = events[0]
    assert e.planet_id == 5
    assert e.prev_owner == -1 and e.new_owner == 0
    assert len(e.linked_launches) == 1
    assert e.linked_launches[0].target_id == 5


def test_no_change_no_event():
    """소유권 변동 없으면 event 0개."""
    t = LaunchCaptureTracker()
    t.register_launches([_meta(turn=0, target_id=5)])
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, -1, 0, 0, 0, 0, 3)]),  # 여전히 중립
        player=0, turn=1,
    )
    assert events == []


def test_capture_without_matching_launch():
    """linked_launches 가 0개일 수 있음 — natural capture / 만료 / target 불일치."""
    t = LaunchCaptureTracker()
    # launch 가 다른 행성을 target 으로 함.
    t.register_launches([_meta(turn=0, target_id=99)])
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 10, 3)]),
        player=0, turn=1,
    )
    assert len(events) == 1
    assert events[0].linked_launches == []


def test_multiple_launches_same_target_all_linked():
    """다중 source 협조 — 같은 target 의 모든 window 내 launch 가 linked 에 들어감."""
    t = LaunchCaptureTracker()
    t.register_launches([
        _meta(turn=0, target_id=5, source_id=10, ships_sent=8),
        _meta(turn=1, target_id=5, source_id=20, ships_sent=12),
    ])
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 0, 3)]),
        player=0, turn=2,
    )
    assert len(events) == 1
    src_ids = sorted(l.source_id for l in events[0].linked_launches)
    assert src_ids == [10, 20]


def test_window_expiry_drops_old_launches():
    """window 보다 오래된 launch 는 attribution 후보에서 제외."""
    t = LaunchCaptureTracker(window_turns=5)
    t.register_launches([_meta(turn=0, target_id=5)])  # 오래됨
    t.register_launches([_meta(turn=4, target_id=5)])  # window 내
    # turn=10 → cutoff = 10-5 = 5. turn=0 만료, turn=4 만료 (4 < 5)
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 10, 3)]),
        player=0, turn=10,
    )
    assert len(events) == 1
    assert events[0].linked_launches == []  # 둘 다 만료


def test_window_boundary_inclusive():
    """cutoff = turn - window. 정확히 그 turn 의 launch 는 살아있어야 함."""
    t = LaunchCaptureTracker(window_turns=5)
    t.register_launches([_meta(turn=5, target_id=5)])  # turn=10 - 5 = 5, 경계
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 10, 3)]),
        player=0, turn=10,
    )
    assert len(events) == 1
    assert len(events[0].linked_launches) == 1


def test_reset_episode_clears_state():
    t = LaunchCaptureTracker()
    t.register_launches([_meta(turn=0, target_id=5)])
    assert len(t) == 1
    t.reset_episode()
    assert len(t) == 0
    # reset 후 attribution 도 끊김
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_obs([(5, 0, 0, 0, 0, 10, 3)]),
        player=0, turn=1,
    )
    assert len(events) == 1
    assert events[0].linked_launches == []


def test_loss_event_also_emitted():
    """소유권 변동이면 모두 event 생성 — 점령뿐 아니라 잃는 것도 (component 가 분기)."""
    t = LaunchCaptureTracker()
    events = t.update_step(
        prev_map={5: (0, 3)},
        curr_obs=_obs([(5, 1, 0, 0, 0, 10, 3)]),  # me → enemy
        player=0, turn=1,
    )
    assert len(events) == 1
    assert events[0].prev_owner == 0 and events[0].new_owner == 1
    assert events[0].linked_launches == []  # 잃는 건 보통 attribution 안 됨


def test_curr_obs_attr_form_supported():
    """curr_obs 가 dict 가 아니라 attr 객체여도 동작 (env raw_obs 호환성)."""
    class _PlanetObj:
        def __init__(self, pid, owner, prod):
            self.id, self.owner, self.production = pid, owner, prod

    class _Obs:
        planets = [_PlanetObj(5, 0, 3)]

    t = LaunchCaptureTracker()
    events = t.update_step(
        prev_map={5: (-1, 3)},
        curr_obs=_Obs(),
        player=0, turn=1,
    )
    assert len(events) == 1
    assert events[0].planet_id == 5


def test_curr_obs_none_yields_no_events():
    t = LaunchCaptureTracker()
    events = t.update_step(prev_map={5: (-1, 3)}, curr_obs=None, player=0, turn=1)
    assert events == []
