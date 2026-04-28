"""Reward trackers — cross-step state 보관용.

규약 (reward/__init__.py docstring 의 stateful reward 분리 정책):
- component 함수는 stateless (ctx 만 받음). cross-step state 가 필요한 경우
  여기 tracker 가 보관 → ctx.capture_events 로 component 에 들어가는 흐름.
- tracker 는 train.py 가 명시적으로 register / update / reset 호출.
- 진단용 HitRateTracker (utils/hit_tracker.py) 와 책임 분리:
  * HitRateTracker: fleet 도착/충돌/점령 결과 진단 통계 (CSV)
  * LaunchCaptureTracker: reward 가 필요한 launch metadata + capture 검출만

핵심 알고리즘 (LaunchCaptureTracker):
    register_launches(launches): 매 step 발사된 LaunchMetadata 누적.
    update_step(prev_map, curr_obs, player, turn) -> List[CaptureEvent]:
        1. window 만료된 launch 제거.
        2. prev_map vs curr_obs 비교로 owner 전환 검출.
        3. 각 transition 에 대해 target_id 일치하는 window 내 launches 를
           linked_launches 로 묶어 CaptureEvent 생성.

attribution 정책: window 안 launch 중 target_id 일치하는 모든 것을 linked_launches 에
넣음. "대표 1개" 선택은 component 책임 (예: turn 가장 가까운 1개, 또는 ships_sent 큰 것).

reset_episode: 에피소드 종료 시 호출. 이전 게임의 launch 가 다음 게임에 묻지 않도록.
"""
from __future__ import annotations

from typing import Iterable, List

from reward.events import LaunchMetadata, CaptureEvent


def _curr_planets(curr_obs):
    """curr_obs (dict / attr / None) → planets 시퀀스."""
    if isinstance(curr_obs, dict):
        return curr_obs.get("planets", [])
    if curr_obs is None:
        return []
    return getattr(curr_obs, "planets", [])


def _planet_attr(p, idx, attr):
    """tuple/list 면 idx, 객체면 attr 로 접근 (raw_obs 표현 양쪽 지원)."""
    if isinstance(p, (list, tuple)):
        return p[idx]
    return getattr(p, attr)


class LaunchCaptureTracker:
    """Launch metadata 보관 + capture 검출 + attribution.

    수명: episode 한 게임. reset_episode() 가 cumulative state 초기화.

    Window 정책:
        window_turns 보다 오래된 launch 는 만료 — capture 가 늦게 발생해도
        그 launch 는 attribution 후보에서 제외. 너무 길면 무관한 launch 가 묶일 수 있음.
        기본 30 turns ≈ 일반 fleet eta 상한 + buffer.
    """

    def __init__(self, window_turns: int = 30):
        self.window_turns = window_turns
        self._launches: List[LaunchMetadata] = []

    def reset_episode(self) -> None:
        """에피소드 종료 시 호출. 누적 launch state 비움."""
        self._launches.clear()

    def register_launches(self, launches: Iterable[LaunchMetadata]) -> None:
        """이번 step 발사된 launch 들을 보관."""
        self._launches.extend(launches)

    def update_step(
        self,
        prev_map: dict,
        curr_obs,
        *,
        player: int,
        turn: int,
    ) -> List[CaptureEvent]:
        """Window 만료 처리 + capture 검출 + attribution.

        prev_map: env.step() 직전 {pid: (owner, prod)} 스냅샷.
        curr_obs: env.step() 직후 raw_obs (dict 또는 attr 객체).
        player:   acting player (현재 진단/attribution 기준).
        turn:     ep_step.

        Returns: 이번 step 에 검출된 CaptureEvent 리스트 (0개 가능).
        """
        # 1. window 만료된 launch 제거.
        cutoff = turn - self.window_turns
        self._launches = [l for l in self._launches if l.turn >= cutoff]

        # 2. capture 검출 + 3. attribution.
        events: List[CaptureEvent] = []
        for p in _curr_planets(curr_obs):
            pid   = _planet_attr(p, 0, "id")
            owner = _planet_attr(p, 1, "owner")
            prod  = _planet_attr(p, 6, "production")
            prev_owner, _prev_prod = prev_map.get(pid, (-1, 0))
            if prev_owner == owner:
                continue  # 변화 없음

            linked = [l for l in self._launches if l.target_id == pid]
            events.append(CaptureEvent(
                turn=turn,
                planet_id=pid,
                prev_owner=prev_owner,
                new_owner=owner,
                target_prod=float(prod),
                linked_launches=list(linked),
            ))
        return events

    def __len__(self) -> int:
        """현재 window 안에 보관된 launch 수 (디버깅용)."""
        return len(self._launches)
