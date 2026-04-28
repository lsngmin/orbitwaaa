"""Reward events — launch / capture 시점 metadata data class.

규약 (reward/__init__.py docstring 의 stateful reward 분리 정책):
- 이 파일은 **데이터 컨테이너만**. 로직 없음.
- launch / capture 검출 / attribution 은 reward/trackers.py 의 LaunchCaptureTracker.
- reward component 는 ctx.capture_events 를 읽어 평가 — events 자체는 이전 step 정보를
  운반하지만 component 가 호출되는 시점엔 stateless (events 가 입력으로 들어옴).

메타데이터 출처:
    LaunchMetadata 는 train.py 의 decode_action_to_moves 가 step 단위로 생성.
    LaunchCaptureTracker 가 window 동안 보관 후 capture 발생 시 attribution.
    CaptureEvent.linked_launches 에 해당 capture 의 후보 launches 들이 들어감.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class LaunchMetadata:
    """발사 시점에 기록되는 launch 정보. attribution / reward 평가에 쓰일 후보.

    필드:
        turn                   : ep_step (에피소드 내 step 번호)
        source_id / target_id  : 행성 id (env 내부 식별자)
        target_owner_at_launch : -1=중립, 0/1=플레이어 — launch 시점 owner.
        target_kind            : "attack" (다른 owner) | "support" (같은 owner)
        ships_sent             : 실제 발사 ships
        eta_turns              : 예상 도착 turns
        req_over_src           : required / src.ships (∈ [0, 1+], cost 측면)
        req_over_prod          : required / src.production (경제 비용)
        nearest_neutral_rank   : src 기준 launchable 중립 ETA 정렬 시 target 의 순위.
                                 1=가장 가까움, 0=계산 안 됨 (target 비중립 또는 fallback).
        target_prod            : target.production (점령 후 가치 / loss penalty 분모)
    """
    turn: int
    source_id: int
    target_id: int
    target_owner_at_launch: int
    target_kind: str
    ships_sent: int
    eta_turns: int
    req_over_src: float
    req_over_prod: float
    nearest_neutral_rank: int
    target_prod: float


@dataclass
class CaptureEvent:
    """env.step() 이후 검출된 행성 소유권 transition.

    LaunchCaptureTracker.update_step 이 prev_map vs curr_obs 비교로 검출하고,
    window 안 launch 들 중 target_id 일치하는 것을 linked_launches 로 attribution.

    필드:
        turn               : capture 검출된 step (env.step() 직후)
        planet_id          : 변화한 행성 id
        prev_owner / new_owner : 전환 (예: -1 → 0 = 중립 점령, 0 → 1 = 빼앗김)
        target_prod        : capture 시점 행성 production (가치 평가용)
        linked_launches    : window 안에서 target_id 가 이 행성과 일치하는 launches.
                             0개일 수 있음 (자연 점령 / 만료된 launch 등 attribution 실패).
                             여러 개일 수 있음 (다중 source 협조). component 가 선택 정책 결정.
    """
    turn: int
    planet_id: int
    prev_owner: int
    new_owner: int
    target_prod: float
    linked_launches: List[LaunchMetadata] = field(default_factory=list)
