"""StepCache — Greedy 봇의 turn-local 데이터 인프라.

act(obs) 진입 시 build, 종료 시 폐기. 전역 변수 금지.
예외 = `committed_to_target` (intra-turn 선택 누적, mutable).

Scoring 함수는 read-only. 무거운 계산 (inbound, projection, local_pressure)
은 모두 build 단계에서 1회만.

planner (Stage 3) 의 rollout 마다 cache rebuild — 그래서 분리.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from prediction import (
    PositionCache,
    fleet_dst_and_eta,
    project_target_at_eta,
)

EPISODE_STEPS = 500.0
LOCAL_RADIUS  = 25.0    # local_pressure 집계 반경 (board 100×100 의 1/4)


class StepCache:
    """Turn-local cache. scoring 은 read-only (committed_to_target 만 mutate).

    build() 가 무거운 계산 (inbound, pressure) 1회 수행.
    project() 는 lazy — inbound 있는 dst 만 호출.
    Per-pair (eta, required, ships) 는 봇의 enumerate 단계에서 채움.
    """

    def __init__(self, planets: List[Planet], fleets: List[Fleet],
                 player: int, av: float, step: int):
        self.planets    = planets
        self.fleets     = fleets
        self.player     = player
        self.av         = av
        self.step       = step
        self.steps_left = max(1, int(EPISODE_STEPS) - int(step))

        self.pos_cache  = PositionCache(planets, av)
        self.my_planets = [p for p in planets if p.owner == player]
        self._planet_by_id = {p.id: p for p in planets}

        # build() 에서 채움
        self.inbound_enemy_ships: Dict[int, int]   = {}    # planet_id → ships 합
        self.inbound_ally_ships:  Dict[int, int]   = {}
        self.inbound_by_owner:    Dict[Tuple[int, int], int] = {}  # (pid, owner) → ships
        self.enemy_min_eta:       Dict[int, int]   = {}    # planet_id → 최소 적 ETA
        self.local_ally_prod:     Dict[int, float] = {}    # planet_id → 반경 내 합
        self.local_enemy_prod:    Dict[int, float] = {}

        # (dst.id, eta) → projection 결과. inbound 있는 dst 만 채움.
        self._proj_cache: Dict[Tuple[int, int], Tuple[int, float]] = {}

        # MUTABLE during scoring — accept 마다 += ships
        self.committed_to_target: Dict[int, int] = {}

    def build(self) -> None:
        """무거운 step-level 계산 1회."""
        self._build_inbound()
        self._build_local_pressure()

    def _build_inbound(self) -> None:
        for f in self.fleets:
            dpid, eta = fleet_dst_and_eta(f, self.planets, av=self.av,
                                            pos_cache=self.pos_cache)
            if dpid == -1:
                continue
            self.inbound_by_owner[(dpid, f.owner)] = (
                self.inbound_by_owner.get((dpid, f.owner), 0) + int(f.ships))
            if f.owner == self.player:
                self.inbound_ally_ships[dpid] = self.inbound_ally_ships.get(dpid, 0) + int(f.ships)
            else:
                self.inbound_enemy_ships[dpid] = self.inbound_enemy_ships.get(dpid, 0) + int(f.ships)
                if dpid not in self.enemy_min_eta or eta < self.enemy_min_eta[dpid]:
                    self.enemy_min_eta[dpid] = eta

    def _build_local_pressure(self) -> None:
        # O(P²) 1회. P~30 → ~900 ops, 무시 가능.
        for target in self.planets:
            ally_p = enemy_p = 0.0
            for other in self.planets:
                if other.id == target.id:
                    continue
                d = math.hypot(target.x - other.x, target.y - other.y)
                if d > LOCAL_RADIUS:
                    continue
                if other.owner == self.player:
                    ally_p += other.production
                elif other.owner != -1:
                    enemy_p += other.production
            self.local_ally_prod[target.id]  = ally_p
            self.local_enemy_prod[target.id] = enemy_p

    def project(self, dst: Planet, eta: int) -> Tuple[int, float]:
        """target_owner_at_eta, target_ships_at_eta — lazy 캐시.

        inbound 없는 dst 면 정적 공식 (production 만 누적). 있으면 호출.
        """
        if (dst.id not in self.inbound_enemy_ships and
            dst.id not in self.inbound_ally_ships):
            ships = dst.ships + (dst.production * eta if dst.owner != -1 else 0)
            return dst.owner, float(ships)
        key = (dst.id, eta)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        owner, ships = project_target_at_eta(
            dst, eta, self.planets, self.fleets,
            av=self.av, pos_cache=self.pos_cache,
        )
        self._proj_cache[key] = (owner, ships)
        return owner, ships
