"""Weighted-greedy 봇 — planner 본체용 leaf evaluator.

설계 원칙:
  - validity_filter (게임 룰 위반만) ↔ score (전략) 엄격 분리.
  - 3-score 분리: neutral / enemy / support — 같은 feature 라도 의미가 다름.
  - 공통 9 weight + kind-specific 7 weight = 16 weight.
  - StepCache: act(obs) 진입 시 build, 종료 시 폐기. 전역 변수 금지.
    예외 = `committed_to_target` (intra-turn 선택 누적, mutable).
  - 후보 scoring 은 캐시만 읽음. 무거운 계산 (eta/required/projection/pressure)
    은 모두 build 단계에서 1회.

진입점:
  - agent(obs): Kaggle / 학습 league opponent / BC trajectory 동일.
  - GreedyExpandSupportBot: weights= 인자로 dict override 가능.

게임 물리는 prediction.py 의 helper 들 재사용. 직접 재구현 금지.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from prediction import (
    PositionCache,
    aim,
    compute_support_required,
    crosses_sun,
    fleet_dst_and_eta,
    project_target_at_eta,
    resolve_ships_for_capture,
    resolve_ships_for_support,
)


# ── 정규화 상수 ─────────────────────────────────────────────────────────
# 모든 feature 가 ~[0, 1] 스케일에 들어오도록 게임 룰 기반 분모.
PROD_MAX        = 5.0      # production ∈ [1, 5] (GAME_RULES Planets)
EPISODE_STEPS   = 500.0    # episodeSteps default
ETA_REF         = 120.0    # 정상 in-range eta 상한 (board 대각선 ~141)
LOCAL_RADIUS    = 25.0     # local_pressure 집계 반경 (board 100×100 의 1/4)
PROD_LIFE_REF   = PROD_MAX * EPISODE_STEPS   # = 2500. prod_save 정규화 분모.


# ── 가중치 (튜닝 대상 16개) ──────────────────────────────────────────────
# 모든 feature 가 정규화돼 weight 들이 같은 스케일. random search / CMA-ES.
# 손튜닝 금지. 공통 9 + kind 7 = 16.

WEIGHTS = {
    # 공통 9 — 모든 score 함수에서 사용
    "common": {
        "cost":          1.0,   # ships_needed / src.ships          (penalty)
        "remaining":     1.0,   # 1 - cost = src 잔존 비율          (bonus)
        "eta":           1.0,   # eta_norm                          (penalty)
        "prod":          1.0,   # dst.production / 5                (bonus)
        "efficiency":    1.0,   # cap_efficiency                    (bonus)
        "committed":     1.0,   # committed_frac                    (penalty)
        "overcommit":    1.0,   # overcommit_frac                   (penalty)
        "race":          1.0,   # eta_advantage (정규화)             (bonus)
        "source_threat": 1.0,   # inbound_enemy_to_src / src.ships  (penalty)
    },
    "neutral": {
        "bias":          0.0,
        "nearest":       1.0,   # 0 = 가장 가까운, 1 = 가장 먼
    },
    "enemy": {
        "bias":          0.0,
        "weakness":      1.0,   # 1 / (1 + ships_at_eta / 50)
    },
    "support": {
        "bias":          0.0,
        "threat":        1.0,   # enemy_inbound / max(1, dst.ships)
        "prod_save":     1.0,   # dst.production · steps_left / 2500
    },
}


# ── StepCache ───────────────────────────────────────────────────────────
# act(obs) 진입 시 build, 종료 시 폐기. 전역 금지.

class StepCache:
    """Turn-local cache. scoring 은 read-only (committed_to_target 만 mutate).

    build() 가 무거운 계산 (inbound, pressure, projection) 1회 수행.
    Per-pair (eta, required, ships) 는 enumerate 단계에서 lazy 채워짐.
    """

    def __init__(self, planets: List[Planet], fleets: List[Fleet],
                 player: int, av: float, step: int):
        self.planets    = planets
        self.fleets     = fleets
        self.player     = player
        self.av         = av
        self.step       = step
        self.steps_left = max(1, int(EPISODE_STEPS) - int(step))

        self.pos_cache = PositionCache(planets, av)
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
        # O(P²) 1회. P~30 이라 ~900 ops, 무시 가능.
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
            # static: owner 변화 없음. neutral 은 production 0.
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


# ── Candidate ───────────────────────────────────────────────────────────
# 후보 1개의 모든 사전 계산값. score 함수가 이걸 받아 가중합만.

class Candidate:
    __slots__ = (
        "src", "dst", "kind", "ships", "angle", "eta", "required",
        # 정규화된 feature 값 — score 가 직접 읽음
        "cost", "remaining", "eta_norm", "prod_norm", "cap_eff",
        "race_advantage", "source_threat_frac",
        "nearest_neutral_rank_norm", "enemy_weakness",
        "support_threat_frac", "prod_save_norm",
    )

    def __init__(self, src, dst, kind, ships, angle, eta, required):
        self.src = src
        self.dst = dst
        self.kind = kind
        self.ships = ships
        self.angle = angle
        self.eta = eta
        self.required = required
        # feature 들은 _build_features 에서 채움
        self.cost = self.remaining = self.eta_norm = 0.0
        self.prod_norm = self.cap_eff = 0.0
        self.race_advantage = self.source_threat_frac = 0.0
        self.nearest_neutral_rank_norm = self.enemy_weakness = 0.0
        self.support_threat_frac = self.prod_save_norm = 0.0


# ── validity_filter ─────────────────────────────────────────────────────
# 게임 룰/물리 위반만. 전략 판단 (멀다/비싸다/위험하다) 절대 금지.

def validity_pre(src: Planet, dst: Planet) -> bool:
    """eta/ships 계산 전 cheap 체크."""
    if src.id == dst.id:
        return False
    if src.ships < 1:           # 발사할 함선 자체 없음 (rule: num_ships > 0)
        return False
    if crosses_sun(src.x, src.y, dst.x, dst.y):
        return False            # 함대 소멸 (rule: 태양 교차)
    return True


def validity_post(ships: int, required: int) -> bool:
    """eta/ships 계산 후 체크."""
    if ships <= 0:              # capacity short / 도착 시 이미 내 거
        return False
    if required <= 0:
        return False
    return True


# ── 봇 본체 ──────────────────────────────────────────────────────────────

def agent(obs) -> list:
    """Kaggle 진입점."""
    return GreedyExpandSupportBot().act(obs)


class GreedyExpandSupportBot:
    """Weighted greedy. weights dict 는 nested (common/neutral/enemy/support)."""

    def __init__(self, weights: dict | None = None):
        self.w = {k: dict(v) for k, v in WEIGHTS.items()}
        if weights:
            for kind, sub in weights.items():
                self.w[kind].update(sub)

    # ── score ───────────────────────────────────────────────────────────

    def _score(self, c: Candidate, cache: StepCache) -> float:
        """공통 9 + kind-specific 가중합. committed/overcommit 은 매 호출 재계산."""
        committed = cache.committed_to_target.get(c.dst.id, 0)
        committed_frac   = committed / max(1, c.required)
        overcommit_frac  = max(0, committed + c.ships - c.required) / max(1, c.required)

        wc = self.w["common"]
        s = (
            - wc["cost"]          * c.cost
            + wc["remaining"]     * c.remaining
            - wc["eta"]           * c.eta_norm
            + wc["prod"]          * c.prod_norm
            + wc["efficiency"]    * c.cap_eff
            - wc["committed"]     * committed_frac
            - wc["overcommit"]    * overcommit_frac
            + wc["race"]          * c.race_advantage
            - wc["source_threat"] * c.source_threat_frac
        )
        if c.kind == "neutral":
            wn = self.w["neutral"]
            s += wn["bias"] + wn["nearest"] * c.nearest_neutral_rank_norm
        elif c.kind == "enemy":
            we = self.w["enemy"]
            s += we["bias"] + we["weakness"] * c.enemy_weakness
        else:  # support
            ws = self.w["support"]
            s += (ws["bias"]
                  + ws["threat"]    * c.support_threat_frac
                  + ws["prod_save"] * c.prod_save_norm)
        return s

    # ── enumerate ────────────────────────────────────────────────────────

    def _build_candidate(self, src: Planet, dst: Planet, cache: StepCache):
        """validity_pre 통과 후 호출. eta/ships/required 계산해 Candidate 반환.

        validity_post 실패 시 None.
        """
        player = cache.player
        if dst.owner == player:
            # support 후보는 적 inbound 가 있을 때만 의미.
            enemy_in = cache.inbound_enemy_ships.get(dst.id, 0)
            if enemy_in <= 0:
                return None
            ally_in = cache.inbound_ally_ships.get(dst.id, 0)
            net = dst.ships + ally_in - enemy_in
            if net >= dst.ships:
                return None
            proj_owner = player if net > 0 else (
                cache.fleets[0].owner if cache.fleets else -1
            )
            proj_ships = abs(net) if net > 0 else (enemy_in - dst.ships)
            required = compute_support_required(src, dst, proj_owner, proj_ships, player)
            if required is None:
                return None
            ships, angle, _, _, eta, req = resolve_ships_for_support(
                src, dst, cache.av, bin_value=1.0,
                src_ships=src.ships, required=required,
                pos_cache=cache.pos_cache,
            )
            kind = "support"
        else:
            # capture (neutral / enemy) — static 공식 (fleets/planets=None).
            ships, angle, _, _, eta, required, _ = resolve_ships_for_capture(
                src, dst, cache.av, bin_value=1.0,
                src_ships=src.ships,
                pos_cache=cache.pos_cache, fleets=None, planets=None,
            )
            # 같은-적 보강 inbound 보정 — 캐시에서 O(1) 조회.
            if dst.owner != -1:
                same_owner_in = cache.inbound_by_owner.get((dst.id, dst.owner), 0)
                if same_owner_in > 0:
                    required += same_owner_in
                    ships    = max(ships, required)
            kind = "neutral" if dst.owner == -1 else "enemy"

        if not validity_post(ships, required):
            return None
        if ships > src.ships:    # 마지막 안전장치 (validity_pre 가 src.ships>=1만 봄)
            return None
        return Candidate(src, dst, kind, int(ships), float(angle),
                         int(eta), int(required))

    def _build_features(self, c: Candidate, cache: StepCache,
                         neutral_rank_lookup: Dict[Tuple[int, int], float]) -> None:
        """Candidate 의 모든 feature 채움. score 가 읽기만 하도록."""
        c.cost      = c.ships / max(1, c.src.ships)
        c.remaining = (c.src.ships - c.ships) / max(1, c.src.ships)
        c.eta_norm  = c.eta / ETA_REF
        c.prod_norm = c.dst.production / PROD_MAX
        c.cap_eff   = (c.prod_norm * (cache.steps_left / EPISODE_STEPS)) / max(0.05, c.cost)

        enemy_eta = cache.enemy_min_eta.get(c.dst.id)
        if enemy_eta is None:
            c.race_advantage = 1.0   # 경쟁 없음 → 항상 우리 승
        else:
            c.race_advantage = max(-1.0, min(1.0, (enemy_eta - c.eta) / ETA_REF))

        src_threat = cache.inbound_enemy_ships.get(c.src.id, 0)
        c.source_threat_frac = min(1.0, src_threat / max(1, c.src.ships))

        if c.kind == "neutral":
            c.nearest_neutral_rank_norm = neutral_rank_lookup.get((c.src.id, c.dst.id), 0.0)
        elif c.kind == "enemy":
            _, ships_at_eta = cache.project(c.dst, c.eta)
            c.enemy_weakness = 1.0 / (1.0 + ships_at_eta / 50.0)
        else:  # support
            c.support_threat_frac = min(2.0,
                cache.inbound_enemy_ships.get(c.dst.id, 0) / max(1, c.dst.ships))
            c.prod_save_norm = (c.dst.production * cache.steps_left) / PROD_LIFE_REF

    # ── act ─────────────────────────────────────────────────────────────

    def act(self, obs) -> list:
        # obs 파싱
        if isinstance(obs, dict):
            raw_planets = obs.get("planets", [])
            raw_fleets  = obs.get("fleets", [])
            player      = obs.get("player", 0)
            av          = obs.get("angular_velocity", 0.0)
            step        = obs.get("step", 0)
        else:
            raw_planets = obs.planets
            raw_fleets  = obs.fleets
            player      = obs.player
            av          = getattr(obs, "angular_velocity", 0.0)
            step        = getattr(obs, "step", 0)

        planets = [Planet(*p) for p in raw_planets]
        fleets  = [Fleet(*f)  for f in raw_fleets]

        cache = StepCache(planets, fleets, player, av, int(step))
        cache.build()
        if not cache.my_planets:
            return []

        # 후보 enumerate — validity_pre → build → validity_post.
        raw_candidates = []
        for src in cache.my_planets:
            for dst in planets:
                if not validity_pre(src, dst):
                    continue
                c = self._build_candidate(src, dst, cache)
                if c is None:
                    continue
                raw_candidates.append(c)

        if not raw_candidates:
            return []

        # nearest_neutral_rank: src 별 neutral 후보 거리 정렬.
        neutral_rank_lookup: Dict[Tuple[int, int], float] = {}
        from collections import defaultdict
        by_src_neutral = defaultdict(list)
        for c in raw_candidates:
            if c.kind == "neutral":
                d = math.hypot(c.dst.x - c.src.x, c.dst.y - c.src.y)
                by_src_neutral[c.src.id].append((d, c.dst.id))
        for src_id, lst in by_src_neutral.items():
            lst.sort()
            n = max(1, len(lst) - 1)
            for rank, (_, dst_id) in enumerate(lst):
                neutral_rank_lookup[(src_id, dst_id)] = rank / n

        # feature 채움
        for c in raw_candidates:
            self._build_features(c, cache, neutral_rank_lookup)

        # greedy 채택 — 매 라운드 score 재계산 (committed 변화 반영).
        moves = []
        src_budget = {p.id: p.ships for p in cache.my_planets}
        used_dst = set()

        while True:
            best = None
            best_score = -math.inf
            for c in raw_candidates:
                if c.dst.id in used_dst:
                    continue
                remaining = src_budget[c.src.id]
                if c.ships > remaining:    # 자원 부족 (validity 가 아닌 누적 결과)
                    continue
                # cost/remaining 은 src_budget 변화에 따라 다시 계산 — 정확성 위해
                c.cost      = c.ships / max(1, remaining)
                c.remaining = (remaining - c.ships) / max(1, remaining)
                s = self._score(c, cache)
                if s > best_score:
                    best_score = s
                    best       = c
            if best is None or best_score <= 0:
                break
            moves.append([best.src.id, float(best.angle), int(best.ships)])
            src_budget[best.src.id] -= best.ships
            cache.committed_to_target[best.dst.id] = (
                cache.committed_to_target.get(best.dst.id, 0) + best.ships
            )
            used_dst.add(best.dst.id)

        return moves
