"""Weighted-greedy 봇 — 모든 후보 (src, dst) 를 단일 점수식으로 평가, argmax.

설계 원칙 (rule-based 가 아님):
  - **하드 phase 컷오프 / 임계값 상수 없음.** 초반/중반/후반 분기 같은 if-else
    없음. 모든 "선호" 는 점수식의 가중치 항으로만 표현.
  - **mask 영역의 invariant 만 hard.** capture 수학적 불가 (ships_needed==0),
    수렴 실패, source 함선 부족 같은 건 후보에서 제외.
  - **중복 send 차단도 임계값 아님.** 한 step 안에서 채택된 dst 는 후속
    라운드에서 후보풀에서 빼는 것으로 처리 (×1.2 같은 magic number 없음).
  - **튜닝 대상은 WEIGHTS 벡터 하나.** grid / CMA-ES 등 black-box 로 self-play
    win_rate 최적화. 각 항의 의미는 docstring 으로만 — 코드는 가중합만.

진입점:
  - agent(obs): Kaggle / 학습 league opponent / BC trajectory 생성 동일 시그니처.
  - GreedyExpandSupportBot: stateless. WEIGHTS override 하려면 weights= 인자.

게임 물리는 prediction.py 의 helper 들 재사용 (aim / first_collision_on_path /
project_target_at_eta / resolve_ships_for_capture / resolve_ships_for_support /
compute_support_required). 직접 재구현 안 함.
"""
from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

from prediction import (
    PositionCache,
    compute_support_required,
    crosses_sun,
    fleet_dst_and_eta,
    resolve_ships_for_capture,
    resolve_ships_for_support,
)


# ── 가중치 벡터 (튜닝 대상) ──────────────────────────────────────────────
# 각 항의 의미는 _score 함수 docstring 참조. 코드는 가중합만 한다.
# black-box 최적화 (CMA-ES / grid search) 의 입력 차원 = len(WEIGHTS).

# ── 정규화 상수 ─────────────────────────────────────────────────────────
# 모든 feature 가 ~[0, 1] 스케일에 들어오도록 게임 룰 기반 분모.
# 가중치 의미를 비교 가능하게 하기 위함 — 튜닝 전 필수.
PROD_MAX      = 5.0      # production ∈ [1, 5] (GAME_RULES Planets)
EPISODE_STEPS = 500.0    # episodeSteps default
SHIPS_REF     = 20.0     # "보통" launch 함선 수 (스케일 단위, 튜닝 무관)
ETA_REF       = 120.0    # 정상 in-range eta 의 상한 (board 대각선 ~141)


# ── 가중치 벡터 (튜닝 대상) ──────────────────────────────────────────────
# 모든 feature 가 정규화돼 있어 weight 들이 같은 스케일에서 비교됨 →
# random search / CMA-ES search 공간이 의미를 가짐. 손튜닝 금지.

WEIGHTS = {
    "cap_eff":     1.0,   # production-per-ship efficiency (정규화)
    "eta":         1.0,   # ETA 길수록 불확실 → 음수 가중
    "src_drain":   1.0,   # src 함선 비율 소진 패널티
    "support":     1.0,   # 내 행성 보강 보너스
    "tgt_prod":    1.0,   # target 자체 production (즉시 가치)
    "neutral_aff": 0.0,   # 중립 vs 적 선호 (default 0)
}


# ── 진입점 ───────────────────────────────────────────────────────────────

def agent(obs) -> list:
    """Kaggle 진입점. obs dict → [[from_pid, angle, ships], ...]."""
    return GreedyExpandSupportBot().act(obs)


class GreedyExpandSupportBot:
    """Step-stateless weighted greedy. WEIGHTS 만 외부 주입 가능."""

    def __init__(self, weights: dict | None = None):
        self.w = dict(WEIGHTS)
        if weights:
            self.w.update(weights)

    # ── 점수 ────────────────────────────────────────────────────────────

    def _score(self, *, dst, ships_needed, src_ships, eta, total_steps_left,
               is_support: bool) -> float:
        """후보 (src→dst) 의 가중합 점수. 모든 feature 는 ~[0, 1] 정규화.

        cap_eff:     (prod/5) · (steps_left/500) / max(1, ships/20)
                     점령 효율 — 한 함선당 미래 누적 생산. 분모는 SHIPS_REF
                     기준으로 1 미만 안 되도록 클램핑 (소량 send 폭주 방지).
        eta:         eta / ETA_REF. 도착까지 턴 (음수 가중).
        src_drain:   ships / src_ships. src 비우는 행동 패널티 (이미 0~1).
        support:     0/1 indicator.
        tgt_prod:    prod / PROD_MAX. target 자체 production (즉시 가치).
        neutral_aff: dst 가 중립이면 +1, 적이면 -1, support 면 0.
                     (capture 후보 안에서만 의미 있는 부호 indicator.)
        """
        prod_norm  = dst.production / PROD_MAX
        steps_norm = max(1.0, total_steps_left) / EPISODE_STEPS
        ships_norm = max(1.0, ships_needed / SHIPS_REF)
        cap_eff    = (prod_norm * steps_norm) / ships_norm
        eta_norm   = eta / ETA_REF
        drain      = ships_needed / max(1, src_ships)
        neutral_sign = 0.0 if is_support else (1.0 if dst.owner == -1 else -1.0)

        return (
              self.w["cap_eff"]     * cap_eff
            - self.w["eta"]         * eta_norm
            - self.w["src_drain"]   * drain
            + self.w["support"]     * (1.0 if is_support else 0.0)
            + self.w["tgt_prod"]    * prod_norm
            + self.w["neutral_aff"] * neutral_sign
        )

    # ── act ─────────────────────────────────────────────────────────────

    def act(self, obs) -> list:
        # obs 파싱 — agents.md 의 namedtuple 변환 컨벤션.
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
        cache   = PositionCache(planets, av)
        steps_left = max(1, 500 - int(step))

        my_planets    = [p for p in planets if p.owner == player]
        if not my_planets:
            return []

        # Step 시작에 한 번만: in-flight fleet 들의 도착지 맵.
        # 용도는 (1) support 게이팅: 위협 없는 내 행성은 보강 후보 자체에서 제외,
        # (2) capture 의 net-arrival 보정: 해당 dst 로 향하는 같은-소유주 fleet 의
        # 합을 staticrequired 에서 깎아줘 (점령 ship 절약). dynamic project_target
        # 호출은 봇에선 비활성 — 1초 budget 보호. 약간의 정확도 손실 < 시간 초과.
        inbound_to_target = {}   # dst_pid → list[(fleet_owner, fleet_ships, eta)]
        for f in fleets:
            dpid, eta = fleet_dst_and_eta(f, planets, av=av, pos_cache=cache)
            if dpid != -1:
                inbound_to_target.setdefault(dpid, []).append((f.owner, f.ships, eta))

        # 후보 enumerate — (src, dst) 모든 쌍.
        # mask 영역 (sun cross / ships_needed==0 / required==None) 통과한 것만.
        candidates = []  # list of dict
        for src in my_planets:
            if src.ships < 2:   # 1 함선만 남기면 src 자체가 위험
                continue
            for dst in planets:
                if dst.id == src.id:
                    continue

                # sun-cross hard 차단 (mask invariant).
                if crosses_sun(src.x, src.y, dst.x, dst.y):
                    continue

                inbound = inbound_to_target.get(dst.id, ())

                if dst.owner == player:
                    # support 후보: 적 inbound 있을 때만 의미.
                    enemy_in = sum(s for o, s, _ in inbound if o != player)
                    if enemy_in <= 0:
                        continue
                    ally_in  = sum(s for o, s, _ in inbound if o == player)
                    # 도착 시 net = my_garrison + ally_in - enemy_in (eta-aware production
                    # 까지 정확히 안 잡지만, 휴리스틱 보강 의도에 충분).
                    net = dst.ships + ally_in - enemy_in
                    if net >= dst.ships:
                        continue   # 이미 충분.
                    proj_owner = player if net > 0 else (
                        max(((s, o) for o, s, _ in inbound if o != player),
                            default=(0, -1))[1]
                    )
                    proj_ships = abs(net) if net > 0 else (enemy_in - dst.ships)
                    required = compute_support_required(src, dst, proj_owner,
                                                         proj_ships, player)
                    if required is None:
                        continue
                    ships, angle, tx, ty, eta, _req = resolve_ships_for_support(
                        src, dst, av, bin_value=1.0,
                        src_ships=src.ships, required=required,
                        pos_cache=cache,
                    )
                    if ships <= 0:
                        continue
                    candidates.append({
                        "src": src, "dst": dst, "angle": angle,
                        "ships": ships, "eta": eta, "is_support": True,
                    })
                else:
                    # capture 후보 — static path (resolve 가 fleets=None 이면 내부적으로
                    # dst.ships + prod*eta + 1 공식 사용). 같은-적 inbound 의 합은
                    # 도착 시 garrison 에 더해질 거라 required 에 가산해서 보정.
                    same_owner_in = sum(s for o, s, _ in inbound if o == dst.owner)
                    ships, angle, tx, ty, eta, required, _ = resolve_ships_for_capture(
                        src, dst, av, bin_value=1.0,
                        src_ships=src.ships,
                        pos_cache=cache,
                        fleets=None, planets=None,
                    )
                    if ships <= 0 or required <= 0:
                        continue
                    if same_owner_in > 0:
                        # 보정: 적 같은편 보강 도착하면 더 많이 필요. src 부족하면 폐기.
                        adj_required = required + same_owner_in
                        if adj_required > src.ships - 1:
                            continue
                        ships = max(ships, adj_required)
                        required = adj_required
                    candidates.append({
                        "src": src, "dst": dst, "angle": angle,
                        "ships": ships, "eta": eta, "is_support": False,
                    })

        if not candidates:
            return []

        # 그리디 선택 루프 — 채택할 때마다 src 잔여 함선 차감, dst 는 후보풀 제거.
        # 임계값 (1.2 등) 없이 "이미 채택된 dst 는 한 번만" 으로 중복 차단.
        moves = []
        src_budget = {p.id: p.ships for p in my_planets}
        used_dst   = set()

        # 점수 계산 → (score, idx) 리스트 한 번 만들고 매 라운드 재정렬.
        # 후보 수 N 작아서 (≤ |my| × |planets|) 단순 정렬로 충분.
        while True:
            best = None
            best_score = -math.inf
            for c in candidates:
                if c["dst"].id in used_dst:
                    continue
                remaining = src_budget[c["src"].id]
                if c["ships"] > remaining - 1:    # src 에 최소 1 남김
                    continue
                s = self._score(
                    dst=c["dst"],
                    ships_needed=c["ships"],
                    src_ships=remaining,
                    eta=c["eta"],
                    total_steps_left=steps_left,
                    is_support=c["is_support"],
                )
                if s > best_score:
                    best_score = s
                    best       = c
            if best is None or best_score <= 0:
                break
            moves.append([best["src"].id, float(best["angle"]), int(best["ships"])])
            src_budget[best["src"].id] -= best["ships"]
            used_dst.add(best["dst"].id)

        return moves
