import math
import os
from collections import defaultdict

import yaml

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
)

# ships head: required_ships 배수 Categorical (commit 2)
_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(_cfg_path) as _f:
    _CFG = yaml.safe_load(_f)
SHIPS_MULTIPLIER_BINS = tuple(_CFG["model"].get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
NUM_SHIPS_BINS        = len(SHIPS_MULTIPLIER_BINS)


def _get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _fleet_speed(ships, max_speed):
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)


class HitRateTracker:
    """행동·launch 결과 메트릭 누적기 (V1 + V2).

    V1 (decode-time): attempts / filtered_* / launched 카운트
    V2 (post-step): fleet id 추적으로 out / sun_crash / target_hit / hit_other /
                    captured 를 exclusive/ambiguous 구분해 집계.

    사용 순서 (rollout 한 스텝):
      1. decode로 moves, counts, launches 얻음
      2. tracker.record(counts)         # V1
      3. tracker.register_launches(launches, prev_next_fleet_id)  # V2
      4. env.step(...) 실행
      5. tracker.resolve_step(prev_obs, curr_obs, max_speed)       # V2
      마지막: tracker.summary() → mean_* 딕셔너리
    """

    METRIC_KEYS = (
        "attempts", "filtered_invalid_target", "filtered_zero_ships",
        "filtered_sun", "filtered_path", "launched", "launched_high_prod",
        "noop_steps",
        "out", "sun_crash",
        "target_hit_exclusive", "target_hit_ambiguous",
        "hit_other_exclusive", "hit_other_ambiguous",
        "captured_exclusive", "captured_ambiguous",
        "captured_neutral", "captured_enemy",
        "early_home_expand",
        # ── 타겟 분포 / 초반 확장 계측 ───────────────────────────────────────
        "target_neutral",         # rollout 전체 중립 타겟 launched 수
        "target_enemy",           # rollout 전체 적 타겟 launched 수
        "early_neutral_attempts", # 초반 20턴 내 중립 타겟 launched 수
        "early_enemy_attempts",   # 초반 20턴 내 적 타겟 launched 수
        "early_neutral_captured", # 초반 20턴 내 (resolve 시점) 중립 점령 성공 수
        "early_launch_neutral_captured",  # 초반 20턴 내 발사된 중립 타겟의 실제 점령 수
                                          # (resolve 시점 무관 — fleet 비행시간 영향 X)
        # ── ships 분포 실측 (commit 2: Categorical multiplier head) ───────
        "chosen_multiplier_sum", "chosen_multiplier_sq_sum",
        "ships_to_send_sum", "required_ships_sum",
        "send_required_ratio_sum", "under_invested_count",
        "unknown_removal",
    ) + tuple(f"ships_bin_hist_{k}" for k in range(NUM_SHIPS_BINS))
    HOME_EXPAND_TURNS = 20
    HOME_EXPAND_RADIUS = 25.0

    def __init__(self, player_id=0):
        self.counters = defaultdict(int)
        self.n_steps  = 0
        self.player_id = player_id
        self.pending = {}  # fleet_id -> meta
        self.episodes = 0
        self.episode_turn = 0
        self.home_positions = []

    def reset_episode(self, obs):
        self.pending.clear()
        self.episodes += 1
        self.episode_turn = 0
        self.home_positions = []
        for p in _get(obs, "planets", []):
            owner = p[1] if isinstance(p, (list, tuple)) else p.owner
            if owner != self.player_id:
                continue
            x = p[2] if isinstance(p, (list, tuple)) else p.x
            y = p[3] if isinstance(p, (list, tuple)) else p.y
            self.home_positions.append((x, y))

    # ── V1 ──────────────────────────────────────────────────────────────────
    def record(self, counts):
        for k, v in counts.items():
            self.counters[k] += v
        # "noop"은 디코드 후 발사 0회가 아니라, 정책이 launch 시도 자체를 안 한 step으로 본다.
        if counts.get("attempts", 0) == 0:
            self.counters["noop_steps"] += 1
        self.n_steps += 1

    # ── V2: 등록 ────────────────────────────────────────────────────────────
    def register_launches(self, launches, next_fleet_id):
        """decode의 launches 메타를 fleet_id와 매핑해 pending에 넣는다.

        engine이 유효 move마다 next_fleet_id를 1씩 증가시키고, decode는 엔진이
        수용하는 조건(상위집합)만 통과시키므로 moves[i]는 fleet_id = next_fleet_id+i.

        동시에 target_owner 기반 launch 분포/초반 attempt 계측도 여기서 수행.
        """
        is_early = self.episode_turn < self.HOME_EXPAND_TURNS
        for i, meta in enumerate(launches):
            fid = next_fleet_id + i
            self.pending[fid] = {
                **meta,
                "last_x": meta["start_x"],
                "last_y": meta["start_y"],
                "launched_at_turn": self.episode_turn,
            }
            target_owner = meta.get("target_owner")
            if target_owner == -1:
                self.counters["target_neutral"] += 1
                if is_early:
                    self.counters["early_neutral_attempts"] += 1
            elif target_owner is not None and target_owner != self.player_id:
                self.counters["target_enemy"] += 1
                if is_early:
                    self.counters["early_enemy_attempts"] += 1

    # ── V2: 해소 ────────────────────────────────────────────────────────────
    def resolve_step(self, prev_obs, curr_obs, max_speed):
        alive_ids = {f[0] for f in _get(curr_obs, "fleets", [])}
        curr_fleet_map = {f[0]: f for f in _get(curr_obs, "fleets", [])}
        prev_fleets = list(_get(prev_obs, "fleets", []))
        prev_planets = list(_get(prev_obs, "planets", []))
        curr_planet_map = {p[0]: p for p in _get(curr_obs, "planets", [])}
        prev_planet_map = {p[0]: p for p in prev_planets}

        # 1. 내 pending fleet 중 소멸한 것 분류
        my_hits_by_planet = defaultdict(list)  # planet_id -> [fleet_id]
        removed_mine = []
        for fid, meta in list(self.pending.items()):
            if fid in alive_ids:
                f = curr_fleet_map[fid]
                meta["last_x"] = f[2]
                meta["last_y"] = f[3]
                continue
            old_pos = (meta["last_x"], meta["last_y"])
            speed = _fleet_speed(meta["ships"], max_speed)
            new_pos = (old_pos[0] + math.cos(meta["angle"]) * speed,
                       old_pos[1] + math.sin(meta["angle"]) * speed)
            cause, planet_id = _classify_removal(old_pos, new_pos, prev_planets, curr_planet_map)
            removed_mine.append((fid, meta, cause, planet_id))
            if cause == "hit" and planet_id is not None:
                my_hits_by_planet[planet_id].append(fid)
            del self.pending[fid]

        # 2. 적 fleet 충돌 attribution용: 이번 스텝에 사라진 적 fleet → planet
        enemy_hits_by_planet = defaultdict(int)
        for f in prev_fleets:
            if f[1] == self.player_id:
                continue
            if f[0] in alive_ids:
                continue
            ships = f[6] if len(f) > 6 else 1
            angle = f[4]
            old_pos = (f[2], f[3])
            speed = _fleet_speed(ships, max_speed)
            new_pos = (old_pos[0] + math.cos(angle) * speed,
                       old_pos[1] + math.sin(angle) * speed)
            cause, planet_id = _classify_removal(old_pos, new_pos, prev_planets, curr_planet_map)
            if cause == "hit" and planet_id is not None:
                enemy_hits_by_planet[planet_id] += 1

        # 3. exclusive/ambiguous 판정 + 카운트
        counted_capture_planets = set()
        for fid, meta, cause, planet_id in removed_mine:
            if cause == "out":
                self.counters["out"] += 1
            elif cause == "sun":
                self.counters["sun_crash"] += 1
            elif cause == "hit":
                exclusive = (len(my_hits_by_planet[planet_id]) == 1
                             and enemy_hits_by_planet[planet_id] == 0)
                suffix = "exclusive" if exclusive else "ambiguous"
                if planet_id == meta["target_id"]:
                    self.counters[f"target_hit_{suffix}"] += 1
                else:
                    self.counters[f"hit_other_{suffix}"] += 1
                prev_planet = prev_planet_map.get(planet_id)
                curr_planet = curr_planet_map.get(planet_id)
                if prev_planet and curr_planet:
                    prev_owner = prev_planet[1]
                    curr_owner = curr_planet[1]
                    if (
                        prev_owner != self.player_id
                        and curr_owner == self.player_id
                        and planet_id not in counted_capture_planets
                    ):
                        counted_capture_planets.add(planet_id)
                        self.counters[f"captured_{suffix}"] += 1
                        if prev_owner == -1:
                            self.counters["captured_neutral"] += 1
                            # 발사 시점 기준: "초반 20턴에 쏜 중립이 점령으로 이어졌나"
                            # resolve 시점이 아니라 launched_at_turn 기준이라 fleet 비행시간 영향 없음
                            if meta.get("launched_at_turn", 999) < self.HOME_EXPAND_TURNS:
                                self.counters["early_launch_neutral_captured"] += 1
                            # resolve 시점 기준 (기존)
                            if self.episode_turn < self.HOME_EXPAND_TURNS:
                                self.counters["early_neutral_captured"] += 1
                                px, py = prev_planet[2], prev_planet[3]
                                if any(
                                    math.hypot(px - hx, py - hy) <= self.HOME_EXPAND_RADIUS
                                    for hx, hy in self.home_positions
                                ):
                                    self.counters["early_home_expand"] += 1
                        else:
                            self.counters["captured_enemy"] += 1
            else:
                self.counters["unknown_removal"] += 1
        self.episode_turn += 1

    # ── 요약 ────────────────────────────────────────────────────────────────
    def summary(self):
        steps = max(self.n_steps, 1)
        out = {f"mean_{k}": self.counters.get(k, 0) / steps for k in self.METRIC_KEYS}
        attempts = self.counters.get("attempts", 0)
        launched = self.counters.get("launched", 0)
        out["launch_rate"] = launched / max(attempts, 1)
        out["noop_rate"] = self.counters.get("noop_steps", 0) / steps
        out["high_prod_target_rate"] = self.counters.get("launched_high_prod", 0) / max(launched, 1)
        out["neutral_capture_rate"] = self.counters.get("captured_neutral", 0) / max(launched, 1)
        out["enemy_capture_rate"] = self.counters.get("captured_enemy", 0) / max(launched, 1)
        out["early_home_expand_per_episode"] = self.counters.get("early_home_expand", 0) / max(self.episodes, 1)
        # ── 타겟 분포 / 초반 확장 파생 지표 ─────────────────────────────────
        eps = max(self.episodes, 1)
        out["target_neutral_rate"] = self.counters.get("target_neutral", 0) / max(launched, 1)
        out["target_enemy_rate"]   = self.counters.get("target_enemy", 0)   / max(launched, 1)
        out["early_neutral_attempts_per_episode"] = self.counters.get("early_neutral_attempts", 0) / eps
        out["early_enemy_attempts_per_episode"]   = self.counters.get("early_enemy_attempts", 0)   / eps
        out["early_neutral_captured_per_episode"] = self.counters.get("early_neutral_captured", 0) / eps
        out["early_launch_neutral_captured_per_episode"] = self.counters.get("early_launch_neutral_captured", 0) / eps
        # 발사대비 점령율: 초반에 쏜 중립 타겟 중 몇 %가 점령으로 이어졌는가
        # (fleet 비행시간 영향 없는 순수 실행 성공률)
        early_n_att = self.counters.get("early_neutral_attempts", 0)
        out["early_neutral_launch_to_cap_rate"] = (
            self.counters.get("early_launch_neutral_captured", 0) / max(early_n_att, 1)
        )
        # ── ships 분포 실측 파생 지표 (commit 2: Categorical multiplier) ──
        # 모든 통계는 "launched" 기준 평균 (filter 통과한 valid launch만 집계)
        cm_sum   = self.counters.get("chosen_multiplier_sum", 0.0)
        cm_sq    = self.counters.get("chosen_multiplier_sq_sum", 0.0)
        sts_sum  = self.counters.get("ships_to_send_sum", 0)
        req_sum  = self.counters.get("required_ships_sum", 0.0)
        srr_sum  = self.counters.get("send_required_ratio_sum", 0.0)
        under    = self.counters.get("under_invested_count", 0)
        if launched > 0:
            cm_mean = cm_sum / launched
            cm_var  = max(cm_sq / launched - cm_mean ** 2, 0.0)
            out["chosen_multiplier_mean"]   = cm_mean
            out["chosen_multiplier_std"]    = math.sqrt(cm_var)
            out["ships_to_send_mean"]       = sts_sum / launched
            out["required_ships_mean"]      = req_sum / launched
            out["send_required_ratio_mean"] = srr_sum / launched
            out["under_invested_rate"]      = under / launched
            # ships_bin 히스토그램 (launched 대비 비율) — commit 3 분포 로깅 선행
            for k in range(NUM_SHIPS_BINS):
                out[f"ships_bin_rate_{k}"] = self.counters.get(f"ships_bin_hist_{k}", 0) / launched
        else:
            for k in ("chosen_multiplier_mean", "chosen_multiplier_std", "ships_to_send_mean",
                      "required_ships_mean", "send_required_ratio_mean", "under_invested_rate"):
                out[k] = 0.0
            for k in range(NUM_SHIPS_BINS):
                out[f"ships_bin_rate_{k}"] = 0.0
        return out


def _classify_removal(old_pos, new_pos, prev_planets, curr_planet_map):
    """엔진 제거 우선순위 재현: out → sun → planet direct → sweep.

    Returns: (cause, planet_id_or_None)
      cause ∈ {"out", "sun", "hit", "unknown"}
    """
    # 1. Out of bounds
    if not (0 <= new_pos[0] <= BOARD_SIZE and 0 <= new_pos[1] <= BOARD_SIZE):
        return ("out", None)
    # 2. Sun collision
    if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
        return ("sun", None)
    # 3. Planet direct collision (prev_obs planet positions — fleet movement phase)
    for planet in prev_planets:
        planet_pos = (planet[2], planet[3])
        radius = planet[4]
        if point_to_segment_distance(planet_pos, old_pos, new_pos) < radius:
            return ("hit", planet[0])
    # 4. Planet sweep (planet moved prev→curr, might have caught fleet at new_pos)
    for planet in prev_planets:
        pid = planet[0]
        curr_planet = curr_planet_map.get(pid)
        if curr_planet is None:
            continue
        p_old = (planet[2], planet[3])
        p_new = (curr_planet[2], curr_planet[3])
        if p_old == p_new:
            continue
        radius = planet[4]
        if point_to_segment_distance(new_pos, p_old, p_new) < radius:
            return ("hit", pid)
    return ("unknown", None)
