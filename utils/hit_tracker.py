import math
from collections import defaultdict

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    point_to_segment_distance,
)


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
        "filtered_sun", "filtered_path", "launched",
        "out", "sun_crash",
        "target_hit_exclusive", "target_hit_ambiguous",
        "hit_other_exclusive", "hit_other_ambiguous",
        "captured_exclusive", "captured_ambiguous",
        "unknown_removal",
    )

    def __init__(self, player_id=0):
        self.counters = defaultdict(int)
        self.n_steps  = 0
        self.player_id = player_id
        self.pending = {}  # fleet_id -> meta

    # ── V1 ──────────────────────────────────────────────────────────────────
    def record(self, counts):
        for k, v in counts.items():
            self.counters[k] += v
        self.n_steps += 1

    # ── V2: 등록 ────────────────────────────────────────────────────────────
    def register_launches(self, launches, next_fleet_id):
        """decode의 launches 메타를 fleet_id와 매핑해 pending에 넣는다.

        engine이 유효 move마다 next_fleet_id를 1씩 증가시키고, decode는 엔진이
        수용하는 조건(상위집합)만 통과시키므로 moves[i]는 fleet_id = next_fleet_id+i.
        """
        for i, meta in enumerate(launches):
            fid = next_fleet_id + i
            self.pending[fid] = {
                **meta,
                "last_x": meta["start_x"],
                "last_y": meta["start_y"],
            }

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
                    if prev_owner != self.player_id and curr_owner == self.player_id:
                        self.counters[f"captured_{suffix}"] += 1
            else:
                self.counters["unknown_removal"] += 1

    # ── 요약 ────────────────────────────────────────────────────────────────
    def summary(self):
        steps = max(self.n_steps, 1)
        out = {f"mean_{k}": self.counters.get(k, 0) / steps for k in self.METRIC_KEYS}
        attempts = self.counters.get("attempts", 0)
        launched = self.counters.get("launched", 0)
        out["launch_rate"] = launched / max(attempts, 1)
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
