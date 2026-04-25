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
        # ── target-type 분리: neutral(prod 없음) vs enemy(prod 회복) ────────
        "ships_to_send_sum_neutral", "ships_to_send_sum_enemy",
        "required_ships_sum_neutral", "required_ships_sum_enemy",
        "send_required_ratio_sum_neutral", "send_required_ratio_sum_enemy",
        "under_invested_count_neutral", "under_invested_count_enemy",
        # ── 연계 공격 계측 (단발 실패 vs 계획된 연속 압박 구분) ──────────────
        # repeat_target: 같은 target에 REPEAT_K턴 내 2회 이상 발사 (현재 발사가 N≥2번째)
        # launch_to_cap_k_{neu,enm}: 발사 후 LAUNCH_TO_CAP_K턴 내 그 target이 결국 점령된 launch 수
        "repeat_target",
        "launch_to_cap_k_neutral", "launch_to_cap_k_enemy",
        # ── 1차 진단 metric 묶음 (방향: 단발 점령 + 유지 + 자원 보존 측정) ───
        # decode-time:
        #   all_in_launches              : ships_needed >= ALL_IN_THRESHOLD * src.ships
        #   remaining_ships_after_launch_sum : 발사 후 source에 남은 ships (per-launch)
        #   distinct_targets_sum         : per-step distinct target_id 수 (같은 턴 분산도)
        # resolve-time:
        #   captured_single_shot         : capture가 일어난 step에 우리 fleet 1개만 충돌
        #   capture_hold_k_total/success : 점령 후 K턴 보유 추적 (denom/num)
        #   post_reloss_k_total/count    : 점령 후 K턴 안에 한 번이라도 재상실됐나
        "all_in_launches",
        "remaining_ships_after_launch_sum",
        "distinct_targets_sum",
        "captured_single_shot",
        "capture_hold_k_total", "capture_hold_k_success",
        "post_reloss_k_total", "post_reloss_k_count",
        "unknown_removal",
    ) + tuple(f"ships_bin_hist_{k}" for k in range(NUM_SHIPS_BINS))
    HOME_EXPAND_TURNS = 20
    HOME_EXPAND_RADIUS = 25.0
    # 연계 공격 윈도우 (평균 flight time이 대략 10~15턴 이므로 20턴이면 fleet 도착 커버)
    REPEAT_K = 20
    LAUNCH_TO_CAP_K = 20
    # 단발 점령 + 자원 보존 진단 윈도우
    ALL_IN_THRESHOLD = 0.8       # ships_needed / src.ships 가 이 이상이면 all-in
    CAPTURE_HOLD_K = 5           # 점령 후 K턴 후 보유/재상실 판정

    def __init__(self, player_id=0):
        self.counters = defaultdict(int)
        self.n_steps  = 0
        self.player_id = player_id
        self.pending = {}  # fleet_id -> meta
        self.episodes = 0
        self.episode_turn = 0
        self.home_positions = []
        # 연계 공격 추적:
        # last_launch_turn[target_id] = 마지막 발사 turn  (repeat_target 검출)
        # launches_by_target[target_id] = [(launch_turn, target_owner_at_launch), ...]
        #                                  (launch_to_cap_k 검출 — 윈도우 만료 시 제거)
        self.last_launch_turn = {}
        self.launches_by_target = defaultdict(list)
        # 점령 후 유지/재상실 추적: planet_id -> {"captured_turn": int, "ever_lost": bool}
        # CAPTURE_HOLD_K턴 경과 시 hold/reloss 카운트로 flush.
        self.capture_pending = {}

    def reset_episode(self, obs):
        self.pending.clear()
        self.episodes += 1
        self.episode_turn = 0
        self.home_positions = []
        self.last_launch_turn = {}
        self.launches_by_target = defaultdict(list)
        self.capture_pending = {}
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
        # distinct_targets_per_turn: 같은 턴에 몇 개 다른 target에 분산 발사하는지.
        # 1등 패턴(여러 source → 1 target 동시발사)이면 작고, 난사(여러 source → 여러 target)면 큼.
        if launches:
            self.counters["distinct_targets_sum"] += len({m["target_id"] for m in launches})
        for i, meta in enumerate(launches):
            fid = next_fleet_id + i
            self.pending[fid] = {
                **meta,
                "last_x": meta["start_x"],
                "last_y": meta["start_y"],
                "launched_at_turn": self.episode_turn,
            }
            target_owner = meta.get("target_owner")
            target_id    = meta.get("target_id")
            if target_owner == -1:
                self.counters["target_neutral"] += 1
                if is_early:
                    self.counters["early_neutral_attempts"] += 1
            elif target_owner is not None and target_owner != self.player_id:
                self.counters["target_enemy"] += 1
                if is_early:
                    self.counters["early_enemy_attempts"] += 1

            # 연계 공격 계측: 같은 target에 REPEAT_K턴 내 재발사인가?
            # 현재 발사가 N≥2번째일 때 repeat_target += 1 (첫 발사는 아직 "연계" 아님).
            if target_id is not None:
                last = self.last_launch_turn.get(target_id)
                if last is not None and (self.episode_turn - last) <= self.REPEAT_K:
                    self.counters["repeat_target"] += 1
                self.last_launch_turn[target_id] = self.episode_turn
                # launch_to_cap_k 추적용 (target 점령 시점에 역조회)
                self.launches_by_target[target_id].append((self.episode_turn, target_owner))

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
                        # single-shot: 우리 fleet 1개만이 이 planet에 충돌해 점령 성사
                        # (ambiguous 분류와 별개 — 적 fleet 동시 충돌은 무시. 우리 측 분산 여부만).
                        if len(my_hits_by_planet[planet_id]) == 1:
                            self.counters["captured_single_shot"] += 1
                        # hold/reloss 추적 시작
                        self.capture_pending[planet_id] = {
                            "captured_turn": self.episode_turn,
                            "ever_lost": False,
                        }
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

        # launch_to_cap_k: 이번 턴에 "내 행성이 된" planet을 찾아,
        # LAUNCH_TO_CAP_K턴 내 그 planet을 target으로 쏜 launch들을 successful로 카운트.
        # 기존 captured_{suffix}는 fleet 충돌 attribution에 의존 (한 캡처당 1회만)
        # 여기서는 연계 launch 전부 카운트 → "지금은 부족해 보여도 결국 먹었나?" 분석용.
        for pid, curr_planet in curr_planet_map.items():
            prev_planet = prev_planet_map.get(pid)
            if prev_planet is None:
                continue
            if prev_planet[1] == self.player_id or curr_planet[1] != self.player_id:
                continue
            launches = self.launches_by_target.pop(pid, [])
            for l_turn, l_owner in launches:
                if self.episode_turn - l_turn > self.LAUNCH_TO_CAP_K:
                    continue
                kind = "neutral" if l_owner == -1 else "enemy"
                self.counters[f"launch_to_cap_k_{kind}"] += 1

        # 윈도우 만료 launch 정리 (메모리는 작지만 일관성 위해)
        for pid in list(self.launches_by_target.keys()):
            self.launches_by_target[pid] = [
                (t, o) for (t, o) in self.launches_by_target[pid]
                if self.episode_turn - t <= self.LAUNCH_TO_CAP_K
            ]
            if not self.launches_by_target[pid]:
                del self.launches_by_target[pid]

        # ── capture_hold_k / post_reloss_k flush ──────────────────────────
        # 1) 모든 pending 점령에 대해 "이번 step에 잃었나" 갱신
        # 2) captured_turn에서 K턴 경과한 항목은 hold/reloss 결과 카운트로 flush
        # K턴 미만에 episode 종료 시 미flush — 짧은 게임 편향 방지.
        for pid, info in self.capture_pending.items():
            curr = curr_planet_map.get(pid)
            if curr is not None and curr[1] != self.player_id:
                info["ever_lost"] = True
        for pid in list(self.capture_pending.keys()):
            info = self.capture_pending[pid]
            age  = self.episode_turn - info["captured_turn"]
            if age >= self.CAPTURE_HOLD_K:
                curr = curr_planet_map.get(pid)
                still_owned = (curr is not None and curr[1] == self.player_id)
                self.counters["capture_hold_k_total"] += 1
                if still_owned:
                    self.counters["capture_hold_k_success"] += 1
                self.counters["post_reloss_k_total"] += 1
                if info["ever_lost"]:
                    self.counters["post_reloss_k_count"] += 1
                del self.capture_pending[pid]

        self.episode_turn += 1

    # ── 요약 ────────────────────────────────────────────────────────────────
    @staticmethod
    def summary_from_counters(counters, n_steps, episodes):
        """Raw counters dict + n_steps + episodes에서 normalized summary 구성.

        병렬 worker에서 받은 raw state를 합산한 뒤 정확한 분모(step/episode/
        launched/attempts)로 한 번에 정규화할 수 있게 separate function으로 분리.
        단순 worker-평균은 episode 길이가 worker마다 다를 때 편향됨.
        """
        steps = max(n_steps, 1)
        out = {f"mean_{k}": counters.get(k, 0) / steps for k in HitRateTracker.METRIC_KEYS}
        attempts = counters.get("attempts", 0)
        launched = counters.get("launched", 0)
        out["launch_rate"] = launched / max(attempts, 1)
        out["noop_rate"] = counters.get("noop_steps", 0) / steps
        out["high_prod_target_rate"] = counters.get("launched_high_prod", 0) / max(launched, 1)
        out["neutral_capture_rate"] = counters.get("captured_neutral", 0) / max(launched, 1)
        out["enemy_capture_rate"] = counters.get("captured_enemy", 0) / max(launched, 1)
        eps = max(episodes, 1)
        out["early_home_expand_per_episode"] = counters.get("early_home_expand", 0) / eps
        out["target_neutral_rate"] = counters.get("target_neutral", 0) / max(launched, 1)
        out["target_enemy_rate"]   = counters.get("target_enemy", 0)   / max(launched, 1)
        out["early_neutral_attempts_per_episode"] = counters.get("early_neutral_attempts", 0) / eps
        out["early_enemy_attempts_per_episode"]   = counters.get("early_enemy_attempts", 0)   / eps
        out["early_neutral_captured_per_episode"] = counters.get("early_neutral_captured", 0) / eps
        out["early_launch_neutral_captured_per_episode"] = counters.get("early_launch_neutral_captured", 0) / eps
        # 발사대비 점령율: 초반에 쏜 중립 타겟 중 몇 %가 점령으로 이어졌는가
        early_n_att = counters.get("early_neutral_attempts", 0)
        out["early_neutral_launch_to_cap_rate"] = (
            counters.get("early_launch_neutral_captured", 0) / max(early_n_att, 1)
        )
        # ── ships 분포 실측 파생 지표 (Categorical multiplier) ──────────────
        # pooled variance: var = E[X²] - E[X]² 는 counter 합산에도 유효.
        cm_sum   = counters.get("chosen_multiplier_sum", 0.0)
        cm_sq    = counters.get("chosen_multiplier_sq_sum", 0.0)
        sts_sum  = counters.get("ships_to_send_sum", 0)
        req_sum  = counters.get("required_ships_sum", 0.0)
        srr_sum  = counters.get("send_required_ratio_sum", 0.0)
        under    = counters.get("under_invested_count", 0)
        if launched > 0:
            cm_mean = cm_sum / launched
            cm_var  = max(cm_sq / launched - cm_mean ** 2, 0.0)
            out["chosen_multiplier_mean"]   = cm_mean
            out["chosen_multiplier_std"]    = math.sqrt(cm_var)
            out["ships_to_send_mean"]       = sts_sum / launched
            out["required_ships_mean"]      = req_sum / launched
            out["send_required_ratio_mean"] = srr_sum / launched
            out["under_invested_rate"]      = under / launched
            for k in range(NUM_SHIPS_BINS):
                out[f"ships_bin_rate_{k}"] = counters.get(f"ships_bin_hist_{k}", 0) / launched
        else:
            for k in ("chosen_multiplier_mean", "chosen_multiplier_std", "ships_to_send_mean",
                      "required_ships_mean", "send_required_ratio_mean", "under_invested_rate"):
                out[k] = 0.0
            for k in range(NUM_SHIPS_BINS):
                out[f"ships_bin_rate_{k}"] = 0.0

        # ── target-type 분리 지표 ───────────────────────────────────────────
        # 분모: target_neutral / target_enemy (register_launches에서 launch 시점에 집계).
        # 의미:
        #   send_required_ratio_mean_{neutral,enemy}:
        #     ships_needed / required 비율. required는 target.ships+prod×turns+1 기준.
        #     1.10 bin 정상 수렴 시 ≈1.1 근처. clip(src.ships 한계)이 자주 발생하면 < 1.
        #   under_invested_rate_{neutral,enemy}:
        #     nominal margin 미달 비율 (= src.ships clip). under-invest가 enemy 쪽에
        #     많으면 prod 재생산으로 장기 waste, 패배 상관 높음.
        #   launch_to_cap_rate_{neutral,enemy}:
        #     launch 후 LAUNCH_TO_CAP_K턴 내 target이 결국 점령됐나.
        #     under-invest가 많아도 이 값이 높으면 "연계 공격 성공" 패턴,
        #     낮으면 단순 waste 패턴.
        for kind in ("neutral", "enemy"):
            n_launch = counters.get(f"target_{kind}", 0)
            if n_launch > 0:
                out[f"ships_to_send_mean_{kind}"]       = counters.get(f"ships_to_send_sum_{kind}", 0) / n_launch
                out[f"required_ships_mean_{kind}"]      = counters.get(f"required_ships_sum_{kind}", 0.0) / n_launch
                out[f"send_required_ratio_mean_{kind}"] = counters.get(f"send_required_ratio_sum_{kind}", 0.0) / n_launch
                out[f"under_invested_rate_{kind}"]      = counters.get(f"under_invested_count_{kind}", 0) / n_launch
                out[f"launch_to_cap_rate_{kind}"]       = counters.get(f"launch_to_cap_k_{kind}", 0) / n_launch
            else:
                out[f"ships_to_send_mean_{kind}"]       = 0.0
                out[f"required_ships_mean_{kind}"]      = 0.0
                out[f"send_required_ratio_mean_{kind}"] = 0.0
                out[f"under_invested_rate_{kind}"]      = 0.0
                out[f"launch_to_cap_rate_{kind}"]       = 0.0

        # repeat_target_rate: 같은 target에 REPEAT_K턴 내 재발사 비율.
        # (첫 발사는 repeat 아니므로 최대값 ≈ (launched-unique_targets)/launched)
        # 높으면 계획된 연속 압박, 낮으면 산발적 단발 시도.
        out["repeat_target_rate"] = counters.get("repeat_target", 0) / max(launched, 1)

        # ── 1차 진단 metric 묶음 (단발 점령 + 유지 + 자원 보존) ─────────────
        # all_in_launch_rate: 발사 중 ALL_IN_THRESHOLD 이상 비율
        #                    (자원 보존 못 하고 비우는 패턴 직접 지표)
        # remaining_ships_after_launch_mean: 평균 발사 후 잔여 ships (방어 reserve)
        # distinct_targets_per_turn: 한 턴 평균 distinct 타겟 수
        #                          1등 패턴(분산 X, 집중 O)이면 작음, 난사면 큼
        out["all_in_launch_rate"] = counters.get("all_in_launches", 0) / max(launched, 1)
        out["remaining_ships_after_launch_mean"] = (
            counters.get("remaining_ships_after_launch_sum", 0) / max(launched, 1)
        )
        out["distinct_targets_per_turn"] = (
            counters.get("distinct_targets_sum", 0) / steps
        )

        # single_shot_capture_rate: 점령 중 우리 fleet 단발로 성사된 비율
        # (멀티-source 동시점령은 0으로 카운트 — 1등은 양쪽 다 쓰지만 단발이 dominant)
        total_caps = (counters.get("captured_exclusive", 0)
                      + counters.get("captured_ambiguous", 0))
        out["single_shot_capture_rate"] = (
            counters.get("captured_single_shot", 0) / max(total_caps, 1)
        )

        # capture_hold_k_rate: 점령 후 K턴 시점 보유 성공률 (분모: 만료된 점령건만)
        # post_capture_reloss_rate_k: 점령 후 K턴 안에 한 번이라도 잃은 비율
        # 둘은 독립: 잃었다 되찾으면 hold=1, reloss=1. 1등 행동(점령→유지)을 쪼갠 신호.
        hold_total   = counters.get("capture_hold_k_total", 0)
        reloss_total = counters.get("post_reloss_k_total", 0)
        out["capture_hold_k_rate"] = (
            counters.get("capture_hold_k_success", 0) / max(hold_total, 1)
        )
        out["post_capture_reloss_rate_k"] = (
            counters.get("post_reloss_k_count", 0) / max(reloss_total, 1)
        )
        return out

    def summary(self):
        """본인의 counters/n_steps/episodes로 summary 구성."""
        return self.summary_from_counters(self.counters, self.n_steps, self.episodes)


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
