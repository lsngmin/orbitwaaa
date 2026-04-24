"""
OrbitWars Gymnasium 환경 wrapper.

관측(obs) → 고정 크기 텐서로 변환
행동(action) → [from_planet_id, angle, num_ships] 리스트로 변환
"""

import os
import math
import numpy as np
import yaml
from collections import deque

try:
    import gymnasium as gym
    from gymnasium import spaces
    from kaggle_environments import make
except ImportError:
    pass

_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_cfg_path) as f:
    CONFIG = yaml.safe_load(f)

MAX_PLANETS  = CONFIG["env"]["max_planets"]
MAX_FLEETS   = CONFIG["env"]["max_fleets"]
HISTORY      = CONFIG["env"]["history_turns"]
PLANET_DIM   = 18  # +3: min_eta_norm, pred_x, pred_y / +2: sun_block, sun_dist_norm
FLEET_DIM    = 7

# ships head: required_ships 배수 Categorical (commit 2)
SHIPS_MULTIPLIER_BINS = tuple(CONFIG["model"].get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
NUM_SHIPS_BINS        = len(SHIPS_MULTIPLIER_BINS)


ETA_NEAR = 5   # 1~5턴: 즉각 위협
ETA_MID  = 15  # 6~15턴: 중기 계획


def encode_planets(raw_planets, raw_fleets, player, comet_ids, angular_velocity=0.0):
    """행성 목록을 (MAX_PLANETS, PLANET_DIM) 배열로 인코딩."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from prediction import is_orbiting, estimate_arrival_turn, predict_position, crosses_sun, sun_approach_distance

    planets    = [Planet(*p) for p in raw_planets]
    fleets     = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]

    # 행성별 ETA bin 집계: near(1~5턴) / mid(6~15턴)
    enemy_near = {p.id: 0 for p in planets}
    enemy_mid  = {p.id: 0 for p in planets}
    mine_near  = {p.id: 0 for p in planets}
    mine_mid   = {p.id: 0 for p in planets}

    for f in fleets:
        dx = math.cos(f.angle)
        dy = math.sin(f.angle)

        # 레이 상에서 가장 먼저 충돌하는 행성 하나만 선택 (게임 규칙: 첫 충돌에서 소멸)
        first_planet = None
        first_t      = math.inf
        for p in planets:
            fx = f.x - p.x
            fy = f.y - p.y
            t  = -(fx * dx + fy * dy)
            if t <= 0:
                continue
            cx = f.x + t * dx
            cy = f.y + t * dy
            if math.hypot(cx - p.x, cy - p.y) > p.radius * 1.5:
                continue
            if t < first_t:
                first_t      = t
                first_planet = p

        if first_planet is None:
            continue

        # ETA는 레이 진행 거리(t)로 계산 — center-to-center 대신 실제 도달 거리
        eta = estimate_arrival_turn(first_t, f.ships)

        if f.owner == player:
            if eta <= ETA_NEAR:
                mine_near[first_planet.id] += f.ships
            elif eta <= ETA_MID:
                mine_mid[first_planet.id]  += f.ships
        else:
            if eta <= ETA_NEAR:
                enemy_near[first_planet.id] += f.ships
            elif eta <= ETA_MID:
                enemy_mid[first_planet.id]  += f.ships

    arr = np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        owner_me      = 1.0 if p.owner == player else 0.0
        owner_enemy   = 1.0 if p.owner not in (-1, player) else 0.0
        owner_neutral = 1.0 if p.owner == -1 else 0.0

        # ETA feature: 내 행성 중 가장 가까운 곳에서 이 행성까지의 최소 ETA
        # 자기 자신은 스킵 (공격 대상이 아님 → 거리 0이면 feature 무의미)
        min_eta = 50.0
        if my_planets:
            for src in my_planets:
                if src.id == p.id:
                    continue
                dist = math.hypot(p.x - src.x, p.y - src.y)
                eta_to = estimate_arrival_turn(dist, 50)  # 50 ships 기준 근사
                if eta_to < min_eta:
                    min_eta = float(eta_to)
        min_eta_norm = min(min_eta / 50.0, 1.0)

        # 예상 도착 위치: min_eta 턴 후 이 행성의 위치
        pred_x, pred_y = predict_position(p, angular_velocity, int(min_eta))

        # 태양 위험도: 내 행성 → 이 행성 경로 기준 (자기 자신은 스킵)
        sun_block = 0.0
        sun_dist_min = 50.0
        if my_planets:
            for src in my_planets:
                if src.id == p.id:
                    continue
                sd = sun_approach_distance(src.x, src.y, pred_x, pred_y)
                if sd < sun_dist_min:
                    sun_dist_min = sd
                if crosses_sun(src.x, src.y, pred_x, pred_y):
                    sun_block = 1.0
        sun_dist_norm = min(sun_dist_min / 50.0, 1.0)

        arr[i] = [
            p.x / 100.0,
            p.y / 100.0,
            owner_me,
            owner_enemy,
            owner_neutral,
            min(p.ships / 1000.0, 1.0),
            p.production / 5.0,
            1.0 if is_orbiting(p) else 0.0,
            1.0 if p.id in comet_ids else 0.0,
            min(enemy_near[p.id] / 1000.0, 1.0),
            min(enemy_mid[p.id]  / 1000.0, 1.0),
            min(mine_near[p.id]  / 1000.0, 1.0),
            min(mine_mid[p.id]   / 1000.0, 1.0),
            # ── ETA / 궤도 ──
            min_eta_norm,
            pred_x / 100.0,
            pred_y / 100.0,
            # ── 태양 위험도 ──
            sun_block,
            sun_dist_norm,
        ]
    return arr


def encode_fleets(raw_fleets, player):
    """fleet 목록을 (MAX_FLEETS, FLEET_DIM) 배열로 인코딩."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet

    fleets = [Fleet(*f) for f in raw_fleets]
    arr    = np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)
    for i, f in enumerate(fleets[:MAX_FLEETS]):
        arr[i] = [
            f.x / 100.0,
            f.y / 100.0,
            math.cos(f.angle),
            math.sin(f.angle),
            min(f.ships / 1000.0, 1.0),
            1.0 if f.owner == player else 0.0,
            1.0 if f.owner not in (-1, player) else 0.0,
        ]
    return arr


class OrbitWarsEnv(gym.Env):
    """
    관측 공간:
      planets_history: (HISTORY, MAX_PLANETS, PLANET_DIM)
      fleets_history:  (HISTORY, MAX_FLEETS,  FLEET_DIM)

    행동 공간:
      (MAX_PLANETS, 1 + NUM_SHIPS_BINS + MAX_PLANETS)
      action[i, 0]                        = 발사 여부 (0~1, 0.5 이상이면 발사)
      action[i, 1:1+NUM_SHIPS_BINS]       = ships_bin one-hot (required 배수 선택)
      action[i, 1+NUM_SHIPS_BINS:]        = 타겟 one-hot (argmax로 선택)

    ships 계산 (decode 시):
      multiplier   = SHIPS_MULTIPLIER_BINS[ships_bin]
      required     = target.ships + target.production × turns + 1
      ships_needed = min(max(int(required × multiplier), 1), src.ships)
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        obs_dim = HISTORY * (MAX_PLANETS * PLANET_DIM + MAX_FLEETS * FLEET_DIM)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(MAX_PLANETS, 1 + NUM_SHIPS_BINS + MAX_PLANETS),
            dtype=np.float32
        )

        self._planet_history = deque(
            [np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY,
            maxlen=HISTORY
        )
        self._fleet_history = deque(
            [np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)] * HISTORY,
            maxlen=HISTORY
        )

        self._env    = None
        self._player = 0
        self._done   = False
        self._step   = 0
        self._planets_snapshot = []
        self._av     = 0.0

    def _get_obs_raw(self):
        obs = self._env.state[self._player].observation
        if isinstance(obs, dict):
            return obs
        return obs.__dict__ if hasattr(obs, '__dict__') else {}

    def _current_obs(self):
        raw = self._get_obs_raw()
        raw_planets = raw.get("planets", []) if isinstance(raw, dict) else getattr(raw, "planets", [])
        raw_fleets  = raw.get("fleets",  []) if isinstance(raw, dict) else getattr(raw, "fleets",  [])
        comet_ids   = set(raw.get("comet_planet_ids", []) if isinstance(raw, dict) else getattr(raw, "comet_planet_ids", []) or [])
        self._planets_snapshot = raw_planets
        return raw_planets, raw_fleets, comet_ids

    def _build_tensor(self):
        p_hist = np.stack(list(self._planet_history), axis=0)  # (H, P, Dp)
        f_hist = np.stack(list(self._fleet_history),  axis=0)  # (H, F, Df)
        return np.concatenate([p_hist.flatten(), f_hist.flatten()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env    = make("orbit_wars", debug=False)
        self._player = 0
        self._done   = False
        self._step   = 0

        self._planet_history = deque(
            [np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)] * HISTORY,
            maxlen=HISTORY
        )
        self._fleet_history = deque(
            [np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)] * HISTORY,
            maxlen=HISTORY
        )

        self._av = 0.0

        # 첫 관측
        self._env.reset()
        raw_planets, raw_fleets, comet_ids = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids, self._av))
        self._fleet_history.append(encode_fleets(raw_fleets, self._player))

        return self._build_tensor(), {}

    def step(self, action):
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
        from prediction import aim, crosses_sun, resolve_ships_for_capture

        raw_planets, raw_fleets, comet_ids = self._current_obs()
        planets = [Planet(*p) for p in raw_planets]
        raw = self._get_obs_raw()
        av  = raw.get("angular_velocity", 0) if isinstance(raw, dict) else getattr(raw, "angular_velocity", 0)
        self._av = av

        my_planets = [p for p in planets if p.owner == self._player]
        moves      = []

        for i, p in enumerate(planets[:MAX_PLANETS]):
            if p.owner != self._player:
                continue

            launch_prob      = float(action[i, 0])
            ships_bin_logits = action[i, 1:1 + NUM_SHIPS_BINS]
            target_logits    = action[i, 1 + NUM_SHIPS_BINS:]

            if launch_prob < 0.5:
                continue

            # 타겟 선택 (argmax, 유효한 타겟만)
            target_probs = np.array(target_logits[:len(planets)])
            # 내 행성은 타겟 제외
            for j, t in enumerate(planets[:MAX_PLANETS]):
                if t.owner == self._player:
                    target_probs[j] = -np.inf
            if np.all(np.isinf(target_probs)):
                continue
            target_idx = int(np.argmax(target_probs))
            target     = planets[target_idx]

            # ships_bin → multiplier → ships_needed
            ships_bin  = int(np.argmax(ships_bin_logits))
            multiplier = float(SHIPS_MULTIPLIER_BINS[ships_bin])

            # 고정점 반복으로 (ships_needed, required) 동시 해결 (commit 3).
            ships_needed, angle, tx, ty, _, _, _ = resolve_ships_for_capture(
                p, target, av, multiplier, p.ships,
            )
            if ships_needed <= 0:
                continue

            if crosses_sun(p.x, p.y, tx, ty):
                continue

            moves.append([p.id, angle, ships_needed])

        # 환경 스텝
        self._env.step([moves, None])
        self._step += 1

        # 새 관측
        raw_planets, raw_fleets, comet_ids = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids, self._av))
        self._fleet_history.append(encode_fleets(raw_fleets, self._player))
        obs = self._build_tensor()

        # 보상 및 종료
        state    = self._env.state
        done     = self._env.done
        reward   = 0.0
        if done:
            r = state[self._player].reward
            reward = 1.0 if r == 1 else (-1.0 if r == -1 else 0.0)

        return obs, reward, done, False, {}

    def render(self):
        pass
