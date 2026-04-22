"""
OrbitWars Gymnasium 환경 wrapper.

관측(obs) → 고정 크기 텐서로 변환
행동(action) → [from_planet_id, angle, num_ships] 리스트로 변환
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from kaggle_environments import make
import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

MAX_PLANETS  = CONFIG["env"]["max_planets"]
MAX_FLEETS   = CONFIG["env"]["max_fleets"]
HISTORY      = CONFIG["env"]["history_turns"]
PLANET_DIM   = 11
FLEET_DIM    = 7


def encode_planets(raw_planets, raw_fleets, player, comet_ids):
    """행성 목록을 (MAX_PLANETS, PLANET_DIM) 배열로 인코딩."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from prediction import is_orbiting, fleet_speed

    planets = [Planet(*p) for p in raw_planets]
    fleets  = [Fleet(*f) for f in raw_fleets]

    # 행성별 incoming fleet 집계
    incoming_enemy = {p.id: 0 for p in planets}
    incoming_mine  = {p.id: 0 for p in planets}
    for f in fleets:
        for p in planets:
            dx = math.cos(f.angle)
            dy = math.sin(f.angle)
            fx = f.x - p.x
            fy = f.y - p.y
            t  = -(fx * dx + fy * dy)
            if t > 0:
                cx = f.x + t * dx
                cy = f.y + t * dy
                if math.hypot(cx - p.x, cy - p.y) <= p.radius * 1.5:
                    if f.owner == player:
                        incoming_mine[p.id]  += f.ships
                    else:
                        incoming_enemy[p.id] += f.ships

    arr = np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)
    for i, p in enumerate(planets[:MAX_PLANETS]):
        owner_me      = 1.0 if p.owner == player else 0.0
        owner_enemy   = 1.0 if p.owner not in (-1, player) else 0.0
        owner_neutral = 1.0 if p.owner == -1 else 0.0
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
            min(incoming_enemy[p.id] / 1000.0, 1.0),
            min(incoming_mine[p.id]  / 1000.0, 1.0),
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
      (MAX_PLANETS, MAX_PLANETS + 2)
      action[i, 0]    = 발사 여부 (0~1, 0.5 이상이면 발사)
      action[i, 1]    = ships 비율 (0~1)
      action[i, 2:]   = 타겟 logits (argmax로 선택)
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
            shape=(MAX_PLANETS, MAX_PLANETS + 2),
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

        # 첫 관측
        self._env.reset()
        raw_planets, raw_fleets, comet_ids = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids))
        self._fleet_history.append(encode_fleets(raw_fleets, self._player))

        return self._build_tensor(), {}

    def step(self, action):
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
        from prediction import aim, crosses_sun

        raw_planets, raw_fleets, comet_ids = self._current_obs()
        planets = [Planet(*p) for p in raw_planets]
        raw = self._get_obs_raw()
        av  = raw.get("angular_velocity", 0) if isinstance(raw, dict) else getattr(raw, "angular_velocity", 0)

        my_planets = [p for p in planets if p.owner == self._player]
        moves      = []

        for i, p in enumerate(planets[:MAX_PLANETS]):
            if p.owner != self._player:
                continue

            launch_prob  = float(action[i, 0])
            ships_ratio  = float(action[i, 1])
            target_logits = action[i, 2:]

            if launch_prob < 0.0:
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

            ships_ratio  = (ships_ratio + 1.0) / 2.0  # [-1,1] → [0,1]
            ships_needed = max(1, int(p.ships * ships_ratio))
            ships_needed = min(ships_needed, p.ships)
            if ships_needed <= 0:
                continue

            angle = aim(p, target, av, ships_needed)
            tx = p.x + math.cos(angle) * math.hypot(target.x - p.x, target.y - p.y)
            ty = p.y + math.sin(angle) * math.hypot(target.x - p.x, target.y - p.y)
            if crosses_sun(p.x, p.y, tx, ty):
                continue

            moves.append([p.id, angle, ships_needed])

        # 환경 스텝
        self._env.step([moves, None])
        self._step += 1

        # 새 관측
        raw_planets, raw_fleets, comet_ids = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids))
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
