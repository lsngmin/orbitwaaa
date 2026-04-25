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
TRAIN_CONFIG = CONFIG["training"]

MAX_PLANETS  = CONFIG["env"]["max_planets"]
MAX_FLEETS   = CONFIG["env"]["max_fleets"]
HISTORY      = CONFIG["env"]["history_turns"]
PLANET_DIM      = 16  # 0-14: numeric features, 15: is_valid (1=real, 0=empty/sentinel)
PLANET_FEAT_DIM = 15  # planet_embed 입력 dim (is_valid 제외)
FLEET_DIM      = 9   # 0-6: numeric features, 7: src_idx, 8: dst_idx
                     # idx 7,8 sentinel: -2=empty slot (둘 다), -1=lookup miss(real fleet), ≥0=valid
FLEET_FEAT_DIM = 7   # fleet_embed 입력 dim (idx 제외)

# ships head: required_ships 배수 Categorical (commit 2)
SHIPS_MULTIPLIER_BINS = tuple(CONFIG["model"].get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
NUM_SHIPS_BINS        = len(SHIPS_MULTIPLIER_BINS)


ETA_NEAR = 5   # 1~5턴: 즉각 위협
ETA_MID  = 15  # 6~15턴: 중기 계획


def new_fleet_slot_state():
    """Fleet 슬롯 안정화용 상태 컨테이너.

    fleet_id → 고정 슬롯 매핑을 유지해 temporal attention 이 같은 슬롯에서
    같은 fleet 의 시간 궤적을 보게 한다. 함대 파괴 시 슬롯 회수, 새 fleet 에
    빈 슬롯 재할당.
    """
    return {
        "fleet_to_slot": {},
        "free_slots": list(range(MAX_FLEETS)),
    }


def update_fleet_slots(fleets, slot_state):
    """slot_state 를 in-place 갱신. 살아있는 fleet 에 안정 슬롯 부여.

    Args:
      fleets:     Fleet namedtuple 리스트 (env raw fleets 변환 후)
      slot_state: new_fleet_slot_state() 로 생성한 dict (mutated)
    Returns:
      fid_to_slot:    {fleet_id: slot_idx} (살아있는 fleet 만)
      slots_to_clear: set[slot_idx] — caller 가 history 에서 비워야 할 슬롯.
                      두 그룹의 union:
                        (1) newly_allocated   — 새 fleet 에 배정된 슬롯
                            (이전 거주자 trajectory ghost 차단)
                        (2) just_freed_no_reuse — 이번 턴 dead 인데 즉시 재할당
                            안 됐으니 free_slots 에 남는 슬롯
                            (current step 만 sentinel 이라 H-1 step 에 죽은
                             함대 잔재 → temporal pool 가 ghost 로 흡수)
                      두 경우 모두 history 클리어가 필요 → 한 set 으로 묶음.
    """
    fleet_to_slot = slot_state["fleet_to_slot"]
    free_slots    = slot_state["free_slots"]

    alive_ids = {f.id for f in fleets}

    # 죽은 fleet 의 슬롯 회수
    dead_ids   = set(fleet_to_slot.keys()) - alive_ids
    just_freed = set()
    for fid in dead_ids:
        slot = fleet_to_slot.pop(fid)
        free_slots.append(slot)
        just_freed.add(slot)
    free_slots.sort()  # 결정론적 할당

    fid_to_slot      = {}
    newly_allocated  = set()
    for f in fleets:
        if f.id in fleet_to_slot:
            fid_to_slot[f.id] = fleet_to_slot[f.id]
            continue
        if not free_slots:
            continue   # MAX_FLEETS 초과 — 무시 (cap 제한)
        slot = free_slots.pop(0)
        fleet_to_slot[f.id] = slot
        fid_to_slot[f.id]   = slot
        newly_allocated.add(slot)

    # 이번 턴 dead → 즉시 reuse 슬롯은 newly 에 들어가서 just_freed 와 겹쳐도 OK.
    # 두 set 의 union 이 곧 history 클리어 대상.
    return fid_to_slot, newly_allocated | just_freed


def clear_fleet_history_at_slots(fleet_history, slots):
    """deque 내 모든 과거 turn 의 지정 슬롯을 빈-슬롯 sentinel 로 리셋.

    새 fleet 이 이전에 다른 fleet 이 쓰던 슬롯을 받을 때, 그 슬롯의 과거 데이터는
    완전히 다른 fleet 의 궤적이라 ghost noise. zero+sentinel 로 덮어 써서 padding
    mask 가 attention 에서 제외할 수 있게 한다.
    """
    if not slots:
        return
    for arr in fleet_history:
        for slot in slots:
            arr[slot] = 0.0
            arr[slot, 7] = -2.0   # src_idx empty-slot sentinel
            arr[slot, 8] = -2.0   # dst_idx empty-slot sentinel


def encode_planets(raw_planets, raw_fleets, player, comet_ids, comets=None,
                   angular_velocity=0.0):
    """행성 목록을 (MAX_PLANETS, PLANET_DIM) 배열로 인코딩.

    velocity feature (idx 9,10):
      - 일반 orbiting 행성: `(vx, vy) = (-ω*(y-50), +ω*(x-50))` — 원형 접선 속도,
        `vel_scale = max(1, 50*|ω|)` 로 정규화
      - **comet (PR3)**: 엔진의 사전 계산된 `paths[i]` 의 다음 step 변위로 실제
        속도를 산출 (`comet_speed=4.0` 단위/턴 등속). `MAX_SPEED=6.0` 정규화 후
        clip [-1, 1]. 마지막 path 인덱스 (다음 턴 만료) 또는 path 데이터 부재 시 0.
      - 그 외 정적 행성: 0

    `comets` (optional): `obs.comets` — list of groups,
      group = `{"planet_ids": [...], "paths": [path0, ...], "path_index": int}`,
      `paths[i][path_index]` 가 i-번째 planet_id 의 현재 위치 `[x, y]`.
      None / [] 이면 comet velocity 는 0 (fallback — 학습 obs 에선 항상 채워짐).
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
    from prediction import CENTER_X, CENTER_Y, MAX_SPEED, is_orbiting, estimate_arrival_turn

    planets    = [Planet(*p) for p in raw_planets]
    fleets     = [Fleet(*f) for f in raw_fleets]
    vel_scale  = max(1.0, 50.0 * abs(float(angular_velocity)))

    # ── Comet velocity map: pid → (vx, vy) ──────────────────────────────────
    # 엔진은 paths 를 comet_speed (=4.0) arc-length 간격으로 샘플하므로
    # paths[i][idx+1] - paths[i][idx] = 다음 턴 변위 = 현재 턴 속도 벡터.
    # path_index 가 마지막 인덱스면 다음 턴 expire — 0 으로 둔다.
    comet_vel = {}
    if comets:
        for group in comets:
            pids   = group.get("planet_ids", []) if isinstance(group, dict) else getattr(group, "planet_ids", [])
            paths  = group.get("paths", [])      if isinstance(group, dict) else getattr(group, "paths", [])
            idx    = group.get("path_index", -1) if isinstance(group, dict) else getattr(group, "path_index", -1)
            if idx < 0:
                continue
            for j, pid in enumerate(pids):
                if j >= len(paths):
                    continue
                p_path = paths[j]
                if idx + 1 >= len(p_path):
                    continue   # 다음 턴 expire → velocity 0
                cur = p_path[idx]
                nxt = p_path[idx + 1]
                comet_vel[pid] = (float(nxt[0] - cur[0]), float(nxt[1] - cur[1]))

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
        is_comet      = (p.id in comet_ids)
        # is_orbiting 은 sun 거리 < 50 인 기하학적 사실. comet 은 타원 궤도라
        # perihelion/aphelion 사이에서 같은 행성이 0/1 깜빡임 → token feature
        # noise. 의미상 "ω 로 회전 중인 일반 행성" 만 1 로 두려고 comet 을 제외.
        orbiting      = is_orbiting(p) and not is_comet
        owner_me      = 1.0 if p.owner == player else 0.0
        owner_enemy   = 1.0 if p.owner not in (-1, player) else 0.0
        owner_neutral = 1.0 if p.owner == -1 else 0.0
        if is_comet:
            cvx, cvy = comet_vel.get(p.id, (0.0, 0.0))
            vx_norm = float(np.clip(cvx / MAX_SPEED, -1.0, 1.0))
            vy_norm = float(np.clip(cvy / MAX_SPEED, -1.0, 1.0))
        elif orbiting:
            vx = -float(angular_velocity) * (p.y - CENTER_Y)
            vy =  float(angular_velocity) * (p.x - CENTER_X)
            vx_norm = float(np.clip(vx / vel_scale, -1.0, 1.0))
            vy_norm = float(np.clip(vy / vel_scale, -1.0, 1.0))
        else:
            vx_norm = 0.0
            vy_norm = 0.0

        arr[i] = [
            p.x / 100.0,
            p.y / 100.0,
            owner_me,
            owner_enemy,
            owner_neutral,
            min(p.ships / 1000.0, 1.0),
            p.production / 5.0,
            1.0 if orbiting else 0.0,
            1.0 if is_comet else 0.0,
            vx_norm,
            vy_norm,
            min(enemy_near[p.id] / 1000.0, 1.0),
            min(enemy_mid[p.id]  / 1000.0, 1.0),
            min(mine_near[p.id]  / 1000.0, 1.0),
            min(mine_mid[p.id]   / 1000.0, 1.0),
            1.0,   # is_valid sentinel (빈 슬롯은 zero-init 으로 0)
        ]
    return arr


def encode_fleets(raw_fleets, raw_planets, player, fid_to_slot=None):
    """fleet 목록을 (MAX_FLEETS, FLEET_DIM) 배열로 인코딩.

    feature layout (FLEET_DIM=9):
      0,1: x, y                              — `/100` 정규화
      2,3: vx_fleet, vy_fleet                — `fleet_speed(ships) × (cos, sin)`,
                                               `/MAX_SPEED` 정규화로 ~`[-1,1]`
      4:   ships                             — `min(ships/1000, 1.0)` (전투력)
      5,6: is_mine, is_enemy                 — 0/1
      7:   src_idx                            — source-planet fusion lookup 포인터
      8:   dst_idx                            — destination-planet fusion lookup 포인터
                                               (ray-cast 첫 충돌 행성 idx)
        sentinel (idx 7,8 공통):
          -2 = 빈 슬롯 (padding mask 대상, 둘 다 -2 로 set)
          -1 = real fleet 인데 lookup 실패 (src 못 찾음 / 충돌 행성 없음)
               → 위치/속도/ships valid → padding 안 함, fusion 만 valid mask 로 차단
          ≥0 = valid planet idx

    bilinear precomputation: `speed(ships) × direction` 곱셈을 미리 풀어
    모델이 첫 Linear 레이어에서 학습할 필요 없게 만든다 — planet vx/vy 와 같은 논리.
    방향은 atan2(vy, vx) 로 복원 가능 (속도 0 이면 방향 정보 없음 — 빈 슬롯과 동치).

    dst_idx ray-cast 는 encode_planets 의 ETA 집계와 같은 logic (radius * 1.5 lenient
    margin) — 행성 회전/sweep_fleets 미시뮬 보상.

    fid_to_slot:
      None  → enumerate 순서로 슬롯 할당 (stateless; 단발성 인코딩, 테스트 호환)
      dict  → {fleet_id: slot_idx} 안정 매핑 (env wrapper / submission agent 가
              update_fleet_slots 로 생성). 매 턴 같은 fleet 이 같은 슬롯에 들어가
              temporal attention 이 진짜 궤적을 학습.
    """
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
    from prediction import fleet_speed, MAX_SPEED

    fleets    = [Fleet(*f)  for f in raw_fleets]
    planets   = [Planet(*p) for p in raw_planets]
    truncated = planets[:MAX_PLANETS]
    id_to_idx = {p.id: idx for idx, p in enumerate(truncated)}

    arr = np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)
    arr[:, 7] = -2.0   # src_idx empty-slot sentinel
    arr[:, 8] = -2.0   # dst_idx empty-slot sentinel
    for i, f in enumerate(fleets):
        if fid_to_slot is None:
            if i >= MAX_FLEETS:
                break
            slot = i
        else:
            slot = fid_to_slot.get(f.id)
            if slot is None or slot >= MAX_FLEETS:
                continue
        src_idx = id_to_idx.get(f.from_planet_id, -1)

        # destination ray-cast: 첫 충돌 행성 idx (게임 규칙: 첫 충돌에서 소멸)
        dx = math.cos(f.angle)
        dy = math.sin(f.angle)
        dst_idx = -1
        first_t = math.inf
        for j, p in enumerate(truncated):
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
                first_t = t
                dst_idx = j

        speed = fleet_speed(f.ships)
        vx    = speed * math.cos(f.angle)
        vy    = speed * math.sin(f.angle)
        arr[slot] = [
            f.x / 100.0,
            f.y / 100.0,
            vx / MAX_SPEED,
            vy / MAX_SPEED,
            min(f.ships / 1000.0, 1.0),
            1.0 if f.owner == player else 0.0,
            1.0 if f.owner not in (-1, player) else 0.0,
            float(src_idx),
            float(dst_idx),
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
        self._fleet_slot_state = new_fleet_slot_state()

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
        comets      = raw.get("comets", []) if isinstance(raw, dict) else getattr(raw, "comets", []) or []
        self._planets_snapshot = raw_planets
        return raw_planets, raw_fleets, comet_ids, comets

    def _build_tensor(self):
        """flat layout: 턴별 interleave (model.forward 의 view 가정과 일치).

        per_turn = [planet_block(P*PD), fleet_block(F*FD)]
        flat     = per_turn[h=0] || per_turn[h=1] || ... || per_turn[H-1]

        과거: `[p_all, f_all]` 분리 concat → model.view(B, H, p_size+f_size)
        와 misalign, h=H-1 의 fleet 만 우연히 맞아 temporal attention 이
        무의미한 데이터로 학습되던 버그를 수정.
        """
        p_hist = np.stack(list(self._planet_history), axis=0)  # (H, P, PD)
        f_hist = np.stack(list(self._fleet_history),  axis=0)  # (H, F, FD)
        p_flat = p_hist.reshape(HISTORY, MAX_PLANETS * PLANET_DIM)  # (H, p_size)
        f_flat = f_hist.reshape(HISTORY, MAX_FLEETS  * FLEET_DIM)   # (H, f_size)
        per_turn = np.concatenate([p_flat, f_flat], axis=1)         # (H, p_size+f_size)
        return per_turn.reshape(-1).astype(np.float32)              # (H*(p_size+f_size),)

    def _encode_fleets_stable(self, raw_fleets, raw_planets):
        """Slot-stable fleet encoding. fleet_id → 고정 슬롯 매핑 갱신 후
        새로 배정된 슬롯의 history 를 클리어해 ghost trajectory 를 막는다."""
        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
        fleets = [Fleet(*f) for f in raw_fleets]
        fid_to_slot, slots_to_clear = update_fleet_slots(fleets, self._fleet_slot_state)
        clear_fleet_history_at_slots(self._fleet_history, slots_to_clear)
        return encode_fleets(raw_fleets, raw_planets, self._player, fid_to_slot)

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
        # init sentinel 채우기 — zero padding 도 빈 슬롯으로 인식되도록 (-2 = empty)
        for arr in self._fleet_history:
            arr[:, 7] = -2.0   # src_idx
            arr[:, 8] = -2.0   # dst_idx
        self._fleet_slot_state = new_fleet_slot_state()

        self._av = 0.0

        # 첫 관측
        self._env.reset()
        raw_planets, raw_fleets, comet_ids, comets = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids, comets, self._av))
        self._fleet_history.append(self._encode_fleets_stable(raw_fleets, raw_planets))

        return self._build_tensor(), {}

    def step(self, action):
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
        from prediction import aim, crosses_sun, resolve_ships_for_capture

        raw_planets, raw_fleets, comet_ids, comets = self._current_obs()
        planets = [Planet(*p) for p in raw_planets]
        fleets  = [Fleet(*f) for f in raw_fleets]
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
            # 동적: in-flight fleet 효과를 ETA forward sim 으로 반영.
            ships_needed, angle, tx, ty, _, _, _ = resolve_ships_for_capture(
                p, target, av, multiplier, p.ships,
                fleets=fleets, planets=planets,
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
        raw_planets, raw_fleets, comet_ids, comets = self._current_obs()
        self._planet_history.append(encode_planets(raw_planets, raw_fleets, self._player, comet_ids, comets, self._av))
        self._fleet_history.append(self._encode_fleets_stable(raw_fleets, raw_planets))
        obs = self._build_tensor()

        # 보상 및 종료
        state    = self._env.state
        done     = self._env.done
        reward   = 0.0
        if done:
            r = state[self._player].reward
            terminal_win_reward = float(TRAIN_CONFIG.get("terminal_win_reward", 1.0))
            reward = terminal_win_reward if r == 1 else (
                -terminal_win_reward if r == -1 else 0.0
            )

        return obs, reward, done, False, {}

    def render(self):
        pass
