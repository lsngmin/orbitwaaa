"""
유전 알고리즘으로 strategy.py 가중치 자동 튜닝.
토너먼트 방식: 개체끼리 서로 대전해서 승점으로 평가.

실행:
    python tune.py

결과:
    best_weights.json — 최적 가중치 저장
"""

import json
import random
import os
from multiprocessing import Pool
from kaggle_environments import make

# 튜닝할 가중치 범위
WEIGHT_RANGES = {
    "W_PRODUCTION": (0.5, 10.0),
    "W_SHIPS":      (0.1, 2.0),
    "W_DISTANCE":   (0.01, 0.5),
    "W_ENEMY":      (0.5, 8.0),
}

# GA 설정
POPULATION_SIZE  = 64   # 한 세대 개체 수
GENERATIONS      = 20   # 세대 수
ELITE_RATIO      = 0.25 # 상위 25% 생존
MUTATION_RATE    = 0.2  # 돌연변이 확률
MATCHES_PER_EVAL = 6    # 개체당 대전 상대 수 (랜덤 샘플)


def random_individual():
    return {k: random.uniform(*v) for k, v in WEIGHT_RANGES.items()}


def make_agent(weights):
    def agent(obs):
        import math as _math
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
        from prediction import aim, crosses_sun, estimate_arrival_turn, predict_position

        if isinstance(obs, dict):
            player      = obs.get("player", 0)
            raw_planets = obs.get("planets", [])
            raw_fleets  = obs.get("fleets", [])
            av          = obs.get("angular_velocity", 0)
            comet_ids   = set(obs.get("comet_planet_ids", []))
        else:
            player      = obs.player
            raw_planets = obs.planets
            raw_fleets  = obs.fleets
            av          = obs.angular_velocity
            comet_ids   = set(obs.comet_planet_ids or [])

        planets = [Planet(*p) for p in raw_planets]
        fleets  = [Fleet(*f) for f in raw_fleets]

        def fleet_will_hit(fleet, planet):
            dx = _math.cos(fleet.angle)
            dy = _math.sin(fleet.angle)
            fx = fleet.x - planet.x
            fy = fleet.y - planet.y
            t = -(fx * dx + fy * dy)
            if t < 0:
                return False
            cx = fleet.x + t * dx
            cy = fleet.y + t * dy
            return _math.hypot(cx - planet.x, cy - planet.y) <= planet.radius * 1.5

        my_planets = [p for p in planets if p.owner == player]
        targets    = [p for p in planets if p.owner != player]
        if not targets:
            return []

        in_transit = {}
        for f in fleets:
            if f.owner == player:
                for t in targets:
                    if fleet_will_hit(f, t):
                        in_transit[t.id] = in_transit.get(t.id, 0) + f.ships

        moves    = []
        assigned = {}

        for mine in my_planets:
            threat    = sum(f.ships for f in fleets if f.owner != player and fleet_will_hit(f, mine))
            available = mine.ships - threat - 1
            if available <= 0:
                continue

            best_score  = float('-inf')
            best_target = None
            best_angle  = None
            best_needed = None

            for t in targets:
                ships_needed = t.ships + 1
                if in_transit.get(t.id, 0) + assigned.get(t.id, 0) >= ships_needed:
                    continue
                if available < ships_needed:
                    continue

                a = aim(mine, t, av, ships_needed)
                tx = mine.x + _math.cos(a) * _math.hypot(t.x - mine.x, t.y - mine.y)
                ty = mine.y + _math.sin(a) * _math.hypot(t.x - mine.x, t.y - mine.y)
                if crosses_sun(mine.x, mine.y, tx, ty):
                    continue

                dist  = _math.hypot(mine.x - t.x, mine.y - t.y)
                score = (
                    t.production * weights["W_PRODUCTION"]
                    - t.ships    * weights["W_SHIPS"]
                    - dist       * weights["W_DISTANCE"]
                    + (1.0 if t.owner >= 0 else 0.0) * weights["W_ENEMY"]
                    - (1.5 if t.id in comet_ids else 0.0)
                )

                if score > best_score:
                    best_score  = score
                    best_target = t
                    best_angle  = a
                    best_needed = ships_needed

            if best_target is None:
                continue

            moves.append([mine.id, best_angle, best_needed])
            assigned[best_target.id] = assigned.get(best_target.id, 0) + best_needed

        return moves
    return agent


def play_match(args):
    """두 가중치 조합 대전. (이긴 쪽 인덱스, 0=무승부) 반환."""
    w1, w2 = args
    try:
        env = make("orbit_wars", debug=False)
        env.run([make_agent(w1), make_agent(w2)])
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        if r0 > r1:
            return 0    # w1 승
        elif r1 > r0:
            return 1    # w2 승
        else:
            return -1   # 무승부
    except Exception:
        return -1


def tournament(population):
    """
    각 개체를 MATCHES_PER_EVAL명 랜덤 상대와 대전.
    승=1점, 무=0.5점, 패=0점으로 총점 계산.
    """
    n = len(population)
    scores = [0.0] * n

    # 대전 쌍 생성
    pairs = []
    pair_idx = []
    for i in range(n):
        opponents = random.sample([j for j in range(n) if j != i], MATCHES_PER_EVAL)
        for j in opponents:
            pairs.append((population[i], population[j]))
            pair_idx.append((i, j))

    with Pool(processes=min(len(pairs), os.cpu_count())) as pool:
        results = pool.map(play_match, pairs)

    for (i, j), result in zip(pair_idx, results):
        if result == 0:
            scores[i] += 1.0
        elif result == 1:
            scores[j] += 1.0
        else:
            scores[i] += 0.5
            scores[j] += 0.5

    # 정규화 (0~1)
    max_score = MATCHES_PER_EVAL * n
    return [s / max_score for s in scores]


def crossover(a, b):
    return {k: a[k] if random.random() < 0.5 else b[k] for k in WEIGHT_RANGES}


def mutate(individual):
    result = dict(individual)
    for k, (lo, hi) in WEIGHT_RANGES.items():
        if random.random() < MUTATION_RATE:
            result[k] = random.uniform(lo, hi)
    return result


def run():
    print(f"GA 시작 (토너먼트): {POPULATION_SIZE}개체 × {GENERATIONS}세대 × {MATCHES_PER_EVAL}대전/개체")
    population   = [random_individual() for _ in range(POPULATION_SIZE)]
    best_overall = None
    best_score   = -1.0

    for gen in range(GENERATIONS):
        scores = tournament(population)
        ranked = sorted(zip(scores, population), key=lambda x: -x[0])
        top_score, top_weights = ranked[0]

        if top_score > best_score:
            best_score   = top_score
            best_overall = top_weights
            with open("best_weights.json", "w") as f:
                json.dump(best_overall, f, indent=2)

        avg = sum(scores) / len(scores)
        print(f"Gen {gen+1:02d} | best={top_score:.3f} | avg={avg:.3f} | {top_weights}")

        n_elite = max(2, int(POPULATION_SIZE * ELITE_RATIO))
        elites  = [w for _, w in ranked[:n_elite]]

        next_gen = list(elites)
        while len(next_gen) < POPULATION_SIZE:
            a, b = random.sample(elites, 2)
            next_gen.append(mutate(crossover(a, b)))
        population = next_gen

    print(f"\n최적 가중치 (토너먼트 점수 {best_score:.3f}):")
    print(json.dumps(best_overall, indent=2))
    print("→ best_weights.json 저장 완료")


if __name__ == "__main__":
    run()
