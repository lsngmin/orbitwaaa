"""
유전 알고리즘으로 strategy.py 가중치 자동 튜닝.

실행:
    python tune.py

결과:
    best_weights.json — 최적 가중치 저장
"""

import json
import math
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
POPULATION_SIZE = 64    # 한 세대 개체 수 (코어 수에 맞춤)
GENERATIONS     = 20    # 세대 수
ELITE_RATIO     = 0.25  # 상위 25% 생존
MUTATION_RATE   = 0.2   # 돌연변이 확률
GAMES_PER_EVAL  = 5     # 개체당 자가대전 판 수


def random_individual():
    return {k: random.uniform(*v) for k, v in WEIGHT_RANGES.items()}


def make_agent(weights):
    """가중치를 주입한 agent 함수 반환."""
    def agent(obs):
        import math as _math
        from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

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

        from prediction import aim, crosses_sun, estimate_arrival_turn, predict_position, fleet_speed

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


def evaluate(weights):
    """가중치 조합의 승률 측정 (자가대전 GAMES_PER_EVAL판)."""
    wins = 0
    agent = make_agent(weights)
    for _ in range(GAMES_PER_EVAL):
        try:
            env = make("orbit_wars", debug=False)
            env.run([agent, agent])
            r = env.steps[-1][0].reward
            if r == 1:
                wins += 1
            elif r == 0:
                wins += 0.5
        except Exception:
            pass
    return wins / GAMES_PER_EVAL


def evaluate_vs_random(weights):
    """random 상대로 승률 측정."""
    wins = 0
    agent = make_agent(weights)
    for _ in range(GAMES_PER_EVAL):
        try:
            env = make("orbit_wars", debug=False)
            env.run([agent, "random"])
            r = env.steps[-1][0].reward
            if r == 1:
                wins += 1
            elif r == 0:
                wins += 0.5
        except Exception:
            pass
    return wins / GAMES_PER_EVAL


def crossover(a, b):
    child = {}
    for k in WEIGHT_RANGES:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return child


def mutate(individual):
    result = dict(individual)
    for k, (lo, hi) in WEIGHT_RANGES.items():
        if random.random() < MUTATION_RATE:
            result[k] = random.uniform(lo, hi)
    return result


def run():
    print(f"GA 시작: {POPULATION_SIZE}개체 × {GENERATIONS}세대 × {GAMES_PER_EVAL}판")
    population = [random_individual() for _ in range(POPULATION_SIZE)]

    best_overall = None
    best_score   = -1.0

    for gen in range(GENERATIONS):
        # 병렬 평가
        with Pool(processes=min(POPULATION_SIZE, os.cpu_count())) as pool:
            scores = pool.map(evaluate_vs_random, population)

        ranked = sorted(zip(scores, population), key=lambda x: -x[0])
        top_score, top_weights = ranked[0]

        if top_score > best_score:
            best_score   = top_score
            best_overall = top_weights
            with open("best_weights.json", "w") as f:
                json.dump(best_overall, f, indent=2)

        print(f"Gen {gen+1:02d} | best={top_score:.3f} | avg={sum(scores)/len(scores):.3f} | weights={top_weights}")

        # 상위 25% 생존
        n_elite = max(2, int(POPULATION_SIZE * ELITE_RATIO))
        elites  = [w for _, w in ranked[:n_elite]]

        # 다음 세대 생성
        next_gen = list(elites)
        while len(next_gen) < POPULATION_SIZE:
            a, b  = random.sample(elites, 2)
            child = mutate(crossover(a, b))
            next_gen.append(child)
        population = next_gen

    print(f"\n최적 가중치 (승률 {best_score:.1%}):")
    print(json.dumps(best_overall, indent=2))
    print("→ best_weights.json 저장 완료")


if __name__ == "__main__":
    run()
