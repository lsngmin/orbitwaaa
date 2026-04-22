from ppo_agent import ppo_agent
from strategy import decide


def agent(obs):
    try:
        moves = ppo_agent(obs)
        if moves:
            return moves
    except Exception:
        pass
    return decide(obs)
