import os
import torch


def save_checkpoint(path, model, optimizer, generation, total_steps, league_agents):
    """학습 상태 전체 저장 (재개용)."""
    torch.save({
        "generation":    generation,
        "total_steps":   total_steps,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "league_agents": league_agents,  # [(path, win_rate, generation), ...]
    }, path)


def load_checkpoint(path, model, optimizer, device):
    """체크포인트 로드. 없으면 None 반환."""
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    print(f"체크포인트 로드: gen={ckpt['generation']}, steps={ckpt['total_steps']:,}")
    return ckpt["generation"], ckpt["total_steps"], ckpt["league_agents"]
