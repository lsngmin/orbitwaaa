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


def load_checkpoint(path, model, optimizer, device, strict=True):
    """체크포인트 로드. 없으면 None 반환.

    strict=False (partial transfer):
      - model.load_state_dict(strict=False) — missing/unexpected key 로그 출력.
      - optimizer state 가 새 파라미터와 mismatch 시 fresh start (warning 만).
      - Step transition (예: target_head → target_q/target_k) 시 encoder
        + amount_pair_head + critic 만 transfer 하고 새 head 는 init 그대로.
    """
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)
    res  = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict and (res.missing_keys or res.unexpected_keys):
        print(f"[partial transfer] missing keys ({len(res.missing_keys)}): {res.missing_keys}")
        print(f"[partial transfer] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except (ValueError, KeyError) as e:
        if strict:
            raise
        print(f"[partial transfer] optimizer state mismatch — fresh start: {e}")

    print(f"체크포인트 로드: gen={ckpt['generation']}, steps={ckpt['total_steps']:,}")
    return ckpt["generation"], ckpt["total_steps"], ckpt["league_agents"]
