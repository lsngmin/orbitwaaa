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


def load_checkpoint(path, model, optimizer, device, strict=True, reset_keys=None):
    """체크포인트 로드. 없으면 None 반환.

    reset_keys: 이 리스트의 키는 ckpt 에서 빼고 model 의 fresh init 유지.
      예: target_bias_alpha 를 학습된 -0.014 대신 새 init 0.1 로 시작.
      strict=True 여도 reset_keys 가 있으면 자동으로 strict=False 로 강등 (해당
      key 가 missing 으로 잡혀서 RuntimeError 나는 걸 방지).
    또한 optimizer state 의 해당 파라미터 momentum/variance 도 리셋해야 함 —
      안 그러면 0.1 init 에 학습된 -0.014 의 momentum 이 즉시 적용돼 다시 음수로
      끌려감. reset_keys 면 optimizer state 전체를 fresh start.

    strict=False (partial transfer):
      - shape-mismatch key 는 명시적으로 skip (PyTorch strict=False 만으론
        shape 가 다르면 RuntimeError 가 남 → 사전에 필터링 후 load).
        예: TARGET_PAIR_FEAT_DIM 4→6 변경 시 target_bias_mlp.0.weight (32,4)
        →(32,6) mismatch — 해당 layer fresh init 유지.
      - model.load_state_dict(strict=False) — missing/unexpected key 로그 출력.
      - optimizer state 가 새 파라미터와 mismatch 시 fresh start (warning 만).
      - Step transition (예: target_head → target_q/target_k) 시 encoder
        + amount_pair_head + critic 만 transfer 하고 새 head 는 init 그대로.
    """
    if not os.path.exists(path):
        return None

    ckpt = torch.load(path, map_location=device)
    state_to_load = ckpt["model"]

    # reset_keys: 학습된 값 무시하고 fresh init 유지할 파라미터.
    reset_applied = []
    if reset_keys:
        for k in reset_keys:
            if k in state_to_load:
                state_to_load.pop(k)
                reset_applied.append(k)
        if reset_applied:
            print(f"[reset_keys] ckpt 에서 제거 → fresh init 유지: {reset_applied}")
            if strict:
                print("[reset_keys] strict=True → False 로 자동 강등 (missing key 허용)")
                strict = False

    if not strict:
        # shape-mismatch key 사전 필터링.
        # PyTorch load_state_dict(strict=False) 는 missing/unexpected 만 허용 —
        # shape 가 다른 key 는 strict 무관하게 RuntimeError 를 던짐.
        own_state = model.state_dict()
        skipped_shape = []
        filtered = {}
        for k, v in state_to_load.items():
            if k in own_state and own_state[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(own_state[k].shape)))
                continue
            filtered[k] = v
        if skipped_shape:
            print(f"[partial transfer] shape-mismatch skipped ({len(skipped_shape)}):")
            for k, src, dst in skipped_shape:
                print(f"    {k}: ckpt{src} → model{dst}  (fresh init 유지)")
        state_to_load = filtered

    res  = model.load_state_dict(state_to_load, strict=strict)
    if not strict and (res.missing_keys or res.unexpected_keys):
        print(f"[partial transfer] missing keys ({len(res.missing_keys)}): {res.missing_keys}")
        print(f"[partial transfer] unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys}")
    if reset_applied:
        # reset_keys 적용했으면 optimizer state 도 fresh — 안 그러면 학습된 값의
        # momentum 이 새 init 을 즉시 끌어내림.
        print(f"[reset_keys] optimizer state 도 fresh start (momentum 잔재 제거)")
    else:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as e:
            if strict:
                raise
            print(f"[partial transfer] optimizer state mismatch — fresh start: {e}")

    print(f"체크포인트 로드: gen={ckpt['generation']}, steps={ckpt['total_steps']:,}")
    return ckpt["generation"], ckpt["total_steps"], ckpt["league_agents"]
