"""perf-rollout: 짧은 학습 러닝을 cProfile로 감싸 hot function 측정.

사용법:
    python scripts/profile_train.py                     # n_envs=1, steps=1024
    python scripts/profile_train.py --n-envs 4
    python scripts/profile_train.py --steps 2048 --top 30

출력:
    perf/{timestamp}_n{n_envs}_s{steps}.prof   — pstats 바이너리
    stdout                                       — cumulative top-N

n_envs > 1일 때 cProfile은 메인 프로세스만 잡으므로,
worker 내부까지 보려면 py-spy 사용 권장:
    py-spy record -o perf/pyspy.svg --subprocesses -- \
        python scripts/profile_train.py --n-envs 4 --steps 1024
"""

import argparse
import cProfile
import os
import pstats
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",        type=int, default=1024, help="rollout step 수 (총 timesteps에 거의 직결)")
    parser.add_argument("--n-envs",       type=int, default=1,    help="병렬 worker 수")
    parser.add_argument("--rollout-steps", type=int, default=512, help="rollout 1회 step 수")
    parser.add_argument("--gpu",          type=int, default=0)
    parser.add_argument("--run-dir",      type=str, default="perf/run")
    parser.add_argument("--output-dir",   type=str, default="perf")
    parser.add_argument("--top",          type=int, default=20,   help="stdout에 출력할 cumulative top-N")
    parser.add_argument("--py-spy-hint",  action="store_true",    help="py-spy 명령만 출력하고 종료")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.run_dir,    exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(args.output_dir, f"{timestamp}_n{args.n_envs}_s{args.steps}.prof")

    if args.py_spy_hint:
        print(f"py-spy record -o {args.output_dir}/{timestamp}_n{args.n_envs}.svg "
              f"--subprocesses --rate 100 -- "
              f"python scripts/profile_train.py --n-envs {args.n_envs} "
              f"--steps {args.steps} --rollout-steps {args.rollout_steps}")
        return

    import torch
    import train as train_module

    train_module.DEVICE   = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    train_module.SAVE_DIR = args.run_dir

    print(f"[profile] device={train_module.DEVICE} n_envs={args.n_envs} "
          f"total_steps={args.steps} rollout_steps={args.rollout_steps}")
    print(f"[profile] output={out_path}")

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        train_module.train(
            n_envs=args.n_envs,
            total_timesteps=args.steps,
            eval_interval=10**9,        # 사실상 평가 비활성
            n_games=1,
            rollout_steps=args.rollout_steps,
        )
    finally:
        profiler.disable()
        profiler.dump_stats(out_path)

    print(f"\n[profile] saved → {out_path}")
    print(f"[profile] cumulative top {args.top}:\n")
    stats = pstats.Stats(out_path).sort_stats("cumulative")
    stats.print_stats(args.top)

    print(f"\n[profile] tottime top {args.top}:\n")
    pstats.Stats(out_path).sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
