import csv
import os
from datetime import datetime


class TrainingLogger:
    """학습 로그를 CSV + 콘솔에 기록."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path   = os.path.join(log_dir, f"train_{timestamp}.csv")
        self.fields = [
            "generation", "total_steps", "match_type",
            "policy_loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_frac",
            "win_rate", "league_size",
        ]
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

        print(f"Logger: {self.path}")

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.fields}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)

        win_rate = kwargs.get("win_rate", "")
        win_str  = f" | win_rate={win_rate:.2%}" if isinstance(win_rate, float) else ""
        approx_kl = kwargs.get("approx_kl", "")
        clip_frac  = kwargs.get("clip_frac", "")
        kl_str     = f" | kl={approx_kl:.4f} | cf={clip_frac:.3f}" if isinstance(approx_kl, float) else ""
        print(
            f"Gen {kwargs.get('generation', '?'):04d} | "
            f"steps={kwargs.get('total_steps', 0):,} | "
            f"match={kwargs.get('match_type', '?')} | "
            f"p_loss={kwargs.get('policy_loss', 0):.4f} | "
            f"v_loss={kwargs.get('value_loss', 0):.4f} | "
            f"e_loss={kwargs.get('entropy_loss', 0):.4f}"
            f"{kl_str}{win_str}"
        )
