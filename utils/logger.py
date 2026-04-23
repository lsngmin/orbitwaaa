import csv
import os
import numbers
from datetime import datetime


def _is_num(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


class TrainingLogger:
    """학습 로그를 CSV + 콘솔에 기록."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path   = os.path.join(log_dir, f"train_{timestamp}.csv")
        self.fields = [
            "generation", "total_steps", "match_type",
            "policy_loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_frac", "epochs_done",
            "ent_launch", "ent_ships", "ent_target",
            "mean_dense_rew", "mean_cap_bonus", "mean_terminal_rew",
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
        win_str  = f" | win_rate={win_rate:.2%}" if _is_num(win_rate) else ""
        approx_kl   = kwargs.get("approx_kl", "")
        clip_frac   = kwargs.get("clip_frac", "")
        epochs_done = kwargs.get("epochs_done", "")
        ent_launch  = kwargs.get("ent_launch", "")
        ent_ships   = kwargs.get("ent_ships", "")
        ent_target  = kwargs.get("ent_target", "")
        mean_dense  = kwargs.get("mean_dense_rew", "")
        mean_cap    = kwargs.get("mean_cap_bonus", "")
        mean_term   = kwargs.get("mean_terminal_rew", "")
        kl_str  = (f" | kl={approx_kl:.4f} | cf={clip_frac:.3f}"
                   f" | ep={epochs_done}") if _is_num(approx_kl) else ""
        ent_str = (f" | el={ent_launch:.2f} | es={ent_ships:.2f} | et={ent_target:.2f}"
                   ) if _is_num(ent_launch) else ""
        rew_str = (f" | dr={mean_dense:+.4f} | cb={mean_cap:+.4f} | tr={mean_term:+.4f}"
                   ) if _is_num(mean_dense) else ""
        print(
            f"Gen {kwargs.get('generation', '?'):04d} | "
            f"steps={kwargs.get('total_steps', 0):,} | "
            f"match={kwargs.get('match_type', '?')} | "
            f"p_loss={kwargs.get('policy_loss', 0):.4f} | "
            f"v_loss={kwargs.get('value_loss', 0):.4f} | "
            f"e_loss={kwargs.get('entropy_loss', 0):.4f}"
            f"{kl_str}{ent_str}{rew_str}{win_str}"
        )
