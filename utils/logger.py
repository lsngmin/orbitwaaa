import csv
import os
import numbers
from datetime import datetime

import yaml


def _is_num(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


# ships_multiplier_bins 로드 (ships_bin_rate_k 필드명 생성용)
_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(_cfg_path) as _f:
    _CFG = yaml.safe_load(_f)
_SHIPS_BINS     = tuple(_CFG["model"].get("ships_multiplier_bins", [1.10, 1.30, 1.60, 2.00]))
_NUM_SHIPS_BINS = len(_SHIPS_BINS)


class TrainingLogger:
    """학습 로그를 CSV + 콘솔에 기록."""

    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path   = os.path.join(log_dir, f"train_{timestamp}.csv")
        self.fields = [
            "timestamp",
            "generation", "total_steps", "match_type",
            "policy_loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_frac", "epochs_done",
            "ent_launch", "ent_ships", "ent_target",
            "kl_launch", "kl_ships", "kl_target",
            "cf_launch", "cf_ships", "cf_target",
            "mean_dense_rew", "mean_cap_bonus", "mean_terminal_rew",
            "mean_attempts", "mean_launched", "launch_rate",
            "mean_filtered_invalid_target", "mean_filtered_zero_ships", "mean_filtered_sun",
            "mean_filtered_path",
            "mean_out", "mean_sun_crash",
            "mean_target_hit_exclusive", "mean_target_hit_ambiguous",
            "mean_hit_other_exclusive", "mean_hit_other_ambiguous",
            "mean_captured_exclusive", "mean_captured_ambiguous",
            "mean_launched_high_prod",
            "mean_captured_neutral", "mean_captured_enemy",
            "mean_early_home_expand",
            "noop_rate", "high_prod_target_rate",
            "neutral_capture_rate", "enemy_capture_rate",
            "early_home_expand_per_episode",
            # ── 타겟 분포 / 초반 확장 계측 ──────────────────────────────────
            "mean_target_neutral", "mean_target_enemy",
            "mean_early_neutral_attempts", "mean_early_enemy_attempts",
            "mean_early_neutral_captured",
            "mean_early_launch_neutral_captured",
            "target_neutral_rate", "target_enemy_rate",
            "early_neutral_attempts_per_episode", "early_enemy_attempts_per_episode",
            "early_neutral_captured_per_episode",
            "early_launch_neutral_captured_per_episode",
            "early_neutral_launch_to_cap_rate",
            # ── ships 분포 실측 (commit 2: Categorical multiplier head) ─────
            "chosen_multiplier_mean", "chosen_multiplier_std",
            "ships_to_send_mean", "required_ships_mean",
            "send_required_ratio_mean", "under_invested_rate",
            # target-type 분리 (neutral prod 없음 / enemy prod 회복 → waste 차이)
            "send_required_ratio_mean_neutral", "send_required_ratio_mean_enemy",
            "under_invested_rate_neutral", "under_invested_rate_enemy",
            "ships_to_send_mean_neutral", "ships_to_send_mean_enemy",
            "required_ships_mean_neutral", "required_ships_mean_enemy",
            # 연계 공격 (단발 실패 vs 연속 압박 구분)
            "repeat_target_rate",
            "launch_to_cap_rate_neutral", "launch_to_cap_rate_enemy",
            # eval 전용: 승리/패배 게임 분리 (under-invest ↔ 패배 상관관계)
            "eval_under_win", "eval_under_loss",
            "eval_sr_win", "eval_sr_loss",
            "eval_under_enemy_win", "eval_under_enemy_loss",
            # ships_bin 히스토그램 (launched 대비 비율)
            *(f"ships_bin_rate_{k}" for k in range(_NUM_SHIPS_BINS)),
            "mean_unknown_removal",
            "win_rate", "league_size",
            # wall-time 계측
            "eval_wall_s", "gen_wall_s",
        ]
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writeheader()

        print(f"Logger: {self.path}")

    def log(self, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {k: kwargs.get(k, "") for k in self.fields}
        row["timestamp"] = now
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
        mean_attempts = kwargs.get("mean_attempts", "")
        mean_launched = kwargs.get("mean_launched", "")
        launch_rate   = kwargs.get("launch_rate", "")
        mean_filtered_invalid = kwargs.get("mean_filtered_invalid_target", "")
        mean_filtered_zero    = kwargs.get("mean_filtered_zero_ships", "")
        mean_filtered_sun     = kwargs.get("mean_filtered_sun", "")
        mean_filtered_path    = kwargs.get("mean_filtered_path", "")
        kl_l = kwargs.get("kl_launch", "")
        kl_s = kwargs.get("kl_ships", "")
        kl_t = kwargs.get("kl_target", "")
        cf_l = kwargs.get("cf_launch", "")
        cf_s = kwargs.get("cf_ships", "")
        cf_t = kwargs.get("cf_target", "")
        kl_str  = (f" | kl={approx_kl:.4f} | cf={clip_frac:.3f}"
                   f" | ep={epochs_done}") if _is_num(approx_kl) else ""
        ent_str = (f" | el={ent_launch:.2f} | es={ent_ships:.2f} | et={ent_target:.2f}"
                   ) if _is_num(ent_launch) else ""
        rew_str = (f" | dr={mean_dense:+.4f} | cb={mean_cap:+.4f} | tr={mean_term:+.4f}"
                   ) if _is_num(mean_dense) else ""
        head_str = (f" | klh=[{kl_l:.3f}/{kl_s:.3f}/{kl_t:.3f}]"
                    f" | cfh=[{cf_l:.2f}/{cf_s:.2f}/{cf_t:.2f}]"
                    ) if _is_num(kl_l) else ""
        decode_str = (
            f" | dec=[a={mean_attempts:.2f}/l={mean_launched:.2f}/r={launch_rate:.0%}"
            f"/inv={mean_filtered_invalid:.2f}/z={mean_filtered_zero:.2f}/sun={mean_filtered_sun:.2f}"
            f"/path={mean_filtered_path:.2f}]"
        ) if _is_num(mean_attempts) else ""
        mean_out        = kwargs.get("mean_out", "")
        mean_sun_crash  = kwargs.get("mean_sun_crash", "")
        mean_th_ex      = kwargs.get("mean_target_hit_exclusive", "")
        mean_th_am      = kwargs.get("mean_target_hit_ambiguous", "")
        mean_ho_ex      = kwargs.get("mean_hit_other_exclusive", "")
        mean_ho_am      = kwargs.get("mean_hit_other_ambiguous", "")
        mean_cap_ex     = kwargs.get("mean_captured_exclusive", "")
        mean_cap_am     = kwargs.get("mean_captured_ambiguous", "")
        noop_rate        = kwargs.get("noop_rate", "")
        high_prod_rate   = kwargs.get("high_prod_target_rate", "")
        neutral_cap_rate = kwargs.get("neutral_capture_rate", "")
        enemy_cap_rate   = kwargs.get("enemy_capture_rate", "")
        home20_per_ep    = kwargs.get("early_home_expand_per_episode", "")
        hit_str = (
            f" | hit=[out={mean_out:.2f}/sun={mean_sun_crash:.2f}"
            f"/th={mean_th_ex:.2f}+{mean_th_am:.2f}"
            f"/ho={mean_ho_ex:.2f}+{mean_ho_am:.2f}"
            f"/cap={mean_cap_ex:.2f}+{mean_cap_am:.2f}]"
        ) if _is_num(mean_out) else ""
        line1 = (
            f"{now} | "
            f"Gen {kwargs.get('generation', '?'):04d} | "
            f"steps={kwargs.get('total_steps', 0):,} | "
            f"match={kwargs.get('match_type', '?')} | "
            f"p_loss={kwargs.get('policy_loss', 0):.4f} | "
            f"v_loss={kwargs.get('value_loss', 0):.4f} | "
            f"e_loss={kwargs.get('entropy_loss', 0):.4f}"
            f"{kl_str}{ent_str}{rew_str}{head_str}{win_str}"
        )
        print(line1)

        if _is_num(mean_attempts) and _is_num(mean_out):
            launched = mean_launched if mean_launched > 0 else 0.0
            attempts = mean_attempts if mean_attempts > 0 else 0.0
            th_total = mean_th_ex + mean_th_am
            ho_total = mean_ho_ex + mean_ho_am
            cap_total = mean_cap_ex + mean_cap_am
            th_rate = (th_total / launched) if launched > 0 else 0.0
            ho_rate = (ho_total / launched) if launched > 0 else 0.0
            out_rate = (mean_out / launched) if launched > 0 else 0.0
            sun_rate = (mean_sun_crash / launched) if launched > 0 else 0.0
            cap_rate = (cap_total / launched) if launched > 0 else 0.0
            path_rate = (mean_filtered_path / attempts) if attempts > 0 else 0.0

            line2 = (
                f"{decode_str}{hit_str}"
                f" | rate=[th={th_rate:.0%}/ho={ho_rate:.0%}/out={out_rate:.0%}"
                f"/sun={sun_rate:.0%}/cap={cap_rate:.0%}/path={path_rate:.0%}]"
            )
            print(line2.lstrip())

        if _is_num(noop_rate):
            line3 = (
                f"strat=[ncap={neutral_cap_rate:.0%}/ecap={enemy_cap_rate:.0%}"
                f"/home20={home20_per_ep:.2f}/noop={noop_rate:.0%}"
                f"/hprod={high_prod_rate:.0%}]"
            )
            print(line3)

        tgt_n_rate    = kwargs.get("target_neutral_rate", "")
        tgt_e_rate    = kwargs.get("target_enemy_rate", "")
        early_n_att   = kwargs.get("early_neutral_attempts_per_episode", "")
        early_e_att   = kwargs.get("early_enemy_attempts_per_episode", "")
        early_n_cap   = kwargs.get("early_neutral_captured_per_episode", "")
        early_lnc     = kwargs.get("early_launch_neutral_captured_per_episode", "")
        early_l2c     = kwargs.get("early_neutral_launch_to_cap_rate", "")
        if _is_num(tgt_n_rate):
            line4 = (
                f"aim=[tgt_n={tgt_n_rate:.0%}/tgt_e={tgt_e_rate:.0%}"
                f" | early20: n_att={early_n_att:.2f}/e_att={early_e_att:.2f}"
                f"/n_cap={early_n_cap:.2f}/home_cap={home20_per_ep:.2f}"
                f" | launch20→cap={early_lnc:.2f}(={early_l2c:.0%})]"
            )
            print(line4)

        cm_mean   = kwargs.get("chosen_multiplier_mean", "")
        cm_std    = kwargs.get("chosen_multiplier_std", "")
        sts_mean  = kwargs.get("ships_to_send_mean", "")
        req_mean  = kwargs.get("required_ships_mean", "")
        srr_mean  = kwargs.get("send_required_ratio_mean", "")
        under     = kwargs.get("under_invested_rate", "")
        if _is_num(cm_mean):
            # bin 히스토그램: launched 대비 각 bin의 선택 비율
            bin_rates = []
            for k in range(_NUM_SHIPS_BINS):
                r = kwargs.get(f"ships_bin_rate_{k}", "")
                if _is_num(r):
                    bin_rates.append(f"{_SHIPS_BINS[k]:.2f}={r:.0%}")
            bin_str = " | bins=[" + "/".join(bin_rates) + "]" if bin_rates else ""
            line5 = (
                f"ships=[mult={cm_mean:.2f}±{cm_std:.2f}"
                f" | send={sts_mean:.1f}/req={req_mean:.1f}"
                f" | s/r={srr_mean:.2f}/under={under:.0%}]"
                f"{bin_str}"
            )
            print(line5)

        # target-type 분리: neutral(prod 없음)은 under-invest 해도 한 번 손실만,
        # enemy(prod 회복)는 재생산으로 장기 waste → under_enemy가 패배 상관 큼.
        srr_n  = kwargs.get("send_required_ratio_mean_neutral", "")
        srr_e  = kwargs.get("send_required_ratio_mean_enemy", "")
        und_n  = kwargs.get("under_invested_rate_neutral", "")
        und_e  = kwargs.get("under_invested_rate_enemy", "")
        sts_n  = kwargs.get("ships_to_send_mean_neutral", "")
        sts_e  = kwargs.get("ships_to_send_mean_enemy", "")
        if _is_num(srr_n) or _is_num(srr_e):
            line6 = (
                f"by_tgt=[neu: s/r={srr_n:.2f}/under={und_n:.0%}/send={sts_n:.1f}"
                f" | enm: s/r={srr_e:.2f}/under={und_e:.0%}/send={sts_e:.1f}]"
            )
            print(line6)

        # 연계 공격 지표: 단발 실패인지 계획된 연속 압박인지 구분.
        # repeat_target_rate: 같은 target에 K턴 내 재발사 비율
        # launch_to_cap_rate_{neu,enm}: launch 후 K턴 내 target 점령된 비율
        # (under 높지만 launch_to_cap_rate도 높으면 연계 성공 — 건강한 지표)
        rpt_rate  = kwargs.get("repeat_target_rate", "")
        l2c_neu   = kwargs.get("launch_to_cap_rate_neutral", "")
        l2c_enm   = kwargs.get("launch_to_cap_rate_enemy", "")
        if _is_num(rpt_rate):
            line7 = (
                f"combo=[repeat={rpt_rate:.0%}"
                f" | l2c_neu={l2c_neu:.0%}/l2c_enm={l2c_enm:.0%}]"
            )
            print(line7)

        # eval win/loss split: 승리 게임 vs 패배 게임의 under/sr 차이
        # (under-invest 가설이 맞다면 loss > win이어야 함)
        e_uw = kwargs.get("eval_under_win", "")
        e_ul = kwargs.get("eval_under_loss", "")
        e_sw = kwargs.get("eval_sr_win", "")
        e_sl = kwargs.get("eval_sr_loss", "")
        e_uew = kwargs.get("eval_under_enemy_win", "")
        e_uel = kwargs.get("eval_under_enemy_loss", "")
        if _is_num(e_uw):
            line8 = (
                f"eval_split=[win: under={e_uw:.0%}(enm={e_uew:.0%})/sr={e_sw:.2f}"
                f" | loss: under={e_ul:.0%}(enm={e_uel:.0%})/sr={e_sl:.2f}]"
            )
            print(line8)
