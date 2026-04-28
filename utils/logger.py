import csv
import os
import numbers
from datetime import datetime

import yaml


def _is_num(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


# ships_surplus_bins 로드 (ships_bin_rate_k 필드명 생성용)
_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(_cfg_path) as _f:
    _CFG = yaml.safe_load(_f)
_SHIPS_BINS     = tuple(_CFG["model"].get("ships_surplus_bins", [0.0, 0.33, 0.66, 1.0]))
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
            "ent_action", "ent_target",
            "kl_action", "kl_target",
            "cf_action", "cf_target",
            "mean_dense_rew",
            # cap_bonus 분리 (gain / loss). 정식 컬럼.
            "mean_neutral_capture_bonus",      # 중립 → 내것 점령 (양수 mean, early_boost 적용).
            "mean_own_planet_loss_penalty",    # 내것 → 잃음 (음수 mean, phase 무관).
            # Deprecated: mean_neutral_capture_bonus + mean_own_planet_loss_penalty.
            # 옛 로그와의 추세 비교용으로만 유지. 새 분석은 분리된 두 컬럼을 본다.
            "mean_cap_bonus",
            "mean_terminal_rew",
            # Sprint 2: 발사 시 자원 보존 페널티 (음수 mean). all_in_penalty=0 이면 0.
            "mean_all_in_penalty",
            # Phase B: per-launch cost penalty (continuous, req/src>0.5). 0 이면 비활성.
            #   target cost feature ↔ reward 다리. alpha 가 자랄 신호 만든다.
            "mean_launch_cost_penalty",
            # 다중 source over-send 페널티 (음수 mean). over_send_penalty=0 이면 0.
            "mean_over_send_penalty",
            "mean_attempts", "mean_launched", "launch_rate",
            # Step 4 측정: active source(자기 행성+ships>0) 중 valid target 없어
            # skip 으로 빠진 비율. mask 가 너무 조이면 행동공간 막힘 신호.
            "self_fallback_active_rate",
            # Phase A 진단 A: src.ships threshold 별 self_fallback rate.
            # 작은 src(1~2)가 만든 false alarm 인지 vs 큰 src 도 진짜 막히는지 구분.
            "self_fallback_active_rate_ge1",
            "self_fallback_active_rate_ge5",
            "self_fallback_active_rate_ge10",
            "self_fallback_active_rate_ge20",
            # Phase A 진단 A': mask first-failure 분해 — 5 게이트 × 4 threshold (20 cols).
            # 분모: active_by_thresh (acting_player 소유 + ships≥thresh src 수).
            # 가설별 신호:
            #   capacity_sufficient_ge20 ↑↑ : 진짜 mask 위기 (큰 src 가 비싼 target 에 막혀 idle).
            #   target_owner_allowed     ↑  : 게임 룰 (acting_player 소유 행성 비율 — 후반 정상 ↑).
            #   attack_still_needed      ↑  : Guard A 가 in-flight 중복 발사 차단 (정상; support 는 통과).
            #   flight_path_clear        ↑  : 사선/태양 차폐 — geometric (mask issue 아님).
            #   projected_arrival_state         : 차단 안 함 (proj 계산만) — 항상 0 기대.
            *(f"mask_block_{g}_ge{t}"
              for g in ("target_owner_allowed", "flight_path_clear", "projected_arrival_state",
                        "attack_still_needed", "capacity_sufficient")
              for t in (1, 5, 10, 20)),
            # Phase A 진단 B: per-launch req/src.ships 분포 (target 비용 측면).
            # send_required_ratio≈1.17 / send_fraction≈0.80 → req/src ≈ 0.68 가설.
            "req_over_src_launched_mean",
            "req_over_src_launched_p50",
            "req_over_src_launched_p75",
            "req_over_src_launched_p90",
            "req_over_src_launched_p95",
            # Phase A 진단 B': production-axis (req/prod, req/(prod×ETA)).
            # req_over_src 가 p90=1.0 saturated 일 때 "회복 가능 vs 자살" 가르는 보조 신호.
            #   req/prod    : 경제 비용 (몇 턴어치 생산을 태우나)
            #   req/prodETA : 도착 전까지 회복 가능성
            "req_over_prod_launched_mean",
            "req_over_prod_launched_p50",
            "req_over_prod_launched_p75",
            "req_over_prod_launched_p90",
            "req_over_prod_launched_p95",
            "req_over_prod_eta_launched_mean",
            "req_over_prod_eta_launched_p50",
            "req_over_prod_eta_launched_p75",
            "req_over_prod_eta_launched_p90",
            "req_over_prod_eta_launched_p95",
            # Phase A 진단 C: per-launch eta_advantage_norm 분포 (race signal).
            # (enemy_min_eta - my_eta) / 30 ∈ [-1, 1]. 양수 = 적보다 빠름 (race 우위).
            # 가설: 정책이 race-favorable target 선호 시 launched p50/p75 → 양수 shift.
            "eta_advantage_launched_mean",
            "eta_advantage_launched_p50",
            "eta_advantage_launched_p75",
            "eta_advantage_launched_p90",
            "eta_advantage_launched_p95",
            # Phase A 진단 C: additive bias channel scalar α (model parameter, learned).
            # 0 → off, 커지면 cost feature bias 가 정책에 영향 시작.
            "target_bias_alpha", "amount_bin_alpha",
            # Phase A 진단 D: send_frac × ships_bin 매트릭스 (multiplier 모드 동작 확인).
            *(f"send_frac_bin{k}_mean" for k in range(_NUM_SHIPS_BINS)),
            # bin0/bin{K-1} (양 극단) 만 p90 추적.
            *(f"send_frac_bin{k}_p90" for k in sorted({0, _NUM_SHIPS_BINS - 1})),
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
            # ── 초반 중립 확장 품질 (turn_norm < 0.25 AND target.owner == -1) ─
            # "가까운 저비용 중립을 빠르게 먹는가" 검증. 모두 launched 기준.
            #   eta_mean             : 평균 eta_turns (작을수록 가까움)
            #   req_over_src_mean    : 평균 cost (낮을수록 저비용)
            #   nearest_rank_mean    : src 에서 launchable 중립 ETA-sort 시 평균 순위
            #                          (1=항상 가장 가까운 중립, ≥3=비효율 확장)
            #   eta_advantage_mean   : 평균 race signal (>0=내가 더 빠름)
            "early_neutral_eta_mean", "early_neutral_req_over_src_mean",
            "early_neutral_nearest_rank_mean", "early_neutral_eta_advantage_mean",
            # Phase C support (자기 행성 지원 launch). neutral/enemy capture 와 분리.
            #   support_launches_per_step : per-step 평균 support launch 수
            #   support_ships_to_send_mean: support launch 당 평균 함선 수
            "support_launches_per_step", "support_ships_to_send_mean",
            # ── ships 분포 실측 (Categorical surplus-fraction head) ─────────
            # chosen_surplus_frac_*: capacity-short 제외한 bin 선택 분포
            # send_fraction_of_src_*: floor 포함 실제 자원 소진률 (ships/src.ships)
            "chosen_surplus_frac_mean", "chosen_surplus_frac_std",
            "send_fraction_of_src_mean", "send_fraction_of_src_std",
            "ships_to_send_mean", "required_ships_mean",
            "send_required_ratio_mean", "under_invested_rate",
            "over_send_excess_per_launch", "over_send_target_rate",
            # target-type 분리 (neutral prod 없음 / enemy prod 회복 → waste 차이)
            "send_required_ratio_mean_neutral", "send_required_ratio_mean_enemy",
            "under_invested_rate_neutral", "under_invested_rate_enemy",
            "ships_to_send_mean_neutral", "ships_to_send_mean_enemy",
            "required_ships_mean_neutral", "required_ships_mean_enemy",
            # 연계 공격 (단발 실패 vs 연속 압박 구분)
            "repeat_target_rate",
            "launch_to_cap_rate_neutral", "launch_to_cap_rate_enemy",
            # multiplicity: capture 1회당 평균 linked launch 수 (협력 공격 측정, ≥1).
            "linked_launches_per_capture_neutral", "linked_launches_per_capture_enemy",
            # ── 1차 진단 metric (단발 점령 + 유지 + 자원 보존) ────────────
            "single_shot_capture_rate",
            "capture_hold_k_rate",
            "post_capture_reloss_rate_k",
            "all_in_launch_rate",
            "remaining_ships_after_launch_mean",
            "distinct_targets_per_turn",
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
        ent_action  = kwargs.get("ent_action", "")
        ent_target  = kwargs.get("ent_target", "")
        mean_dense  = kwargs.get("mean_dense_rew", "")
        mean_cap    = kwargs.get("mean_cap_bonus", "")
        mean_term   = kwargs.get("mean_terminal_rew", "")
        mean_aip    = kwargs.get("mean_all_in_penalty", "")
        mean_osp    = kwargs.get("mean_over_send_penalty", "")
        mean_attempts = kwargs.get("mean_attempts", "")
        mean_launched = kwargs.get("mean_launched", "")
        launch_rate   = kwargs.get("launch_rate", "")
        mean_filtered_invalid = kwargs.get("mean_filtered_invalid_target", "")
        mean_filtered_zero    = kwargs.get("mean_filtered_zero_ships", "")
        mean_filtered_sun     = kwargs.get("mean_filtered_sun", "")
        mean_filtered_path    = kwargs.get("mean_filtered_path", "")
        kl_a = kwargs.get("kl_action", "")
        kl_t = kwargs.get("kl_target", "")
        cf_a = kwargs.get("cf_action", "")
        cf_t = kwargs.get("cf_target", "")
        kl_str  = (f" | kl={approx_kl:.4f} | cf={clip_frac:.3f}"
                   f" | ep={epochs_done}") if _is_num(approx_kl) else ""
        ent_str = (f" | ea={ent_action:.2f} | et={ent_target:.2f}"
                   ) if _is_num(ent_action) else ""
        # aip = all-in penalty (Sprint 2). osp = over-send penalty (다중 source 협조).
        # 둘 다 0 이면 비활성. 음수로 찍힘.
        aip_str = f" | aip={mean_aip:+.4f}" if _is_num(mean_aip) else ""
        osp_str = f" | osp={mean_osp:+.4f}" if _is_num(mean_osp) else ""
        rew_str = (f" | dr={mean_dense:+.4f} | cb={mean_cap:+.4f} | tr={mean_term:+.4f}"
                   f"{aip_str}{osp_str}"
                   ) if _is_num(mean_dense) else ""
        head_str = (f" | klh=[{kl_a:.3f}/{kl_t:.3f}]"
                    f" | cfh=[{cf_a:.2f}/{cf_t:.2f}]"
                    ) if _is_num(kl_a) else ""
        sfb_rate = kwargs.get("self_fallback_active_rate", "")
        sfb_str  = f"/sfb={sfb_rate:.0%}" if _is_num(sfb_rate) else ""
        decode_str = (
            f" | dec=[a={mean_attempts:.2f}/l={mean_launched:.2f}/r={launch_rate:.0%}"
            f"/inv={mean_filtered_invalid:.2f}/z={mean_filtered_zero:.2f}/sun={mean_filtered_sun:.2f}"
            f"/path={mean_filtered_path:.2f}{sfb_str}]"
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

        cm_mean   = kwargs.get("chosen_surplus_frac_mean", "")
        cm_std    = kwargs.get("chosen_surplus_frac_std", "")
        sf_mean   = kwargs.get("send_fraction_of_src_mean", "")
        sf_std    = kwargs.get("send_fraction_of_src_std", "")
        sts_mean  = kwargs.get("ships_to_send_mean", "")
        req_mean  = kwargs.get("required_ships_mean", "")
        srr_mean  = kwargs.get("send_required_ratio_mean", "")
        under     = kwargs.get("under_invested_rate", "")
        os_excess = kwargs.get("over_send_excess_per_launch", "")
        if _is_num(cm_mean):
            # bin 히스토그램: launched 대비 각 bin의 선택 비율
            bin_rates = []
            for k in range(_NUM_SHIPS_BINS):
                r = kwargs.get(f"ships_bin_rate_{k}", "")
                if _is_num(r):
                    bin_rates.append(f"{_SHIPS_BINS[k]:.2f}={r:.0%}")
            bin_str = " | bins=[" + "/".join(bin_rates) + "]" if bin_rates else ""
            os_str = f" | os={os_excess:.1f}" if _is_num(os_excess) else ""
            # surf: bin 값 (capacity-short 제외) — 정책의 bin 선택 분포
            # send%: floor 포함 실제 ships_sent/src.ships — 자원 소진률
            sf_str = f" | send%={sf_mean:.0%}±{sf_std:.0%}" if _is_num(sf_mean) else ""
            line5 = (
                f"ships=[surf={cm_mean:.2f}±{cm_std:.2f}{sf_str}"
                f" | send={sts_mean:.1f}/req={req_mean:.1f}"
                f" | s/r={srr_mean:.2f}/under={under:.0%}{os_str}]"
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

        # Phase A 진단 (post-Phase-A 병목 분석):
        #   sfb_ge*  : src.ships 임계 별 self_fallback (작은 src false alarm vs 진짜 mask 위기)
        #   r/s p90  : per-launch req/src.ships (target 비용 분포 — 비싼 target 한계 추적)
        #   α        : Phase A bias α (target/amount). 0 → off, 학습 진행 시 |α| 증가.
        #   b{0/K-1} : send_frac bin0(just-cap) / bin top-mult (양 극단 p90)
        sfb_ge1  = kwargs.get("self_fallback_active_rate_ge1",  "")
        sfb_ge5  = kwargs.get("self_fallback_active_rate_ge5",  "")
        sfb_ge10 = kwargs.get("self_fallback_active_rate_ge10", "")
        sfb_ge20 = kwargs.get("self_fallback_active_rate_ge20", "")
        ros_mean = kwargs.get("req_over_src_launched_mean", "")
        ros_p50  = kwargs.get("req_over_src_launched_p50",  "")
        ros_p90  = kwargs.get("req_over_src_launched_p90",  "")
        ros_p95  = kwargs.get("req_over_src_launched_p95",  "")
        a_t      = kwargs.get("target_bias_alpha", "")
        a_a      = kwargs.get("amount_bin_alpha", "")
        bin0_p90 = kwargs.get("send_frac_bin0_p90", "")
        bin_top_p90 = kwargs.get(f"send_frac_bin{_NUM_SHIPS_BINS - 1}_p90", "")
        if _is_num(sfb_ge1):
            line9 = (
                f"phaseA=[sfb≥1={sfb_ge1:.0%}/≥5={sfb_ge5:.0%}/≥10={sfb_ge10:.0%}/≥20={sfb_ge20:.0%}"
                f" | r/s μ={ros_mean:.2f}/p50={ros_p50:.2f}/p90={ros_p90:.2f}/p95={ros_p95:.2f}"
                f" | α_t={a_t:+.3f}/α_a={a_a:+.3f}"
                f" | b0_p90={bin0_p90:.2f}/b{_NUM_SHIPS_BINS - 1}_p90={bin_top_p90:.2f}]"
            )
            print(line9)

        # Phase A' 진단 (production-axis): req_over_src 포화 상태에서
        # "경제적으로 비싼가 / 도착 전 회복 가능한가" 분리 추적.
        #   r/p     : req/prod (몇 턴어치 생산)
        #   r/pE    : req/(prod×ETA) (도착 전 회복 가능성)
        #   둘 다 p90 으로 보면 정책이 한계 경제 비용에 몰리는지 명확.
        rop_mean   = kwargs.get("req_over_prod_launched_mean", "")
        rop_p50    = kwargs.get("req_over_prod_launched_p50",  "")
        rop_p90    = kwargs.get("req_over_prod_launched_p90",  "")
        rop_p95    = kwargs.get("req_over_prod_launched_p95",  "")
        rope_mean  = kwargs.get("req_over_prod_eta_launched_mean", "")
        rope_p50   = kwargs.get("req_over_prod_eta_launched_p50",  "")
        rope_p90   = kwargs.get("req_over_prod_eta_launched_p90",  "")
        rope_p95   = kwargs.get("req_over_prod_eta_launched_p95",  "")
        if _is_num(rop_mean):
            line10 = (
                f"phaseA'=[r/p μ={rop_mean:.2f}/p50={rop_p50:.2f}/p90={rop_p90:.2f}/p95={rop_p95:.2f}"
                f" | r/pE μ={rope_mean:.2f}/p50={rope_p50:.2f}/p90={rope_p90:.2f}/p95={rope_p95:.2f}]"
            )
            print(line10)

        # Phase A 진단 A' (mask first-failure ge20): 큰 src(≥20 ships) 기준 게이트별 차단 rate.
        # 작은 src threshold 는 노이즈 많아 ge20 만 콘솔 — 다른 threshold 는 CSV 에 있음.
        # cap ↑↑ → 진짜 mask 위기, own ↑ → 게임 룰, atk ↑ → Guard A 적정 (support 통과),
        # path ↑ → geometric (mask issue 아님).
        mb_own_ge20  = kwargs.get("mask_block_target_owner_allowed_ge20", "")
        mb_atk_ge20  = kwargs.get("mask_block_attack_still_needed_ge20",  "")
        mb_path_ge20 = kwargs.get("mask_block_flight_path_clear_ge20",    "")
        mb_cap_ge20  = kwargs.get("mask_block_capacity_sufficient_ge20",  "")
        if _is_num(mb_own_ge20):
            line11 = (
                f"mask_ge20=[own={mb_own_ge20:.1f}/atk={mb_atk_ge20:.1f}"
                f"/path={mb_path_ge20:.1f}/cap={mb_cap_ge20:.1f}]"
            )
            print(line11)

        # Phase A 진단 C (eta_advantage launched): race signal 의 per-launch 분포.
        # 양수 dominant → 정책이 race-favorable target 선호 (초반 중립 race 우위).
        # 음수 dominant → 늦은 race 만 보냄 (보내봤자 경합 손실 — 위험 신호).
        eta_mean = kwargs.get("eta_advantage_launched_mean", "")
        eta_p50  = kwargs.get("eta_advantage_launched_p50",  "")
        eta_p75  = kwargs.get("eta_advantage_launched_p75",  "")
        eta_p90  = kwargs.get("eta_advantage_launched_p90",  "")
        if _is_num(eta_mean):
            line12 = (
                f"eta_adv=[μ={eta_mean:+.2f}/p50={eta_p50:+.2f}"
                f"/p75={eta_p75:+.2f}/p90={eta_p90:+.2f}]"
            )
            print(line12)
