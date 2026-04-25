#!/usr/bin/env python3
"""
build_submission.py — Kaggle 제출 archive 자동 빌드.

목적
----
.pt 가 나오면 manual 코드 수정 없이 바로 제출 가능한 tar.gz 를 만든다.
차원이 바뀌어도 .pt 와 현재 코드가 *정합* 하면 패키징, 어긋나면 *큰 소리로* 실패.

검증 항목 (조용한 실패 차단)
---------------------------
1) .pt state_dict 에서 차원 추출
   - PLANET_DIM   = planet_embed.weight.shape[1]
   - FLEET_DIM    = fleet_embed.weight.shape[1]
   - EMBED_DIM    = planet_embed.weight.shape[0]
   - HISTORY      = planet_temporal_pos.weight.shape[0]
   - ACTION_DIM   = actor.2.weight.shape[0]
2) 코드 상수와 비교
   - submission_features.PLANET_DIM / FLEET_DIM
   - submission_actor.PLANET_DIM / EMBED_DIM / HISTORY
   - env_wrapper.PLANET_DIM (train/submission parity)
   - config.yaml: embed_dim, history_turns, max_planets, ships_multiplier_bins
   - ACTION_DIM = 1 + NUM_SHIPS_BINS + MAX_PLANETS
3) Smoke load + forward
   - OrbitWarsActor() 인스턴스화 → load_state_dict(strict=False)
   - missing/unexpected 키가 critic.* 외엔 없는지 확인
   - 0-tensor obs 로 forward → 출력 shape == (1, MAX_PLANETS, ACTION_DIM)
4) encode 출력 dim 확인
   - submission_features.encode_planets(...) 의 arr.shape[1] == .pt PLANET_DIM
5) 통과 시에만 bundle 생성

산출물
------
./submissions/sub_<YYYYmmdd_HHMMSS>_<gitsha>.tar.gz  (또는 -o 로 override)

  포함 파일:
    main.py                        # Kaggle entry
    submission_features.py         # encode_planets/fleets
    submission_actor.py            # OrbitWarsActor
    train.py                       # analyze_action_space, decode_action_to_moves
    model.py / env_wrapper.py      # train.py top-level import 의존
    prediction.py                  # encode/aim/sun helpers
    utils/                         # train.py 가 import (logger/checkpoint/hit_tracker)
    config.yaml                    # embed_dim/layers/bins
    weights.pt                     # 입력 .pt 를 이 이름으로 rename

Usage
-----
  python build_submission.py                            # submission.yaml 의 checkpoint 사용
  python build_submission.py path/to/checkpoint.pt      # CLI 인자가 yaml 을 override
  python build_submission.py path/to/checkpoint.pt -o out.tar.gz
  python build_submission.py path/to/checkpoint.pt --skip-smoke    # forward 생략
  python build_submission.py path/to/checkpoint.pt --skip-encode   # encode 검증 생략
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
SUBMISSION_YAML = ROOT / "submission.yaml"

# Kaggle 런타임에 함께 들어가야 하는 파일/디렉토리 — main.py import chain 따라 결정.
BUNDLE_FILES = [
    "main.py",
    "submission_features.py",
    "submission_actor.py",
    "train.py",
    "model.py",
    "env_wrapper.py",
    "prediction.py",
    "config.yaml",
]
BUNDLE_DIRS = [
    "utils",
]


# ── 색깔 출력 (TTY only) ──────────────────────────────────────────────────────

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def info(s):  print(_c("36", "[info] ")  + s)
def ok(s):    print(_c("32", "[ ok ] ")  + s)
def warn(s):  print(_c("33", "[warn] ") + s)
def fail(s):
    print(_c("31", "[FAIL] ") + s, file=sys.stderr)
    sys.exit(1)


# ── 차원 추출 ────────────────────────────────────────────────────────────────

def extract_pt_dims(ckpt: dict) -> dict:
    """state_dict 에서 차원 추출. 키 없으면 즉시 실패."""
    required = [
        "planet_embed.weight",
        "fleet_embed.weight",
        "planet_temporal_pos.weight",
        "actor.0.weight",
        "actor.2.weight",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        fail(
            f".pt 에 필수 키 누락: {missing}\n"
            f"  → 학습/제출 모델 키 layout 이 어긋난 .pt 일 수 있음."
        )

    embed_dim, planet_dim = ckpt["planet_embed.weight"].shape
    _,         fleet_feat = ckpt["fleet_embed.weight"].shape   # 입력 features (idx 제외)
    history,   _          = ckpt["planet_temporal_pos.weight"].shape
    action_dim, _         = ckpt["actor.2.weight"].shape

    return {
        "PLANET_DIM":     int(planet_dim),
        "FLEET_FEAT_DIM": int(fleet_feat),       # fleet_embed 입력 dim
        "FLEET_DIM":      int(fleet_feat) + 2,   # obs 저장 dim (= FEAT + src_idx + dst_idx)
        "EMBED_DIM":      int(embed_dim),
        "HISTORY":        int(history),
        "ACTION_DIM":     int(action_dim),
    }


def extract_code_dims() -> dict:
    """현재 코드 상수 추출. 모듈을 fresh import 해서 단발 평가."""
    # 현재 인터프리터에 캐시된 모듈을 우회 — sys.path 주도 fresh import
    sys.path.insert(0, str(ROOT))
    for mod in ("submission_features", "submission_actor", "env_wrapper"):
        sys.modules.pop(mod, None)

    sf  = importlib.import_module("submission_features")
    sa  = importlib.import_module("submission_actor")
    ew  = importlib.import_module("env_wrapper")

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    bins = tuple(cfg["model"].get("ships_multiplier_bins", []))
    return {
        "sf_PLANET_DIM": int(sf.PLANET_DIM),
        "sf_FLEET_DIM":  int(sf.FLEET_DIM),
        "sf_HISTORY":    int(sf.HISTORY),
        "sf_MAX_PLANETS": int(sf.MAX_PLANETS),
        "sa_PLANET_DIM": int(sa.PLANET_DIM),
        "sa_EMBED_DIM":  int(sa.EMBED_DIM),
        "sa_HISTORY":    int(sa.HISTORY),
        "sa_NUM_SHIPS_BINS": int(sa.NUM_SHIPS_BINS),
        "sa_MAX_PLANETS": int(sa.MAX_PLANETS),
        "sa_ACTION_DIM": int(sa.ACTION_DIM),
        "ew_PLANET_DIM": int(ew.PLANET_DIM),
        "ew_FLEET_DIM":  int(ew.FLEET_DIM),
        "cfg_embed_dim": int(cfg["model"]["embed_dim"]),
        "cfg_history":   int(cfg["env"]["history_turns"]),
        "cfg_max_planets": int(cfg["env"]["max_planets"]),
        "cfg_max_fleets":  int(cfg["env"]["max_fleets"]),
        "cfg_ships_bins_len": len(bins),
        "cfg_module": cfg,
    }


# ── Verify ────────────────────────────────────────────────────────────────────

def verify(pt: dict, code: dict) -> None:
    """모든 일치 검증. 하나라도 어긋나면 fail() 으로 즉시 종료."""
    errs: list[str] = []

    def need(label: str, lhs, rhs):
        if lhs != rhs:
            errs.append(f"  {label}: .pt={lhs} ≠ code={rhs}")

    # 1) submission_features
    need("submission_features.PLANET_DIM", pt["PLANET_DIM"], code["sf_PLANET_DIM"])
    need("submission_features.FLEET_DIM",  pt["FLEET_DIM"],  code["sf_FLEET_DIM"])
    need("submission_features.HISTORY",    pt["HISTORY"],    code["sf_HISTORY"])

    # 2) submission_actor
    need("submission_actor.PLANET_DIM",    pt["PLANET_DIM"], code["sa_PLANET_DIM"])
    need("submission_actor.EMBED_DIM",     pt["EMBED_DIM"],  code["sa_EMBED_DIM"])
    need("submission_actor.HISTORY",       pt["HISTORY"],    code["sa_HISTORY"])

    # 3) env_wrapper (train/submission parity)
    need("env_wrapper.PLANET_DIM",         pt["PLANET_DIM"], code["ew_PLANET_DIM"])
    need("env_wrapper.FLEET_DIM",          pt["FLEET_DIM"],  code["ew_FLEET_DIM"])

    # 4) config.yaml
    need("config.yaml model.embed_dim",       pt["EMBED_DIM"], code["cfg_embed_dim"])
    need("config.yaml env.history_turns",     pt["HISTORY"],   code["cfg_history"])
    need("config.yaml env.max_planets",       code["sa_MAX_PLANETS"], code["cfg_max_planets"])

    # 5) ACTION_DIM 분해
    expected_action_dim = 1 + code["sa_NUM_SHIPS_BINS"] + code["sa_MAX_PLANETS"]
    if pt["ACTION_DIM"] != expected_action_dim:
        errs.append(
            f"  ACTION_DIM: .pt={pt['ACTION_DIM']} ≠ "
            f"1 + NUM_SHIPS_BINS({code['sa_NUM_SHIPS_BINS']}) "
            f"+ MAX_PLANETS({code['sa_MAX_PLANETS']}) = {expected_action_dim}"
        )
    if code["sa_ACTION_DIM"] != expected_action_dim:
        errs.append(
            f"  submission_actor.ACTION_DIM={code['sa_ACTION_DIM']} ≠ "
            f"derived {expected_action_dim}"
        )

    if errs:
        msg = "차원 불일치 — 다음을 맞춘 후 재실행:\n" + "\n".join(errs)
        msg += "\n\n  대처:"
        msg += "\n   • PLANET_DIM 변경: env_wrapper.py / submission_features.py / submission_actor.py 모두 동기화"
        msg += "\n   • EMBED_DIM 변경: config.yaml 의 model.embed_dim 만 동기화 (코드는 자동 반영)"
        msg += "\n   • HISTORY 변경:   config.yaml 의 env.history_turns 동기화"
        msg += "\n   • 배수 bins 변경: config.yaml 의 model.ships_multiplier_bins 동기화"
        fail(msg)

    ok("차원 검증 통과 ("
       f"PLANET_DIM={pt['PLANET_DIM']} FLEET_DIM={pt['FLEET_DIM']} "
       f"EMBED_DIM={pt['EMBED_DIM']} HISTORY={pt['HISTORY']} "
       f"ACTION_DIM={pt['ACTION_DIM']})")


# ── Smoke ────────────────────────────────────────────────────────────────────

def smoke_load_forward(ckpt: dict, code: dict) -> None:
    """OrbitWarsActor 인스턴스화 → state_dict load → forward(zeros)."""
    sys.path.insert(0, str(ROOT))
    for mod in ("submission_actor", "submission_features"):
        sys.modules.pop(mod, None)
    sa = importlib.import_module("submission_actor")

    model = sa.OrbitWarsActor()
    res = model.load_state_dict(ckpt, strict=False)

    # critic.* 외에 unexpected 가 있으면 의심스러움.
    unexpected_non_critic = [k for k in res.unexpected_keys if not k.startswith("critic.")]
    if unexpected_non_critic:
        fail(
            "state_dict 에 critic 외 unexpected 키 발견 — 모델 layout 어긋남:\n"
            f"  {unexpected_non_critic[:10]}"
        )
    if res.missing_keys:
        # actor 모델이 .pt 보다 더 많은 파라미터를 요구 — 무조건 실패.
        fail(
            "state_dict 에 missing 키 발견 (모델이 더 큼) — 학습/제출 model 정의 어긋남:\n"
            f"  {res.missing_keys[:10]}"
        )

    # 0-tensor forward
    H, P, PD = code["sf_HISTORY"], code["sa_MAX_PLANETS"], code["sa_PLANET_DIM"]
    F, FD    = code["cfg_max_fleets"], code["sa_NUM_SHIPS_BINS"]
    fleet_dim = code["sf_FLEET_DIM"]
    flat_size = H * (P * PD + F * fleet_dim)
    obs = torch.zeros(1, flat_size, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        out = model(obs)

    expected_shape = (1, P, code["sa_ACTION_DIM"])
    if tuple(out.shape) != expected_shape:
        fail(
            f"forward 출력 shape 불일치: got {tuple(out.shape)} "
            f"expected {expected_shape}"
        )

    if torch.isnan(out).any() or torch.isinf(out).any():
        fail("forward 출력에 NaN/Inf — 학습 발산 또는 .pt 손상 의심.")

    ok(f"smoke load+forward 통과 (input_size={flat_size}, "
       f"missing/critic={len(res.unexpected_keys)})")


# ── encode_planets 출력 dim ──────────────────────────────────────────────────

def smoke_encode(code: dict) -> None:
    """encode_planets 출력 shape 가 PLANET_DIM 과 일치하는지."""
    sys.path.insert(0, str(ROOT))
    sf = importlib.import_module("submission_features")

    # 더미 행성 1개 (id, owner, x, y, radius, ships, production)
    raw_planets = [(0, 0, 50.0, 50.0, 5.0, 50, 2)]
    raw_fleets  = []
    arr = sf.encode_planets(raw_planets, raw_fleets, player=0, comet_ids=set(), comets=[])

    if arr.shape != (sf.MAX_PLANETS, sf.PLANET_DIM):
        fail(
            f"encode_planets 출력 shape 어긋남: "
            f"got {arr.shape} expected ({sf.MAX_PLANETS}, {sf.PLANET_DIM})"
        )
    ok(f"encode_planets 출력 shape 검증 통과 ({arr.shape})")


# ── Bundle ────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
        # dirty 표시
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL,
        )
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "nogit"


def _load_submission_yaml() -> dict:
    """submission.yaml 읽기. 없으면 빈 dict. 형 검증은 호출자 책임."""
    if not SUBMISSION_YAML.exists():
        return {}
    try:
        with open(SUBMISSION_YAML) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        fail(f"submission.yaml 파싱 실패: {e}")
    if not isinstance(data, dict):
        fail(f"submission.yaml 의 최상위가 mapping 이 아님 (got {type(data).__name__}).")
    return data


def _resolve_pt_path(cli_arg: str | None) -> Path:
    """CLI arg → submission.yaml 순으로 .pt 경로 결정.

    상대 경로는 repo root 기준으로 해석.
    """
    if cli_arg:
        p = Path(cli_arg)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    cfg = _load_submission_yaml()
    ckpt = cfg.get("checkpoint")
    if not ckpt or not isinstance(ckpt, str):
        fail(
            "checkpoint 경로 미지정.\n"
            "  → CLI arg: `python build_submission.py path/to/.pt`\n"
            f"  → 또는 {SUBMISSION_YAML.name} 의 `checkpoint:` 필드 채우기"
        )
    p = Path(ckpt)
    resolved = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
    info(f"submission.yaml 에서 checkpoint 사용: {resolved}")
    return resolved


def _resolve_output_path(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if p.is_dir():
            fail(f"-o 가 디렉토리: {p}. 파일 경로를 지정하세요.")
        return p
    out_dir = ROOT / "submissions"
    out_dir.mkdir(exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"sub_{ts}_{_git_sha()}.tar.gz"


def bundle(pt_path: Path, out_path: Path) -> None:
    """필요 파일 + .pt 를 weights.pt 로 묶어 tar.gz 생성."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "submission"
        staging.mkdir()

        # 1) 필수 파일/디렉토리 복사
        for fname in BUNDLE_FILES:
            src = ROOT / fname
            if not src.exists():
                fail(f"번들 대상 파일 없음: {src}")
            shutil.copy2(src, staging / fname)

        for dname in BUNDLE_DIRS:
            src = ROOT / dname
            if not src.exists():
                fail(f"번들 대상 디렉토리 없음: {src}")
            # __pycache__ 는 제외
            shutil.copytree(
                src, staging / dname,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        # 2) .pt → weights.pt 로 rename
        shutil.copy2(pt_path, staging / "weights.pt")

        # 3) tar.gz 압축 — 최상위 경로 평탄화 (Kaggle 은 tar 풀어서 그대로 사용)
        with tarfile.open(out_path, "w:gz") as tar:
            for entry in sorted(staging.iterdir()):
                tar.add(entry, arcname=entry.name)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    ok(f"bundle 생성: {out_path}  ({size_mb:.2f} MiB)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Kaggle 제출 archive 자동 빌드.
            .pt 의 차원과 현재 코드를 검증한 뒤, 통과 시 tar.gz 생성.
        """),
    )
    ap.add_argument("pt", nargs="?",
                    help="checkpoint .pt 경로 (생략 시 submission.yaml 의 checkpoint 사용)")
    ap.add_argument("-o", "--output", help="출력 tar.gz 경로 (기본: submissions/sub_<ts>_<sha>.tar.gz)")
    ap.add_argument("--skip-smoke",  action="store_true", help="forward smoke 생략")
    ap.add_argument("--skip-encode", action="store_true", help="encode_planets shape 검증 생략")
    args = ap.parse_args()

    pt_path = _resolve_pt_path(args.pt)
    if not pt_path.exists():
        fail(f".pt 파일 없음: {pt_path}")

    info(f"checkpoint 로드: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        fail(f".pt 가 dict(state_dict) 아님: type={type(ckpt).__name__}. "
             f"학습 코드가 model.state_dict() 로 저장하는지 확인.")

    pt_dims = extract_pt_dims(ckpt)
    info(
        f".pt dims: PLANET_DIM={pt_dims['PLANET_DIM']} "
        f"FLEET_DIM={pt_dims['FLEET_DIM']} EMBED_DIM={pt_dims['EMBED_DIM']} "
        f"HISTORY={pt_dims['HISTORY']} ACTION_DIM={pt_dims['ACTION_DIM']}"
    )

    code_dims = extract_code_dims()
    info(
        f"code dims: PLANET_DIM={code_dims['sf_PLANET_DIM']} "
        f"FLEET_DIM={code_dims['sf_FLEET_DIM']} EMBED_DIM={code_dims['cfg_embed_dim']} "
        f"HISTORY={code_dims['sf_HISTORY']} "
        f"NUM_SHIPS_BINS={code_dims['sa_NUM_SHIPS_BINS']} "
        f"MAX_PLANETS={code_dims['sa_MAX_PLANETS']}"
    )

    verify(pt_dims, code_dims)

    if not args.skip_encode:
        smoke_encode(code_dims)
    else:
        warn("encode 검증 생략 (--skip-encode)")

    if not args.skip_smoke:
        smoke_load_forward(ckpt, code_dims)
    else:
        warn("smoke load+forward 생략 (--skip-smoke)")

    out_path = _resolve_output_path(args.output)
    bundle(pt_path, out_path)

    info("끝.")


if __name__ == "__main__":
    main()
