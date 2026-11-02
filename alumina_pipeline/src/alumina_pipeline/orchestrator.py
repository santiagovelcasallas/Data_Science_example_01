"""Orchestrator — runs the 4-phase active-learning loop.

    Phase 1 (once, cached)  Al2O3 surface target          ASE + MACE/CHGNet
    loop for n_rounds:
      Phase 2  propose pool of IL / DES solvents        REINVENT 4 (RL agent)
      Phase 4  pick q most informative candidates       BoTorch qLogEI
      Phase 3  evaluate dG_extract, binary success       Chemprop (+ MACE check)
      Phase 4  refit GP, persist it -> next REINVENT round is rewarded by it

Usage:
    python -m alumina_pipeline.orchestrator --config configs/pipeline.yaml
    python -m alumina_pipeline.orchestrator --config configs/pipeline.yaml --smoke
    python -m alumina_pipeline.orchestrator --config configs/pipeline.yaml --phase1-only
    python -m alumina_pipeline.orchestrator --config configs/pipeline.yaml --overlay kaggle.yaml \
        --resume runs/alumina_v0_20261101-120000 --rounds 20
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from .config import dump_json, load_config, pick_device, resolve, seed_everything, setup_logging
from .phase1_target import compute_target
from .phase2_generator import SolventGenerator
from .phase3_evaluator import SolvationEvaluator
from .phase4_optimizer import BayesianOptimizer

log = logging.getLogger("alumina")


def apply_smoke(cfg: dict) -> dict:
    """CPU-only, minutes-scale configuration that exercises every code path."""
    cfg = copy.deepcopy(cfg)
    cfg["run"]["name"] += "_smoke"
    cfg["phase1"].update(structure_source="builtin", calculator="emt", supercell=[1, 1, 1],
                         max_steps=30, min_slab_size=8.0, cache_file="data/phase1_target_smoke.json")
    cfg["phase2"]["backend"] = "enumerator"
    cfg["phase2"]["pool_size"] = 128
    cfg["phase3"]["backend"] = "surrogate"
    cfg["phase3"]["explicit_interface"]["enabled"] = False
    cfg["phase4"].update(n_rounds=4, batch_size=6, n_initial=8)
    return cfg


def summarize_round(results: list[dict], round_idx: int) -> dict:
    df = pd.DataFrame(results)
    reasons = Counter(r for rs in df["failure_reason"].dropna() for r in rs.split("; ")
                      if not r.startswith("thermodynamic"))
    s = {
        "round": round_idx,
        "n_evaluated": len(df),
        "n_success": int(df["success"].sum()),
        "n_lowers_barrier": int(df["lowers_barrier"].sum()),
        "best_dG_extract_kJ_mol": float(df["dG_extract_kJ_mol"].min()),
        "best_smiles": df.loc[df["objective"].idxmax(), "smiles"],
        "constraint_failures": dict(reasons),
        "mean_objective_by_kind": df.groupby("kind")["objective"].mean().round(3).to_dict(),
    }
    log.info("Round %d: %d/%d success | best dG_extract %.1f kJ/mol | %s",
             round_idx, s["n_success"], s["n_evaluated"], s["best_dG_extract_kJ_mol"], s["best_smiles"])
    if reasons:
        log.info("  constraint failures: %s", dict(reasons))
    return s


def deep_update(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def run(cfg: dict, force_phase1: bool = False, phase1_only: bool = False,
        resume: str | None = None) -> Path:
    seed_everything(cfg["run"]["seed"])
    device = pick_device(cfg["run"]["device"])
    if resume:
        run_dir = resolve(resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"--resume: run directory not found: {run_dir}")
    else:
        run_dir = resolve(cfg["run"]["output_dir"]) / f"{cfg['run']['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # The REINVENT scorer subprocess reads this exact config (incl. GP state path).
    cfg["phase4"]["surrogate_state"] = str(run_dir / "bo_state.pt")
    eff_cfg = run_dir / "effective_config.yaml"
    eff_cfg.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}))
    log.info("Run dir: %s | device: %s", run_dir, device)

    # ---------------- Phase 1 ----------------
    target = compute_target(cfg["phase1"], device=device, force=force_phase1)
    dump_json(target.to_dict(), run_dir / "phase1_target.json")
    log.info("Target: gamma=%.3f J/m^2, E_break(O)=%.1f kJ/mol [%s]",
             target.surface_energy, target.e_break_o_kj_mol, target.calculator)
    if phase1_only:
        return run_dir

    gen = SolventGenerator(cfg["phase2"], run_dir, eff_cfg, seed=cfg["run"]["seed"], device=device)
    evaluator = SolvationEvaluator(cfg["phase3"], cfg["phase2"]["constraints"], target, run_dir,
                                   device=device, calculator_cfg=cfg["phase1"])
    bo = BayesianOptimizer(cfg["phase4"], seed=cfg["run"]["seed"])

    history: list[dict] = []
    summaries: list[dict] = []
    evaluated: set[str] = set()
    start_round = 0

    # ---------------- Resume (e.g. after a Kaggle session timeout) ----------------
    if resume and (run_dir / "history.csv").exists():
        hist = pd.read_csv(run_dir / "history.csv")
        history = hist.to_dict("records")
        evaluated = set(hist["smiles"])
        start_round = int(hist["round"].max()) + 1
        if (run_dir / "rounds.json").exists():
            summaries = json.loads((run_dir / "rounds.json").read_text())
        bo.observe(history)
        bo.save_state(Path(cfg["phase4"]["surrogate_state"]))
        chkpts = sorted((run_dir / "reinvent").glob("agent_round*.chkpt"))
        if chkpts:
            gen.agent_file = chkpts[-1]
        log.info("Resumed %s: %d evaluations, continuing at round %d (agent: %s)",
                 run_dir.name, len(history), start_round, gen.agent_file)

    for r in range(start_round, cfg["phase4"]["n_rounds"]):
        pool = gen.propose(r, exclude=evaluated)                                   # Phase 2
        if not pool:
            log.warning("Generator exhausted; stopping.")
            break
        picks = bo.suggest([c.smiles for c in pool], cfg["phase4"]["batch_size"])   # Phase 4
        batch = [pool[i] for i in picks]
        results = evaluator.evaluate(batch, r)                                      # Phase 3
        bo.observe(results)                                                         # Phase 4
        bo.save_state(Path(cfg["phase4"]["surrogate_state"]))

        evaluated |= {c.smiles for c in batch}
        history += results
        summaries.append(summarize_round(results, r))
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        dump_json(summaries, run_dir / "rounds.json")

    if not history:
        log.warning("No evaluations were made (n_rounds already reached?).")
        return run_dir
    df = pd.DataFrame(history).sort_values("objective", ascending=False)
    top = df.head(10)[["smiles", "kind", "partner", "dG_extract_kJ_mol", "success", "failure_reason"]]
    top.to_csv(run_dir / "top_candidates.csv", index=False)
    log.info("Done. %d evaluated, %d successes. Top candidates:\n%s",
             len(df), int(df["success"].sum()), top.to_string(index=False))
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Alumina solvent discovery pipeline")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--smoke", action="store_true", help="CPU smoke test with fallback backends")
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--force-phase1", action="store_true", help="recompute the cached target")
    ap.add_argument("--overlay", action="append", default=[],
                    help="YAML file deep-merged over --config (repeatable), e.g. Kaggle settings")
    ap.add_argument("--resume", help="existing run directory to continue")
    ap.add_argument("--rounds", type=int, help="override phase4.n_rounds (total, incl. resumed)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(resolve(args.config))
    for ov in args.overlay:
        with open(resolve(ov)) as fh:
            deep_update(cfg, yaml.safe_load(fh) or {})
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.rounds is not None:
        cfg["phase4"]["n_rounds"] = args.rounds
    run(cfg, force_phase1=args.force_phase1, phase1_only=args.phase1_only, resume=args.resume)


if __name__ == "__main__":
    main()
