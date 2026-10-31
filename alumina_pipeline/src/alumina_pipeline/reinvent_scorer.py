"""REINVENT 4 ``ExternalProcess`` scoring endpoint.

REINVENT pipes one SMILES per line on stdin and reads JSON from stdout:

    {"version": 1, "payload": {"predictions": [score_1, score_2, ...]}}

Score for a generated component g (in [0, 1]):

    max over partners p of   constraint_penalty(g.p) * UCB_reward(g.p)

and 0 if g cannot seed an IL (cation) or DES (neutral H-bond donor).
Before the GP exists, UCB_reward = 0.5, so early RL only learns the class +
toxicity + volatility constraints.

NOTE: verify the JSON contract against the REINVENT 4 version you install
(``reinvent_plugins/components/comp_external_process.py``).
"""

from __future__ import annotations

import argparse
import json
import sys

from .chem_space import assemble, constraint_report
from .config import load_config, resolve
from .phase4_optimizer import SurrogateScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    p2 = cfg["phase2"]
    scorer = SurrogateScorer(resolve(cfg["phase4"]["surrogate_state"]))

    smiles = [line.strip() for line in sys.stdin if line.strip()]
    scores = []
    for smi in smiles:
        cands = assemble(smi, p2["solvent_classes"])
        if not cands:
            scores.append(0.0)
            continue
        pen = [constraint_report(c, p2["constraints"])["penalty"] for c in cands]
        rew = scorer.ucb_reward([c.smiles for c in cands])
        scores.append(float(max(p * r for p, r in zip(pen, rew))))

    json.dump({"version": 1, "payload": {"predictions": scores}}, sys.stdout)


if __name__ == "__main__":
    main()
