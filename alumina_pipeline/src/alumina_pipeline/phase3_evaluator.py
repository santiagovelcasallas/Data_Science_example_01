"""Phase 3 — The Evaluator: solvent vs. Al2O3 surface ("the clash").

Thermodynamic cycle (descriptor level)
--------------------------------------
Pulling one surface O out of the lattice costs ``E_break`` (Phase 1). The
solvent pays part of that bill by solvating the species that leaves the
lattice. With ``dG_solv(probe | solvent)`` predicted by Chemprop:

    dG_extract = E_break + sum_p  w_p * dG_solv(p | solvent) + calibration_offset

Binary success criterion (as specified):

    success  <=>  dG_extract < 0
    (removing O into the solvent is thermodynamically cheaper than keeping the
     crystal intact)

A weaker, always-informative flag ``lowers_barrier`` (dG_solv_eff < 0) records
whether the solvent helps at all. The continuous ``objective`` that BoTorch
maximises is ``-dG_extract`` (kJ/mol / 100) minus a constraint penalty, so the
optimiser gets gradient information even while every candidate is failing.

IMPORTANT caveat: the cycle mixes a solid-state energy (MLIP, 0 K, vs. 1/2 O2)
with a solution free energy (Chemprop). It is a *ranking descriptor*, not a
balanced reaction. Calibrate ``calibration_offset`` against a reference system
(e.g. a solvent with measured Al2O3 solubility) before trusting the 0 threshold.
The optional ``explicit_interface`` check (MACE on slab + solvent) is the
physics-based cross-check for the top candidates.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .chem_space import Candidate, constraint_report
from .config import EV_TO_KJ_MOL, KCAL_TO_KJ, resolve
from .phase1_target import TargetResult

log = logging.getLogger("alumina.phase3")

INFEASIBLE_PENALTY = 5.0  # objective units (= 500 kJ/mol)


class SolvationEvaluator:
    def __init__(self, cfg: dict, constraints: dict, target: TargetResult, run_dir: Path,
                 device: str = "cpu", calculator_cfg: dict | None = None):
        self.cfg = cfg
        self.constraints = constraints
        self.target = target
        self.run_dir = run_dir
        self.device = device
        self.calculator_cfg = calculator_cfg or {}
        self.probes: dict[str, str] = cfg["probe_solutes"]
        self.weights: dict[str, float] = cfg["probe_weights"]
        self.offset = float(cfg.get("calibration_offset_kJ_mol", 0.0))

    # ------------------------------------------------------------------
    def evaluate(self, cands: list[Candidate], round_idx: int) -> list[dict]:
        if not cands:
            return []
        if self.cfg["backend"] == "chemprop":
            dg = self._chemprop_dG(cands, round_idx)
        else:
            dg = self._surrogate_dG(cands)

        e_break = self.target.e_break_o_kj_mol
        results = []
        for i, c in enumerate(cands):
            rep = constraint_report(c, self.constraints)
            dg_eff = float(sum(self.weights[p] * dg[p][i] for p in self.probes))
            dg_extract = e_break + dg_eff + self.offset
            objective = -dg_extract / 100.0 - (1.0 - rep["penalty"]) * INFEASIBLE_PENALTY
            results.append({
                "smiles": c.smiles, "generated": c.generated, "partner": c.partner, "kind": c.kind,
                **{f"dG_solv_{p}_kJ_mol": float(dg[p][i]) for p in self.probes},
                "dG_solv_eff_kJ_mol": dg_eff,
                "E_break_kJ_mol": e_break,
                "dG_extract_kJ_mol": dg_extract,
                "success": bool(dg_extract < 0 and rep["penalty"] > 0),
                "lowers_barrier": bool(dg_eff < 0),
                "objective": objective,
                **{f"c_{k}": v for k, v in rep.items()},
                "failure_reason": self._failure_reason(dg_extract, rep),
                "round": round_idx,
            })

        if self.cfg["explicit_interface"]["enabled"]:
            self._explicit_check(results)
        return results

    @staticmethod
    def _failure_reason(dg_extract: float, rep: dict) -> str | None:
        reasons = []
        if rep.get("volatile"):
            reasons.append("volatile (Tb < limit)")
        if rep.get("toxicity_alerts", 0):
            reasons.append(f"{rep['toxicity_alerts']} toxicity alert(s)")
        if rep.get("penalty", 1) < 1 and not reasons:
            reasons.append("size/chain constraint")
        if dg_extract >= 0:
            reasons.append(f"thermodynamic shortfall {dg_extract:.0f} kJ/mol")
        return "; ".join(reasons) or None

    # ------------------------------------------------------------------
    # Chemprop (MIT) — 2-component model: (solvent, solute) -> dG_solv
    # ------------------------------------------------------------------
    def _chemprop_dG(self, cands: list[Candidate], round_idx: int) -> dict[str, np.ndarray]:
        cp = self.cfg["chemprop"]
        model = resolve(cp["model_path"])
        if not model.exists():
            raise FileNotFoundError(f"Chemprop checkpoint not found: {model} (see README: train it)")
        workdir = self.run_dir / "chemprop"
        workdir.mkdir(parents=True, exist_ok=True)

        rows = [{"solvent": c.smiles, "solute": s, "probe": p, "idx": i}
                for i, c in enumerate(cands) for p, s in self.probes.items()]
        inp, out = workdir / f"round{round_idx:02d}_in.csv", workdir / f"round{round_idx:02d}_preds.csv"
        pd.DataFrame(rows)[["solvent", "solute"]].to_csv(inp, index=False)
        cmd = [cp["executable"], "predict", "--test-path", str(inp), "--preds-path", str(out),
               "--model-paths", str(model), "--smiles-columns", "solvent", "solute"]
        subprocess.run(cmd, check=True)

        preds = pd.read_csv(out)
        value_col = [c for c in preds.columns if c not in ("solvent", "solute")][0]
        vals = preds[value_col].to_numpy(dtype=float)
        if cp.get("output_units", "kcal/mol") == "kcal/mol":
            vals = vals * KCAL_TO_KJ

        dg = {p: np.zeros(len(cands)) for p in self.probes}
        for row, v in zip(rows, vals):
            dg[row["probe"]][row["idx"]] = v
        return dg

    # ------------------------------------------------------------------
    # Heuristic surrogate — WIRING TEST ONLY, not physics.
    # Encodes the qualitative trend that acidic, strongly H-bonding,
    # coordinating-anion solvents dissolve metal oxides better.
    # ------------------------------------------------------------------
    def _surrogate_dG(self, cands: list[Candidate]) -> dict[str, np.ndarray]:
        acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
        sulf = Chem.MolFromSmarts("S(=O)(=O)[OX2H1]")
        coord = {"chloride": 1.0, "hydrogen_sulfate": 1.2, "dihydrogen_phosphate": 1.1,
                 "acetate": 0.8, "methanesulfonate": 0.6, "triflate": 0.3, "TFSI": 0.1,
                 "dicyanamide": 0.5, "choline_chloride": 0.9, "betaine_hcl": 1.0,
                 "tetrabutylammonium_bromide": 0.4}
        out = {p: np.zeros(len(cands)) for p in self.probes}
        for i, c in enumerate(cands):
            mol = Chem.MolFromSmiles(c.smiles)
            n_acid = len(mol.GetSubstructMatches(acid)) + 1.5 * len(mol.GetSubstructMatches(sulf))
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            logp = Descriptors.MolLogP(mol)
            base = -(330 + 45 * n_acid + 20 * hbd + 90 * coord.get(c.partner, 0.5) - 12 * max(logp, 0))
            noise = int(hashlib.md5(c.smiles.encode()).hexdigest()[:6], 16) / 0xFFFFFF - 0.5
            for p in self.probes:
                scale = {"hydroxide": 1.0, "water": 0.12, "aluminate": 0.8}.get(p, 0.5)
                out[p][i] = scale * base + 20 * noise
        return out

    # ------------------------------------------------------------------
    # Optional physics cross-check: MACE on slab + solvent
    #   E_break_solv = E(slab_vac + solv) + 1/2 E(O2) - E(slab + solv)
    # ------------------------------------------------------------------
    def _explicit_check(self, results: list[dict]) -> None:
        from ase.io import read
        from ase.optimize import BFGS
        from rdkit.Chem import AllChem

        from .phase1_target import _fix_lower_half, get_calculator

        ec = self.cfg["explicit_interface"]
        calc = get_calculator(self.calculator_cfg.get("calculator", "mace"), self.device,
                              self.calculator_cfg.get("mace_model", "medium"),
                              self.calculator_cfg.get("mace_dtype", "float64"))
        slab = read(self.target.slab_file)
        o_idx = self.target.removed_o_index
        site = slab.positions[o_idx]

        top = sorted((r for r in results if r["c_penalty"] > 0), key=lambda r: -r["objective"])[: ec["top_k"]]
        for r in top:
            mol = Chem.AddHs(Chem.MolFromSmiles(r["smiles"]))
            if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
                r["explicit_error"] = "embedding failed"
                continue
            AllChem.MMFFOptimizeMolecule(mol)
            pos = mol.GetConformer().GetPositions()
            pos = pos - pos.mean(axis=0) + np.array([site[0], site[1], slab.positions[:, 2].max() + 3.0])
            syms = [a.GetSymbol() for a in mol.GetAtoms()]

            from ase import Atoms

            ads = Atoms(syms, positions=pos)
            energies = {}
            for tag, base in (("intact", slab.copy()), ("vacancy", slab.copy())):
                if tag == "vacancy":
                    del base[o_idx]
                sys_ = base + ads
                _fix_lower_half(sys_)
                sys_.calc = calc
                BFGS(sys_, logfile=None).run(fmax=ec["fmax"], steps=ec["max_steps"])
                energies[tag] = float(sys_.get_potential_energy())
            e_break_solv = energies["vacancy"] + 0.5 * self.target.e_o2 - energies["intact"]
            r["explicit_E_break_kJ_mol"] = e_break_solv * EV_TO_KJ_MOL
            r["explicit_lowers_barrier"] = bool(e_break_solv < self.target.e_break_o)
            log.info("  explicit MACE: %s  E_break %.0f -> %.0f kJ/mol", r["smiles"],
                     self.target.e_break_o_kj_mol, r["explicit_E_break_kJ_mol"])
