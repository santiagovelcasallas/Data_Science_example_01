# Alumina Solvent Discovery Pipeline

Generative active-learning loop that searches for an **Ionic Liquid (IL)** or
**Deep Eutectic Solvent (DES)** that weakens/dissolves the α‑Al₂O₃ (corundum)
lattice at low temperature. The aim is a low-energy recycling route for the
Al–Ga hydrogen-generation cycle (Woodall, Purdue 2007), as an alternative to
Hall–Héroult.

**Core directive:** the Al₂O₃ lattice is a *static target*. Nothing in the loop
mutates, dopes or optimises the solid. All the generative effort goes into the
solvent. CrysVCD (doping filter) is left out on purpose for this reason.

```
                 ┌────────────── Phase 1 (once, cached) ──────────────┐
                 │ ASE + MACE-MP / CHGNet  → corundum (0001) slab     │
                 │ γ_surface, E_break(O) = E(slab−O) + ½E(O₂) − E(slab)│
                 └──────────────────────────┬─────────────────────────┘
                                            │ E_break
   ┌─ Phase 2 ─────────────┐  pool  ┌─ Phase 4 ─────────┐ batch ┌─ Phase 3 ─────────────────────┐
   │ REINVENT 4 RL agent   │──────▶│ BoTorch GP         │──────▶│ Chemprop ΔG_solv(probe|solv)   │
   │ cation → IL + anion   │        │ Tanimoto kernel    │       │ ΔG_extract = E_break + ΣwΔG    │
   │ HBD    → DES + HBA    │◀──────│ qLogEI             │◀──────│ success ⇔ ΔG_extract < 0       │
   └───────────────────────┘ reward └────────────────────┘ y     │ (+ optional MACE slab check)   │
        (GP UCB via ExternalProcess scorer)                       └────────────────────────────────┘
```

## Layout

| Path | Role |
|---|---|
| `configs/pipeline.yaml` | All parameters. Each phase has a real backend and a fallback |
| `src/alumina_pipeline/phase1_target.py` | Corundum from Materials Project (`mp-1143`), slab terminations, surface energy, O-extraction cost |
| `src/alumina_pipeline/chem_space.py` | IL/DES assembly, toxicity alerts, alkyl-chain limit, Joback T_b screen |
| `src/alumina_pipeline/phase2_generator.py` | REINVENT 4 staged-learning driver (writes TOML, carries the agent checkpoint between rounds) + enumerator fallback |
| `src/alumina_pipeline/reinvent_scorer.py` | REINVENT `ExternalProcess` endpoint: constraints × GP-UCB reward |
| `src/alumina_pipeline/phase3_evaluator.py` | Thermodynamic cycle, binary success, failure reasons, optional MACE interface check |
| `src/alumina_pipeline/phase4_optimizer.py` | BoTorch GP (Tanimoto kernel) + qLogEI over the discrete pool |
| `src/alumina_pipeline/orchestrator.py` | The loop |

## Quick start (on your machine)

```bash
cp .env.example .env        # add MP_API_KEY (optional: without it a built-in corundum cell is used)
```

```bash
docker compose build
```

1. **Smoke test.** CPU only, about a minute. It uses EMT, the enumerator and the heuristic surrogate, so it tests the wiring and **not the physics**:

```bash
docker compose run --rm smoke
```

2. **Phase 1 with MACE on GPU.** The result is cached in `data/phase1_target.json`:

```bash
docker compose run --rm pipeline --phase1-only
```

3. **Full loop.** First switch the backends in `configs/pipeline.yaml` (`phase2.backend: reinvent` and `phase3.backend: chemprop`), then:

```bash
docker compose up pipeline
```

Outputs go to `runs/<name>_<timestamp>/`: `history.csv` (every evaluation with
its `failure_reason`), `rounds.json`, `top_candidates.csv`, `bo_state.pt`,
`reinvent/` (TOMLs, agent checkpoints, CSVs, TensorBoard logs).

Without Docker:

```bash
pip install -r requirements.txt && pip install -e . && python -m alumina_pipeline.orchestrator --smoke
```

## Assets you must provide

| Asset | Where | How |
|---|---|---|
| REINVENT prior | `priors/reinvent.prior` | Download from the REINVENT 4 Zenodo record linked in the REINVENT4 README |
| Chemprop solvation model | `models/chemprop_solvation/` | Train a **2-component** (solvent, solute) → ΔG_solv model. Example: `chemprop train --data-path combisolv.csv --smiles-columns solvent solute --target-columns dG --task-type regression -o models/chemprop_solvation`. Use CombiSolv / MNSol / FreeSolv data, and add IL/DES data if you have it |
| MP API key | `.env` | https://next-gen.materialsproject.org/api |

## Scientific caveats (read before trusting a "success")

1. **ΔG_extract is a ranking descriptor, not a balanced reaction.** It adds a 0 K
   solid-state energy (MLIP, referenced to ½O₂) to a solution free energy of a
   *probe* species (OH⁻, H₂O, AlO₂⁻ proxy) predicted by Chemprop. The binary
   `success ⇔ ΔG_extract < 0` criterion is implemented as you specified, but set
   `phase3.calibration_offset_kJ_mol` against a reference solvent with measured
   Al₂O₃ solubility before you treat the zero line as meaningful.
2. **Chemprop domain of applicability.** Public solvation datasets contain few ILs/DES
   and almost no metal-oxo solutes. Predictions for charged probes in ionic
   solvents are extrapolations. Track the ensemble variance and fine-tune on any
   in-house data.
3. **MLIP limits.** MACE-MP/CHGNet are charge- and spin-unaware. The O₂ reference
   (triplet) and charged adsorbates are approximate. Validate the final hits with
   DFT (e.g. VASP/CP2K, + implicit solvent) and the explicit-interface check
   (`phase3.explicit_interface.enabled: true`).
4. **Volatility / toxicity are screens.** Joback T_b has an error of about ±25 K, and the
   structural alerts are not a tox model. Swap in a trained Chemprop T_b / LC50
   model through `chem_space.constraint_report` when you have one.
5. **REINVENT ExternalProcess contract.** The scorer emits
   `{"version":1,"payload":{"predictions":[...]}}`. Check this against the
   REINVENT 4 version you install. `REINVENT_REF` is pinned at build time.
6. **ALIGNN** (mentioned in the brief) is not wired in. Its DGL pins clash with
   the rest of the stack. MACE already covers the solid side. It can be added as a
   separate service if a crystal-property model is needed.

## How the feedback loop "learns from failure"

Every evaluation stores a `failure_reason`: volatility, toxicity alerts,
size/chain limits, or the thermodynamic shortfall in kJ/mol. The objective
`−ΔG_extract/100 − 5·(1−penalty)` stays continuous, so the GP learns *how far*
each failure was from success. After every round the GP is saved. In the next
round REINVENT's RL reward for a molecule is `penalty × sigmoid(UCB − best)`, so the
agent's weights shift toward chemistry the GP rates as promising or still
uncertain. Per-round summaries (`rounds.json`) aggregate the failure modes and
the mean objective per solvent class.
