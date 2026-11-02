"""Phase 1 — The Target: static simulation of the alpha-Al2O3 (corundum) surface.

The lattice is treated as a *fixed enemy*: nothing in this module is ever
optimised or mutated by the generative loop (core directive). It produces three
numbers that the rest of the pipeline consumes:

* ``e_bulk_per_atom``  — relaxed bulk reference energy (eV/atom)
* ``surface_energy``   — gamma of the most stable stoichiometric (0001) termination (J/m^2)
* ``e_break_o``        — energy to pull the most exposed surface O out of the
  lattice, referenced to 1/2 O2(g):  E(slab - O) + 1/2 E(O2) - E(slab)   (eV)

``e_break_o`` is the "cost of breaking the crystal" that a solvent must beat in
Phase 3.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from .config import EV_TO_KJ_MOL, dump_json, pick_device, resolve

log = logging.getLogger("alumina.phase1")

EV_A2_TO_J_M2 = 16.0217663


@dataclass
class TargetResult:
    source: str
    calculator: str
    n_atoms_bulk: int
    e_bulk_per_atom: float          # eV/atom
    surface_energy: float           # J/m^2
    termination_index: int
    n_atoms_slab: int
    e_slab: float                   # eV
    e_slab_vacancy: float           # eV
    e_o2: float                     # eV
    e_break_o: float                # eV per O
    e_break_o_kj_mol: float         # kJ/mol
    removed_o_index: int
    slab_file: str

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Structure + calculator
# -----------------------------------------------------------------------------
def load_bulk(cfg: dict) -> tuple[Atoms, str]:
    """Corundum conventional (hexagonal, 30-atom) cell."""
    if cfg["structure_source"] == "materials_project":
        api_key = os.environ.get("MP_API_KEY")
        if api_key:
            from mp_api.client import MPRester
            from pymatgen.io.ase import AseAtomsAdaptor

            with MPRester(api_key) as mpr:
                struct = mpr.get_structure_by_material_id(
                    cfg["mp_material_id"], conventional_unit_cell=True
                )
            log.info("Loaded %s from Materials Project (%d atoms)", cfg["mp_material_id"], len(struct))
            return AseAtomsAdaptor.get_atoms(struct), f"materials_project:{cfg['mp_material_id']}"
        log.warning("MP_API_KEY not set -> falling back to built-in corundum structure")

    from ase.spacegroup import crystal

    # Experimental corundum (R-3c, #167), hexagonal setting.
    atoms = crystal(
        ["Al", "O"],
        basis=[(0.0, 0.0, 0.35216), (0.30624, 0.0, 0.25)],
        spacegroup=167,
        cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120],
    )
    syms = atoms.get_chemical_symbols()
    assert (syms.count("Al"), syms.count("O")) == (12, 18), atoms.get_chemical_formula()
    return atoms, "builtin:corundum_R-3c"


def get_calculator(name: str, device: str, mace_model: str = "medium", dtype: str = "float64"):
    name = name.lower()
    if name == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model=mace_model, device=device, default_dtype=dtype)
    if name == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator

        return CHGNetCalculator(use_device=device)
    if name == "emt":
        log.warning("EMT selected: energies are NOT physical for Al2O3. Wiring test only.")
        from ase.calculators.emt import EMT

        return EMT()
    raise ValueError(f"Unknown calculator: {name}")


def _relax(atoms: Atoms, fmax: float, steps: int, cell: bool = False) -> float:
    if cell:
        try:
            from ase.filters import FrechetCellFilter as CellFilter
        except ImportError:  # ase < 3.23
            from ase.constraints import ExpCellFilter as CellFilter
        target = CellFilter(atoms)
    else:
        target = atoms
    BFGS(target, logfile=None).run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())


# -----------------------------------------------------------------------------
# Surface construction
# -----------------------------------------------------------------------------
def build_slabs(bulk: Atoms, cfg: dict) -> list[Atoms]:
    """All stoichiometric terminations of the requested facet (symmetric first)."""
    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    struct = AseAtomsAdaptor.get_structure(bulk)
    gen = SlabGenerator(
        struct,
        tuple(cfg["miller_index"]),
        min_slab_size=cfg["min_slab_size"],
        min_vacuum_size=cfg["min_vacuum_size"],
        center_slab=True,
    )
    bulk_formula = struct.composition.reduced_formula
    slabs = [s for s in gen.get_slabs(symmetrize=False)
             if s.composition.reduced_formula == bulk_formula]
    if not slabs:
        raise RuntimeError("No stoichiometric slab found; increase min_slab_size.")
    symmetric = [s for s in slabs if s.is_symmetric()]
    if symmetric:
        slabs = symmetric
    else:
        log.warning("No symmetric stoichiometric slab: gamma will average two different faces.")
    log.info("Generated %d candidate terminations for %s", len(slabs), cfg["miller_index"])
    out = []
    for s in slabs:  # drop pymatgen site_properties: plain Atoms only
        a = AseAtomsAdaptor.get_atoms(s)
        out.append(Atoms(a.numbers, positions=a.positions, cell=a.cell, pbc=True))
    return out


def _fix_lower_half(atoms: Atoms) -> None:
    z = atoms.positions[:, 2]
    mid = 0.5 * (z.min() + z.max())
    atoms.set_constraint(FixAtoms(mask=z < mid))


def _surface_area(atoms: Atoms) -> float:
    a, b = atoms.cell[0], atoms.cell[1]
    return float(np.linalg.norm(np.cross(a, b)))


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def compute_target(cfg: dict, device: str = "auto", force: bool = False) -> TargetResult:
    from ase.io import write

    cache = resolve(cfg["cache_file"])
    if cache.exists() and not force:
        import json

        log.info("Phase 1 is static -> reusing cached target %s", cache)
        with open(cache) as fh:
            res = TargetResult(**json.load(fh))
        # The cache may come from another machine (e.g. a previous Kaggle notebook).
        if not Path(res.slab_file).exists():
            res.slab_file = str(resolve("data/al2o3_slab.extxyz"))
        return res

    device = pick_device(device)
    calc = get_calculator(cfg["calculator"], device, cfg.get("mace_model", "medium"),
                          cfg.get("mace_dtype", "float64"))

    # 1. Bulk ------------------------------------------------------------------
    bulk, source = load_bulk(cfg)
    bulk.calc = calc
    if cfg["relax_bulk"]:
        e_bulk = _relax(bulk, cfg["fmax"], cfg["max_steps"], cell=True)
    else:
        e_bulk = float(bulk.get_potential_energy())
    e_bulk_pa = e_bulk / len(bulk)
    log.info("Bulk: %d atoms, E = %.4f eV/atom", len(bulk), e_bulk_pa)

    # 2. Surface energy over terminations ---------------------------------------
    best = None
    for i, slab in enumerate(build_slabs(bulk, cfg)):
        slab.calc = calc
        e_slab = (_relax(slab, cfg["fmax"], cfg["max_steps"]) if cfg["relax_slab"]
                  else float(slab.get_potential_energy()))
        gamma = (e_slab - len(slab) * e_bulk_pa) / (2 * _surface_area(slab)) * EV_A2_TO_J_M2
        log.info("  termination %d: %d atoms, gamma = %.3f J/m^2", i, len(slab), gamma)
        if best is None or gamma < best[0]:
            best = (gamma, i, slab)
    gamma, term_idx, slab = best

    # 3. O-extraction energy on a laterally repeated slab ---------------------
    slab = slab.repeat(tuple(cfg["supercell"]))
    _fix_lower_half(slab)
    slab.calc = calc
    e_slab = _relax(slab, cfg["fmax"], cfg["max_steps"])

    o_idx = [a.index for a in slab if a.symbol == "O"]
    top_o = int(max(o_idx, key=lambda k: slab.positions[k, 2]))
    defect = slab.copy()
    del defect[top_o]
    _fix_lower_half(defect)
    defect.calc = calc
    e_def = _relax(defect, cfg["fmax"], cfg["max_steps"])

    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]], cell=[15, 15, 15], pbc=True)
    o2.center()
    o2.calc = calc
    e_o2 = _relax(o2, cfg["fmax"], cfg["max_steps"])

    e_break = e_def + 0.5 * e_o2 - e_slab
    log.info("Surface O extraction cost: %.3f eV  (%.1f kJ/mol)", e_break, e_break * EV_TO_KJ_MOL)

    slab_file = resolve("data/al2o3_slab.extxyz")
    slab_file.parent.mkdir(parents=True, exist_ok=True)
    write(slab_file, Atoms(slab.numbers, positions=slab.positions, cell=slab.cell, pbc=True))

    res = TargetResult(
        source=source,
        calculator=cfg["calculator"],
        n_atoms_bulk=len(bulk),
        e_bulk_per_atom=e_bulk_pa,
        surface_energy=gamma,
        termination_index=term_idx,
        n_atoms_slab=len(slab),
        e_slab=e_slab,
        e_slab_vacancy=e_def,
        e_o2=e_o2,
        e_break_o=e_break,
        e_break_o_kj_mol=e_break * EV_TO_KJ_MOL,
        removed_o_index=top_o,
        slab_file=str(slab_file),
    )
    dump_json(res.to_dict(), cache)
    return res
