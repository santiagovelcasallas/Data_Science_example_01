"""Chemical space for the solvent: IL / DES assembly and hard constraints.

Representation
--------------
A solvent candidate always has two parts:

* the **generated** part (what REINVENT designs de novo):
    - a cation (formal charge +1)              -> Ionic Liquid
    - a neutral hydrogen-bond donor (HBD)      -> Deep Eutectic Solvent
* a **partner** from a curated library:
    - an anion for ILs
    - a hydrogen-bond acceptor salt (HBA) for DES

REINVENT's ChEMBL prior is trained on single connected molecules, so letting it
design one component and enumerating the counter-component is far more
reliable than asking it to emit dot-disconnected salts directly.

Constraints (penalised, Phase 2 directive)
------------------------------------------
* toxicity  — RDKit Brenk + PAINS structural alerts and long alkyl chains
  (IL ecotoxicity grows roughly monotonically with side-chain length);
* volatility — boiling point < 25 degC. Salts (ILs, the HBA) have negligible
  vapour pressure; for the neutral HBD we use a Joback group-contribution
  estimate of Tb (typical error ~ +/-25 K, so it is a coarse screen only).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "B"}

# --- partner libraries ---------------------------------------------------------
ANIONS = {
    "chloride": "[Cl-]",
    "acetate": "CC(=O)[O-]",
    "hydrogen_sulfate": "OS(=O)(=O)[O-]",
    "methanesulfonate": "CS(=O)(=O)[O-]",
    "triflate": "[O-]S(=O)(=O)C(F)(F)F",
    "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    "dicyanamide": "N#C[N-]C#N",
    "dihydrogen_phosphate": "OP(=O)(O)[O-]",
}
HBA_SALTS = {
    "choline_chloride": "C[N+](C)(C)CCO.[Cl-]",
    "betaine_hcl": "C[N+](C)(C)CC(=O)O.[Cl-]",
    "tetrabutylammonium_bromide": "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]",
}

# --- fallback enumerator building blocks --------------------------------------
CATION_CORES = [
    "C[n+]1ccn({R})c1",          # imidazolium
    "{R}[n+]1ccccc1",             # pyridinium
    "C[N+]1({R})CCCC1",           # pyrrolidinium
    "C[N+](C)(C){R}",             # quaternary ammonium (cholinium-like)
    "CC[N+](CC)(CC){R}",          # triethylammonium
    "C[P+](C)(C){R}",             # phosphonium
]
SUBSTITUENTS = [
    "C", "CC", "CCC", "CCCC", "CCCCCC", "CCCCCCCC", "CCO", "CCCO",
    "CC(=O)O", "CCCS(=O)(=O)O", "CCOC", "CC#N", "CC(N)=O", "CCOCCO",
]
HBD_SEEDS = [
    "NC(N)=O", "NC(N)=S", "CNC(N)=O", "OCCO", "OCC(O)CO", "OCCCO", "OCCOCCO",
    "OC(=O)C(=O)O", "OC(=O)CC(=O)O", "OC(=O)CCC(=O)O", "CC(O)C(=O)O",
    "OCC(=O)O", "OC(=O)C(O)C(O)C(=O)O", "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "NC(=O)C", "OC(=O)c1ccccc1O", "OCC(O)C(O)C(O)C(O)CO", "NCC(=O)O",
]


@dataclass
class Candidate:
    smiles: str                # full solvent, dot-joined (generated . partner)
    generated: str             # de novo component (canonical)
    partner: str               # name of anion / HBA
    kind: str                  # ionic_liquid | deep_eutectic
    meta: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
def canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def classify(smi: str) -> str | None:
    """Which solvent class can this generated molecule seed?"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED_ELEMENTS:
        return None
    charge = Chem.GetFormalCharge(mol)
    if charge == 1:
        return "ionic_liquid"
    if charge == 0 and rdMolDescriptors.CalcNumHBD(mol) >= 1:
        return "deep_eutectic"
    return None


def assemble(generated: str, classes: list[str]) -> list[Candidate]:
    """Pair a generated component with every compatible partner."""
    g = canon(generated)
    kind = classify(g) if g else None
    if kind is None or kind not in classes:
        return []
    lib = ANIONS if kind == "ionic_liquid" else HBA_SALTS
    return [Candidate(f"{g}.{p}", g, name, kind) for name, p in lib.items()]


def enumerate_generated(rng: random.Random, classes: list[str]) -> list[str]:
    """Fallback 'generator': combinatorial cations + seed HBDs (no ML)."""
    out: list[str] = []
    if "ionic_liquid" in classes:
        for core, r in itertools.product(CATION_CORES, SUBSTITUENTS):
            smi = canon(core.format(R=r))
            if smi:
                out.append(smi)
    if "deep_eutectic" in classes:
        out += [c for c in (canon(s) for s in HBD_SEEDS) if c]
    rng.shuffle(out)
    return list(dict.fromkeys(out))


# -----------------------------------------------------------------------------
# Constraints
# -----------------------------------------------------------------------------
_ALERTS = None


def _alert_catalog() -> FilterCatalog:
    global _ALERTS
    if _ALERTS is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _ALERTS = FilterCatalog(params)
    return _ALERTS


def longest_alkyl_chain(mol: Chem.Mol) -> int:
    """Longest path of sp3, acyclic, non-hetero-substituted CH2/CH3 carbons."""
    carbons = [a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "C" and not a.IsInRing() and not a.GetIsAromatic()
               and a.GetHybridization() == Chem.HybridizationType.SP3 and a.GetTotalNumHs() >= 2]
    cset = set(carbons)
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in cset]
           for i in carbons}
    best = 0
    for start in carbons:  # small molecules -> brute-force DFS is fine
        stack = [(start, {start})]
        while stack:
            node, seen = stack.pop()
            best = max(best, len(seen))
            stack.extend((n, seen | {n}) for n in adj[node] if n not in seen)
    return best


# Joback Tb group contributions (K). Tb = 198 + sum.
_JOBACK = {
    "CH3": 23.58, "CH2": 22.88, "CH": 21.74, "C": 18.25,
    "rCH2": 27.15, "rCH": 21.78, "rC": 21.32,
    "=CH2": 18.18, "=CH": 24.96, "=C": 24.14,
    "aCH": 26.73, "aC": 31.01,
    "OH": 92.88, "aOH": 76.34, "O": 22.42, "rO": 31.22,
    "C=O": 76.75, "rC=O": 94.97, "CHO": 72.24, "COOH": 169.09, "COO": 81.10,
    "NH2": 73.23, "NH": 50.17, "rNH": 52.82, "N": 11.74, "aN": 57.55, "CN": 125.66,
    "SH": 63.56, "S": 68.78, "rS": 52.10, "F": -0.03, "Cl": 38.13, "Br": 66.86,
    "=S": 68.78,
}


def joback_tb_kelvin(mol: Chem.Mol) -> float:
    total, used = 198.0, set()
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i in used:
            continue
        s, h, ring, arom = a.GetSymbol(), a.GetTotalNumHs(), a.IsInRing(), a.GetIsAromatic()
        if s == "C":
            dbl_o = [n for n in a.GetNeighbors() if n.GetSymbol() == "O"
                     and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondTypeAsDouble() == 2]
            sgl_o = [n for n in a.GetNeighbors() if n.GetSymbol() == "O"
                     and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondTypeAsDouble() == 1]
            trip_n = [n for n in a.GetNeighbors() if n.GetSymbol() == "N"
                      and mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondTypeAsDouble() == 3]
            if dbl_o:
                acid_oh = [o for o in sgl_o if o.GetTotalNumHs() == 1]
                if acid_oh:
                    total += _JOBACK["COOH"]; used |= {i, dbl_o[0].GetIdx(), acid_oh[0].GetIdx()}
                elif sgl_o:
                    total += _JOBACK["COO"]; used |= {i, dbl_o[0].GetIdx(), sgl_o[0].GetIdx()}
                elif h == 1:
                    total += _JOBACK["CHO"]; used |= {i, dbl_o[0].GetIdx()}
                else:
                    total += _JOBACK["rC=O" if ring else "C=O"]; used |= {i, dbl_o[0].GetIdx()}
                continue
            if trip_n:
                total += _JOBACK["CN"]; used |= {i, trip_n[0].GetIdx()}
                continue
            if arom:
                total += _JOBACK["aCH" if h else "aC"]
            elif any(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds()):
                total += _JOBACK["=CH2" if h == 2 else "=CH" if h == 1 else "=C"]
            elif ring:
                total += _JOBACK["rCH2" if h >= 2 else "rCH" if h == 1 else "rC"]
            else:
                total += _JOBACK["CH3" if h == 3 else "CH2" if h == 2 else "CH" if h == 1 else "C"]
        elif s == "O":
            if h == 1:
                total += _JOBACK["aOH" if any(n.GetIsAromatic() for n in a.GetNeighbors()) else "OH"]
            else:
                total += _JOBACK["rO" if ring else "O"]
        elif s == "N":
            if arom:
                total += _JOBACK["aN"]
            elif h == 2:
                total += _JOBACK["NH2"]
            elif h == 1:
                total += _JOBACK["rNH" if ring else "NH"]
            else:
                total += _JOBACK["N"]
        elif s == "S":
            total += _JOBACK["SH" if h else ("=S" if a.GetDegree() == 1 else "rS" if ring else "S")]
        elif s in ("F", "Cl", "Br"):
            total += _JOBACK[s]
        else:
            total += 30.0  # unparameterised atom: generic contribution
    return total


def constraint_report(cand: Candidate, cfg: dict) -> dict:
    """Return per-constraint diagnostics and a multiplicative penalty in [0, 1]."""
    mol = Chem.MolFromSmiles(cand.generated)
    rep: dict = {"valid": mol is not None}
    if mol is None:
        rep["penalty"] = 0.0
        return rep

    n_alerts = sum(1 for _ in _alert_catalog().GetMatches(mol))
    chain = longest_alkyl_chain(mol)
    heavy = mol.GetNumHeavyAtoms()
    is_salt = cand.kind == "ionic_liquid"
    tb_c = float("inf") if is_salt else joback_tb_kelvin(mol) - 273.15

    rep.update(
        toxicity_alerts=n_alerts,
        alkyl_chain=chain,
        heavy_atoms=heavy,
        mw=Descriptors.MolWt(mol),
        tb_estimate_C=None if is_salt else round(tb_c, 1),
    )

    tox_w = cfg["toxicity_penalty_weight"]
    penalty = 1.0
    penalty *= 1.0 / (1.0 + tox_w * n_alerts)                           # soft: alerts
    penalty *= 1.0 if chain <= cfg["max_alkyl_chain"] else 0.2          # strong: long chains
    penalty *= 1.0 if heavy <= cfg["max_heavy_atoms"] else 0.1
    penalty *= 1.0 if tb_c >= cfg["min_boiling_point_C"] else 0.0       # hard: volatile
    rep["volatile"] = tb_c < cfg["min_boiling_point_C"]
    rep["penalty"] = penalty
    return rep
