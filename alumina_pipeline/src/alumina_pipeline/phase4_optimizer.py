"""Phase 4 — The Optimizer: Bayesian optimisation / active learning with BoTorch.

* Surrogate: exact GP over Morgan fingerprints of the full solvent (generated
  component + partner) with a Tanimoto kernel — the standard choice for
  molecular BO (cf. GAUCHE).
* Acquisition: batch qLogEI (or qUCB) maximised over the discrete pool that
  Phase 2 proposed (``optimize_acqf_discrete``, greedy sequential batch).
* Feedback to the generator: the fitted GP is persisted to ``surrogate_state``.
  REINVENT's ExternalProcess scorer reloads it and rewards molecules by their
  UCB value, so the RL update moves the generator's weights toward regions the
  GP believes will beat the lattice (or where it is still uncertain).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete

log = logging.getLogger("alumina.phase4")
DTYPE = torch.double


class TanimotoKernel(Kernel):
    """k(x, x') = <x, x'> / (|x|^2 + |x'|^2 - <x, x'>)  for non-negative vectors."""

    has_lengthscale = False

    def forward(self, x1, x2, diag: bool = False, **params):
        dot = x1 @ x2.transpose(-2, -1)
        n1 = (x1 * x1).sum(-1, keepdim=True)
        n2 = (x2 * x2).sum(-1, keepdim=True)
        k = dot / (n1 + n2.transpose(-2, -1) - dot).clamp_min(1e-8)
        return k.diagonal(dim1=-2, dim2=-1) if diag else k


class Featurizer:
    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def __call__(self, smiles: list[str]) -> torch.Tensor:
        fps = [self.gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float64)
               for s in smiles]
        return torch.tensor(np.stack(fps), dtype=DTYPE)


def build_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    return SingleTaskGP(X, Y, covar_module=ScaleKernel(TanimotoKernel()),
                        outcome_transform=Standardize(m=1))


class BayesianOptimizer:
    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.feat = Featurizer(cfg["fingerprint_bits"], cfg["fingerprint_radius"])
        self.rng = random.Random(seed)
        self.smiles: list[str] = []
        self.X = torch.empty(0, cfg["fingerprint_bits"], dtype=DTYPE)
        self.Y = torch.empty(0, 1, dtype=DTYPE)
        self.model: SingleTaskGP | None = None

    # ------------------------------------------------------------------
    def observe(self, results: list[dict]) -> None:
        if not results:
            return
        self.smiles += [r["smiles"] for r in results]
        self.X = torch.cat([self.X, self.feat([r["smiles"] for r in results])])
        self.Y = torch.cat([self.Y, torch.tensor([[r["objective"]] for r in results], dtype=DTYPE)])
        self._fit()

    def _fit(self) -> None:
        if len(self.Y) < 2:
            return
        self.model = build_gp(self.X, self.Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.model.likelihood, self.model))
        log.info("GP refit on %d observations (best objective %.3f)", len(self.Y), self.Y.max().item())

    # ------------------------------------------------------------------
    def suggest(self, pool_smiles: list[str], q: int) -> list[int]:
        """Indices into ``pool_smiles`` to send to the (expensive) evaluator."""
        q = min(q, len(pool_smiles))
        if self.model is None or len(self.Y) < self.cfg["n_initial"]:
            log.info("Initial design: random batch of %d", q)
            return self.rng.sample(range(len(pool_smiles)), q)

        Xp = self.feat(pool_smiles)
        if self.cfg["acquisition"] == "qUCB":
            acq = qUpperConfidenceBound(self.model, beta=self.cfg["ucb_beta"])
        else:
            acq = qLogExpectedImprovement(self.model, best_f=self.Y.max())
        chosen, _ = optimize_acqf_discrete(acq, q=q, choices=Xp, unique=True)
        idx = []
        for row in chosen:
            match = torch.nonzero((Xp == row).all(dim=-1)).flatten().tolist()
            idx.append(next(m for m in match if m not in idx))
        return idx

    # ------------------------------------------------------------------
    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "smiles": self.smiles, "X": self.X, "Y": self.Y, "cfg": self.cfg,
            "state_dict": self.model.state_dict() if self.model is not None else None,
        }, path)


# -----------------------------------------------------------------------------
# Used by the REINVENT scorer subprocess
# -----------------------------------------------------------------------------
class SurrogateScorer:
    """Reload the persisted GP and score arbitrary solvents by UCB."""

    def __init__(self, path: Path):
        self.model = None
        if not Path(path).exists():
            return
        st = torch.load(path, weights_only=False)
        if st["state_dict"] is None:
            return
        self.feat = Featurizer(st["cfg"]["fingerprint_bits"], st["cfg"]["fingerprint_radius"])
        self.beta = st["cfg"]["ucb_beta"]
        self.y_best = st["Y"].max().item()
        self.y_std = st["Y"].std().item() if len(st["Y"]) > 1 else 1.0
        self.model = build_gp(st["X"], st["Y"])
        self.model.load_state_dict(st["state_dict"])
        self.model.eval()

    def ucb_reward(self, smiles: list[str]) -> np.ndarray:
        """Map UCB relative to the incumbent into (0, 1) via a logistic."""
        if self.model is None:
            return np.full(len(smiles), 0.5)
        with torch.no_grad():
            post = self.model.posterior(self.feat(smiles))
            ucb = post.mean.squeeze(-1) + self.beta ** 0.5 * post.variance.clamp_min(0).sqrt().squeeze(-1)
        z = (ucb.numpy() - self.y_best) / max(self.y_std, 1e-6)
        return 1.0 / (1.0 + np.exp(-2.0 * z))
