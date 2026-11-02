"""Generate the three Kaggle notebooks for the alumina pipeline."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def cell(t, s):
    c = {"cell_type": t, "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
    if t == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


def nb(name, cells):
    doc = {
        "cells": [cell(t, s) for t, s in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(doc, indent=1))
    print("wrote", name)


SETUP = ("code", r'''
# ==== Setup comun: codigo en /tmp (efimero), estado en /kaggle/working/alumina (se guarda) ====
import os, glob, json, shutil, subprocess
from pathlib import Path

REPO_URL = "https://github.com/santiagovelcasallas/Data_Science_example_01.git"
BRANCH   = "claude/affectionate-johnson-q9wtj0"   # cambia a "main" tras el merge

ROOT = Path("/kaggle/working/alumina")
CODE = Path("/tmp/repo")
PKG  = CODE / "alumina_pipeline"
for d in ["data", "models", "priors", "runs"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
os.environ["ALUMINA_ROOT"] = str(ROOT)

def secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None

token = secret("GITHUB_TOKEN")   # solo si el repo es privado
url = REPO_URL.replace("https://", f"https://{token}@") if token else REPO_URL
if not CODE.exists():
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, url, str(CODE)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git clone fallo: " + r.stderr[-300:])
print("Codigo en", PKG)
''')

INSTALL = ("code", r'''
# ==== Instalacion (~3-6 min). Se fija la version de torch de Kaggle para no romper CUDA ====
!pip freeze | grep -E "^torch==" > /tmp/constraints.txt; cat /tmp/constraints.txt
!pip install -q -c /tmp/constraints.txt -r {PKG}/kaggle/requirements-kaggle.txt
!pip install -q --no-deps -e {PKG}
!python -c "import torch,rdkit,ase,botorch,alumina_pipeline as a; print('torch',torch.__version__,'| GPU:',torch.cuda.is_available())"
''')


def inputs(pattern):
    return f'sorted(glob.glob("/kaggle/input/**/{pattern}", recursive=True))'


# ---------------------------------------------------------------- 01
nb("01_target_al2o3.ipynb", [
    ("markdown", r'''
# 01 - Fase 1: el objetivo (superficie de Al2O3 con MACE)

Calcula **una sola vez** el objetivo estatico: corindon a-Al2O3 -> cara (0001) -> energia para arrancar un O superficial (**E_break**). El solido nunca se modifica.

**Settings (panel derecho):** Accelerator = GPU **P100** (mejor que T4 en float64) · Internet = On · *(opcional)* Add-ons -> Secrets: `MP_API_KEY`; sin ella se usa una celda de corindon interna. Repo privado -> secret `GITHUB_TOKEN`.

**Guardar:** Save Version -> Save & Run All (Commit). ~10-40 min.
'''),
    SETUP, INSTALL,
    ("code", r'''
MACE_MODEL = "medium"     # small | medium | large
MACE_DTYPE = "float64"    # float64 (preciso, P100) | float32 (rapido, para probar)
SUPERCELL  = [2, 2, 1]    # [1,1,1] = prueba rapida

mp_key = secret("MP_API_KEY")
if mp_key: os.environ["MP_API_KEY"] = mp_key
print("Materials Project:", "si (mp-1143)" if mp_key else "no -> estructura interna")

import yaml
yaml.safe_dump({
    "run": {"device": "cuda", "name": "kaggle_phase1"},
    "phase1": {"calculator": "mace", "mace_model": MACE_MODEL, "mace_dtype": MACE_DTYPE,
               "supercell": SUPERCELL, "structure_source": "materials_project"},
}, open(ROOT / "overlay_phase1.yaml", "w"))
'''),
    ("code", r'''
!python -m alumina_pipeline.orchestrator --config {PKG}/configs/pipeline.yaml --overlay {ROOT}/overlay_phase1.yaml --phase1-only --force-phase1 2>&1 | grep -v Warning
'''),
    ("code", r'''
t = json.loads((ROOT / "data/phase1_target.json").read_text())
for k in ["source","calculator","n_atoms_slab","surface_energy","e_break_o","e_break_o_kj_mol"]:
    print(f"{k:>18}: {t[k]}")
if not 0.5 < t["surface_energy"] < 4.0: print("[!] gamma fuera de ~1-3 J/m2")
if not 3.0 < t["e_break_o"] < 10.0:     print("[!] E_break fuera de ~4-8 eV")
'''),
    ("markdown", r'''
## Que guardar

| Archivo | Para que | Imprescindible |
|---|---|---|
| `alumina/data/phase1_target.json` | E_break, gamma: lo lee el nb 03 | **Si** |
| `alumina/data/al2o3_slab.extxyz` | Superficie relajada: chequeo MACE del 03 | **Si** |
| `alumina/runs/kaggle_phase1_*/` | Config efectiva (trazabilidad) | Recomendado |

En el nb 03: Add Input -> Notebook Output -> este notebook.
'''),
])

# ---------------------------------------------------------------- 02
nb("02_train_chemprop.ipynb", [
    ("markdown", r'''
# 02 - Fase 3 (preparacion): entrenar Chemprop (DG de solvatacion)

Modelo de **2 componentes**: `(solvente, soluto) -> DG_solv [kcal/mol]`. Se entrena una vez; el nb 03 lo usa en cada ronda.

**Elige la fuente de datos abajo (`DATA_SOURCE`):**
- `"freesolv"` (por defecto, cero setup): baja FreeSolv y arma un CSV de arranque. **Solo agua como solvente** -> sirve para verificar que todo el pipeline corre, pero el modelo NO aprende a distinguir solventes. Dataset de cableado, no de descubrimiento.
- `"zenodo"` (recomendado para ciencia real): baja **CombiSolv-Exp** (Vermeire & Green 2021, ~10k pares, muchos solventes) desde Zenodo. Kaggle tiene internet abierto, asi que se descarga aqui mismo.
- `"upload"`: tu propio CSV subido como Kaggle Dataset (Add Input). Lo mejor si tienes datos de ILs/DES.

Casi ningun dataset publico tiene iones metal-oxo ni muchos ILs/DES; la celda de cobertura lo cuantifica para que sepas cuanto fiarte.

**Settings:** GPU · Internet On · (si `upload`) Add Input -> tu dataset. Save & Run All (Commit).
'''),
    SETUP, INSTALL,
    ("code", r'''
DATA_SOURCE = "freesolv"    # "freesolv" | "zenodo" | "upload"
EPOCHS, ENSEMBLE = 100, 3

# Solo para "upload": ajusta a los nombres de columna de TU csv
CSV_PATTERN = "*.csv"
SOLVENT_COL, SOLUTE_COL, TARGET_COL, TARGET_UNITS = "solvent", "solute", "dG", "kcal/mol"

# Solo para "zenodo": record de la coleccion SolProp (contiene CombiSolv-Exp)
ZENODO_RECORD = "5970538"
'''),
    ("code", r'''
import re, io, zipfile, urllib.request, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def load_freesolv():
    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    txt = urllib.request.urlopen(url, timeout=60).read().decode()
    rows = []
    for ln in txt.splitlines():
        if ln.startswith("#") or not ln.strip(): continue
        p = [x.strip() for x in ln.split(";")]
        if len(p) >= 5: rows.append({"solvent": "O", "solute": p[1], "dG": float(p[3])})
    return pd.DataFrame(rows)   # agua (O) como unico solvente

def load_zenodo(record):
    api = f"https://zenodo.org/api/records/{record}"
    files = json.load(urllib.request.urlopen(api, timeout=60))["files"]
    print("Archivos en el record:")
    for f in files: print(f"  {f['size']:>10} {f['key']}")
    # busca un archivo cuyo nombre sugiera CombiSolv-Exp
    pick = next((f for f in files if re.search(r"combisolv.*exp", f["key"], re.I)), None) \
        or next((f for f in files if f["key"].lower().endswith((".csv", ".zip"))), None)
    if pick is None: raise RuntimeError("No encontre un CSV/zip; revisa la lista de arriba")
    print("Descargando:", pick["key"])
    raw = urllib.request.urlopen(pick["links"]["self"], timeout=300).read()
    if pick["key"].lower().endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        print("  CSV dentro del zip:", name); raw = z.read(name)
    d = pd.read_csv(io.BytesIO(raw))
    print("Columnas:", list(d.columns))   # revisa y renombra si hiciera falta
    cols = {c.lower(): c for c in d.columns}
    sv = cols.get("solvent") or cols.get("solvent_smiles")
    su = cols.get("solute") or cols.get("solute_smiles")
    tg = cols.get("dgsolv") or cols.get("dg") or cols.get("dgsolv_kcal/mol") or cols.get("target")
    return d.rename(columns={sv: "solvent", su: "solute", tg: "dG"})[["solvent", "solute", "dG"]]

def load_upload():
    hits = ''' + inputs("{CSV_PATTERN}").replace("{CSV_PATTERN}", '" + CSV_PATTERN + "') + r'''
    if not hits: raise FileNotFoundError("Sin CSV: Add Input -> tu dataset")
    d = pd.read_csv(hits[0]).rename(columns={SOLVENT_COL: "solvent", SOLUTE_COL: "solute", TARGET_COL: "dG"})
    if TARGET_UNITS == "kJ/mol": d["dG"] = d["dG"] / 4.184
    return d[["solvent", "solute", "dG"]]

df = {"freesolv": load_freesolv, "zenodo": lambda: load_zenodo(ZENODO_RECORD), "upload": load_upload}[DATA_SOURCE]()
print(f"\nFuente '{DATA_SOURCE}': {len(df)} filas crudas")
'''),
    ("code", r'''
# Limpieza + cobertura del dominio
def canon(s):
    m = Chem.MolFromSmiles(str(s));  return Chem.MolToSmiles(m) if m else None
n0 = len(df)
df = df.dropna(subset=["solvent", "solute", "dG"]).copy()
df["dG"] = pd.to_numeric(df["dG"], errors="coerce")
df["solvent"], df["solute"] = df["solvent"].map(canon), df["solute"].map(canon)
df = df.dropna()
print(f"Filas validas: {len(df)}/{n0} | solventes unicos: {df.solvent.nunique()} | solutos unicos: {df.solute.nunique()}")

ionic = df.solvent.map(lambda s: "." in s or "+" in s).mean()
print(f"Cobertura -> solventes ionicos/mezclas: {ionic:.1%}")
if df.solvent.nunique() == 1:
    print("[!] Un solo solvente: el modelo NO podra distinguir solventes (dataset de cableado).")
elif ionic < 0.01:
    print("[!] Casi no hay solventes ionicos: las predicciones para ILs/DES seran extrapolaciones.")
train_csv = ROOT / "data/solvation_train.csv"; df.to_csv(train_csv, index=False)
'''),
    ("code", r'''
!chemprop train --data-path {train_csv} --smiles-columns solvent solute --target-columns dG \
  --task-type regression --epochs {EPOCHS} --ensemble-size {ENSEMBLE} \
  --output-dir {ROOT}/models/chemprop_solvation 2>&1 | grep -viE "warning|it/s" | tail -n 40
'''),
    ("code", r'''
# Chequeo de cordura: agua en agua ~ -6.3 kcal/mol
test = pd.DataFrame({"solvent":["O","CO","O"], "solute":["O","O","[OH-]"],
                     "nota":["agua/agua ~-6.3","agua/metanol","OH-/agua"]})
test[["solvent","solute"]].to_csv("/tmp/test.csv", index=False)
!chemprop predict --test-path /tmp/test.csv --preds-path /tmp/tp.csv --model-paths {ROOT}/models/chemprop_solvation --smiles-columns solvent solute 2>&1 | tail -n 1
tp = pd.read_csv("/tmp/tp.csv")
test["pred_kcal_mol"] = tp[[c for c in tp.columns if c not in ("solvent","solute")][0]].values
test
'''),
    ("markdown", r'''
## Que guardar

| Archivo | Para que | Imprescindible |
|---|---|---|
| `alumina/models/chemprop_solvation/` (carpeta) | El modelo que usa el nb 03 | **Si** |
| RMSE/MAE de test (impresos arriba) | Cuanto confiar en DG | **Si** (anotalos) |
| `alumina/data/solvation_train.csv` | Reproducibilidad | Recomendado |

En el nb 03: Add Input -> Notebook Output -> este notebook.
'''),
])

# ---------------------------------------------------------------- 03
nb("03_active_learning_loop.ipynb", [
    ("markdown", r'''
# 03 - Fases 2+3+4: bucle de aprendizaje activo

REINVENT genera solventes -> BoTorch elige los mas informativos -> Chemprop evalua DG_extract -> el GP se reajusta y premia a REINVENT en la ronda siguiente.

**Add Input:** (1) salida del nb **01**, (2) salida del nb **02**, (3) Dataset con el prior de REINVENT (`reinvent.prior`, del Zenodo del README de REINVENT4), (4) *para reanudar*, la salida de una version previa de ESTE notebook (se detecta sola).

**Primera vez:** `GENERATOR="enumerator"`, `EVALUATOR="surrogate"` para verificar que corre en ~2 min; luego cambia a `reinvent`/`chemprop`.

**Settings:** GPU T4 basta · Internet On · Save & Run All (Commit) corre hasta 12 h sin navegador.
'''),
    SETUP, INSTALL,
    ("code", r'''
GENERATOR      = "reinvent"    # reinvent | enumerator
EVALUATOR      = "chemprop"    # chemprop | surrogate
N_ROUNDS       = 10            # TOTAL (al reanudar, incluye las ya hechas)
BATCH          = 8
RL_STEPS       = 50
EXPLICIT_CHECK = False         # chequeo MACE solvente+superficie del top-3 (lento)
'''),
    ("code", r'''
%%bash -s "$GENERATOR"
# REINVENT 4 en venv aislado (solo si GENERATOR == reinvent)
if [ "$1" != "reinvent" ]; then echo "GENERATOR=$1 -> se omite REINVENT"; exit 0; fi
set -e
if [ ! -x /tmp/reinvent-venv/bin/reinvent ]; then
  python -m venv --system-site-packages /tmp/reinvent-venv
  git clone -q --depth 1 https://github.com/MolecularAI/REINVENT4.git /tmp/REINVENT4
  /tmp/reinvent-venv/bin/pip install -q /tmp/REINVENT4 || true
  if ! /tmp/reinvent-venv/bin/reinvent --help >/dev/null 2>&1; then
    cd /tmp/REINVENT4 && /tmp/reinvent-venv/bin/python install.py cu124
  fi
fi
/tmp/reinvent-venv/bin/reinvent --help | head -n 2
'''),
    ("code", r'''
# Copiar inputs: salidas 01/02, prior y (si hay) corrida previa para reanudar
def first(pat, must=None):
    hits = [h for h in ''' + inputs("{pat}").replace("{pat}", '" + pat + "') + r''' if must is None or must in h]
    return Path(hits[0]) if hits else None

p1, slab = first("phase1_target.json", "/data/"), first("al2o3_slab.extxyz")
if p1 and slab:
    shutil.copy(p1, ROOT/"data/phase1_target.json"); shutil.copy(slab, ROOT/"data/al2o3_slab.extxyz")
    print("ok Fase 1:", p1)
else:
    print("[!] Sin nb 01: la Fase 1 se calcula aqui (minutos en GPU)")

cp = first("chemprop_solvation")
if cp and cp.is_dir():
    shutil.copytree(cp, ROOT/"models/chemprop_solvation", dirs_exist_ok=True); print("ok Chemprop:", cp)
elif EVALUATOR == "chemprop":
    raise FileNotFoundError("Falta nb 02 (chemprop_solvation/) o usa EVALUATOR='surrogate'")

prior = first("*.prior")
if prior:
    shutil.copy(prior, ROOT/"priors/reinvent.prior"); print("ok Prior:", prior)
elif GENERATOR == "reinvent":
    raise FileNotFoundError("Falta reinvent.prior (Add Input) o usa GENERATOR='enumerator'")

RESUME_DIR = None
prev = first("history.csv", "/runs/")
if prev:
    RESUME_DIR = ROOT/"runs"/prev.parent.name
    shutil.copytree(prev.parent, RESUME_DIR, dirs_exist_ok=True); print("ok Reanudando:", RESUME_DIR.name)
'''),
    ("code", r'''
import yaml
yaml.safe_dump({
    "run": {"device": "cuda", "name": "kaggle_loop"},
    "phase1": {"calculator": "mace"},
    "phase2": {"backend": GENERATOR,
               "reinvent": {"executable": "/tmp/reinvent-venv/bin/reinvent",
                            "prior_file": "priors/reinvent.prior", "steps_per_round": RL_STEPS}},
    "phase3": {"backend": EVALUATOR, "explicit_interface": {"enabled": EXPLICIT_CHECK}},
    "phase4": {"batch_size": BATCH},
}, open(ROOT/"overlay_loop.yaml", "w"))
resume_flag = f"--resume {RESUME_DIR}" if RESUME_DIR else ""
print(resume_flag or "Corrida nueva")
'''),
    ("code", r'''
!python -m alumina_pipeline.orchestrator --config {PKG}/configs/pipeline.yaml --overlay {ROOT}/overlay_loop.yaml --rounds {N_ROUNDS} {resume_flag} 2>&1 | grep -v Warning | tee -a {ROOT}/loop.log
'''),
    ("code", r'''
import pandas as pd
run = max((ROOT/"runs").glob("kaggle_loop_*"), key=lambda p: p.stat().st_mtime)
print("Corrida:", run.name)
rounds = pd.DataFrame(json.loads((run/"rounds.json").read_text()))
display(rounds[["round","n_evaluated","n_success","n_lowers_barrier","best_dG_extract_kJ_mol","best_smiles"]])
display(pd.read_csv(run/"top_candidates.csv"))
'''),
    ("markdown", r'''
## Que guardar

Todo queda en `alumina/runs/kaggle_loop_<fecha>/`:

| Archivo | Para que | Imprescindible |
|---|---|---|
| `history.csv` | Cada candidato evaluado, con DG y motivo de fallo | **Si** (es el resultado) |
| `top_candidates.csv` | Los 10 mejores solventes | **Si** |
| `rounds.json` | Progreso y modos de fallo por ronda | **Si** |
| `bo_state.pt` + `reinvent/agent_round*.chkpt` | Estado para **reanudar** otra sesion | **Si**, si continuaras |
| `reinvent/*.csv`, `tb_logs/` | Historial de REINVENT, TensorBoard | Recomendado |

**Para reanudar:** Add Input -> Notebook Output -> esta corrida, sube `N_ROUNDS` y Commit de nuevo.
'''),
])
