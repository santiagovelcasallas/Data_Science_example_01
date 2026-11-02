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
        raise RuntimeError("git clone fallo: " + r.stderr[-300:].replace(token or "\0", "***"))
print("Codigo en", PKG)

import re, sys
NOISE = re.compile(r"warn|it/s|Tip:|^\s*$|cuequivariance|torch\.load", re.I)

def sh(cmd, log, show=lambda line: not NOISE.search(line)):
    """Ejecuta un comando mostrando su progreso. Si falla, DETIENE el notebook
    mostrando las ultimas lineas del error (nada de fallos silenciosos)."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    tail = []
    with open(log, "w") as fh, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, bufsize=1) as p:
        for line in p.stdout:
            fh.write(line)
            tail = (tail + [line])[-60:]
            if show(line):
                print(line, end="", flush=True)
    if p.returncode != 0:
        print("\n========== ERROR: ultimas lineas ==========\n" + "".join(tail))
        raise RuntimeError(f"El comando fallo (codigo {p.returncode}). Log completo: {log}")
''')

INSTALL = ("code", r'''
# ==== Instalacion (~3-6 min). Se fija la version de torch de Kaggle para no romper CUDA ====
torch_pin = [l for l in subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                       text=True).stdout.splitlines() if l.startswith("torch==")]
Path("/tmp/constraints.txt").write_text("\n".join(torch_pin) + "\n")
print("torch fijado:", torch_pin)
sh([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/constraints.txt",
    "-r", PKG / "kaggle/requirements-kaggle.txt"], "/tmp/pip.log")
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", PKG], "/tmp/pip_pkg.log")
sh([sys.executable, "-c", "import torch,rdkit,ase,botorch,chemprop,alumina_pipeline; "
    "print('torch', torch.__version__, '| chemprop', chemprop.__version__, '| GPU:', torch.cuda.is_available())"],
   "/tmp/check.log")
CHEMPROP = shutil.which("chemprop") or str(Path(sys.executable).parent / "chemprop")
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
sh([sys.executable, "-m", "alumina_pipeline.orchestrator", "--config", PKG / "configs/pipeline.yaml",
    "--overlay", ROOT / "overlay_phase1.yaml", "--phase1-only", "--force-phase1"], ROOT / "phase1.log")
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

**Fuente de datos (`DATA_SOURCE`):**
- `"zenodo"` (por defecto): baja la coleccion **SolProp** (Vermeire & Green; incluye CombiSolv-Exp, muchos solventes) desde Zenodo, ~270 MB, y elige el CSV de energia libre de solvatacion **mirando sus columnas**.
- `"upload"`: tu propio CSV (Add Input). Lo mejor si consigues datos de ILs/DES.
- `"freesolv"`: solo agua. Unicamente para depurar; no sirve para elegir solventes.

Antes del entrenamiento largo hay un **pre-chequeo corto** (~1 min): si algo esta mal, el notebook se detiene ahi mostrando el error real.

**Settings:** GPU · Internet On · Save & Run All (Commit). Tiempo total aprox.: 25-45 min.
'''),
    SETUP, INSTALL,
    ("code", r'''
DATA_SOURCE = "zenodo"      # "zenodo" | "upload" | "freesolv"
EPOCHS, ENSEMBLE = 100, 3

# Solo para "upload": nombres de columna de TU csv
CSV_PATTERN = "*.csv"
SOLVENT_COL, SOLUTE_COL, TARGET_COL, TARGET_UNITS = "solvent", "solute", "dG", "kcal/mol"

# Solo para "zenodo"
ZENODO_RECORD = "5970538"   # coleccion SolProp v1.2
ZENODO_CSV    = None        # None = automatico. Para forzar uno: parte de su nombre, p.ej. "CombiSolv"
'''),
    ("code", r'''
import io, zipfile, urllib.request, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

def load_freesolv():
    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    rows = []
    for ln in urllib.request.urlopen(url, timeout=60).read().decode().splitlines():
        p = [x.strip() for x in ln.split(";")]
        if not ln.startswith("#") and len(p) >= 5:
            rows.append({"solvent": "O", "solute": p[1], "dG": float(p[3])})
    return pd.DataFrame(rows)

BAD_TARGET = ("std", "err", "unc", "count", "log", "hsolv", "enthalp")

def find_cols(columns):
    """(col_solvente, col_soluto, col_dG) o None.
    Prefiere columnas SMILES (no nombres ni InChI) y el PROMEDIO de dG (no su desviacion)."""
    low = [(c, c.lower().replace(" ", "")) for c in columns]
    sv = next((c for c, l in low if "solvent" in l and "smiles" in l), None) \
         or next((c for c, l in low if l == "solvent"), None)
    su = next((c for c, l in low if "solute" in l and "smiles" in l), None) \
         or next((c for c, l in low if l == "solute"), None)
    g = [c for c, l in low if ("gsolv" in l or l in ("dg", "deltag")) and not any(b in l for b in BAD_TARGET)]
    g.sort(key=lambda c: ("avg" not in c.lower() and "mean" not in c.lower(), len(c)))
    return (sv, su, g[0]) if sv and su and g else None

def load_zenodo(record):
    cache = ROOT / "data" / f"zenodo_{record}.zip"
    if not cache.exists():
        files = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60))["files"]
        f = next(f for f in files if f["key"].lower().endswith((".zip", ".csv")))
        print("Descargando", f["key"], f"({f['size']/1e6:.0f} MB)...", flush=True)
        urllib.request.urlretrieve(f["links"]["self"], cache)
    z = zipfile.ZipFile(cache)
    usable = []
    print("CSVs dentro del zip:")
    for n in [n for n in z.namelist() if n.lower().endswith(".csv")]:
        try:
            cols = list(pd.read_csv(z.open(n), nrows=0).columns)
        except Exception as e:
            print(f"  [ilegible] {n}: {e}"); continue
        hit = find_cols(cols)
        print(f"  {'[USABLE]' if hit else '[  no  ]'} {n}  {cols}")
        if hit and (ZENODO_CSV is None or ZENODO_CSV.lower() in n.lower()):
            usable.append((n, hit))
    if not usable:
        raise RuntimeError("Ningun CSV tiene (solvente SMILES, soluto SMILES, dGsolv). "
                           "Revisa la lista de arriba y fija ZENODO_CSV.")
    # preferir datos experimentales ('exp') sobre calculados ('qm')
    usable.sort(key=lambda u: ("exp" not in u[0].lower(), "qm" in u[0].lower()))
    name, (sv, su, tg) = usable[0]
    print(f"\nELEGIDO: {name}\n  solvente <- {sv!r} | soluto <- {su!r} | dG <- {tg!r}")
    d = pd.read_csv(z.open(name), usecols=[sv, su, tg])[[sv, su, tg]]
    d.columns = ["solvent", "solute", "dG"]          # sin renombrar: imposible duplicar columnas
    if "kj" in tg.lower():
        d["dG"] = pd.to_numeric(d["dG"], errors="coerce") / 4.184
    return d

def load_upload():
    hits = ''' + inputs("{CSV_PATTERN}").replace("{CSV_PATTERN}", '" + CSV_PATTERN + "') + r'''
    if not hits: raise FileNotFoundError("Sin CSV: Add Input -> tu dataset")
    d = pd.read_csv(hits[0], usecols=[SOLVENT_COL, SOLUTE_COL, TARGET_COL])[[SOLVENT_COL, SOLUTE_COL, TARGET_COL]]
    d.columns = ["solvent", "solute", "dG"]
    if TARGET_UNITS == "kJ/mol": d["dG"] = pd.to_numeric(d["dG"], errors="coerce") / 4.184
    return d

df = {"zenodo": lambda: load_zenodo(ZENODO_RECORD), "upload": load_upload, "freesolv": load_freesolv}[DATA_SOURCE]()
print(f"\nFuente '{DATA_SOURCE}': {len(df)} filas crudas")
display(df.head())
'''),
    ("code", r'''
# ==== Limpieza + cobertura del dominio ====
def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m else None

n0 = len(df)
df["dG"] = pd.to_numeric(df["dG"], errors="coerce")
df["solvent"], df["solute"] = df["solvent"].map(canon), df["solute"].map(canon)
df = df.dropna().reset_index(drop=True)
print(f"Filas validas: {len(df)}/{n0} | solventes unicos: {df.solvent.nunique()} | solutos unicos: {df.solute.nunique()}")
print(f"dG [kcal/mol]: min {df.dG.min():.1f} | mediana {df.dG.median():.1f} | max {df.dG.max():.1f}")
if len(df) < 0.5 * n0:
    raise RuntimeError("Mas de la mitad de las filas no son SMILES validos: columnas equivocadas.")

ionic = df.solvent.map(lambda s: "." in s or "+" in s or "-" in s).mean()
print(f"COBERTURA -> solventes ionicos/mezclas: {ionic:.1%} de las filas")
if df.solvent.nunique() == 1:
    print("[!] Un solo solvente: el modelo NO puede distinguir solventes.")
elif ionic < 0.01:
    print("[!] Casi no hay solventes ionicos: las predicciones para ILs/DES seran extrapolaciones.")

train_csv = ROOT / "data/solvation_train.csv"
df.to_csv(train_csv, index=False)
'''),
    ("code", r'''
# ==== Pre-chequeo: 3 epocas (Chemprop exige > 2 de calentamiento) con 300 filas. Falla AQUI en ~1 min si algo esta mal ====
pre_csv, pre_out = ROOT / "data/preflight.csv", Path("/tmp/preflight_model")
df.sample(min(300, len(df)), random_state=0).to_csv(pre_csv, index=False)
shutil.rmtree(pre_out, ignore_errors=True)
sh([CHEMPROP, "train", "--data-path", pre_csv, "--smiles-columns", "solvent", "solute",
    "--target-columns", "dG", "--task-type", "regression", "--epochs", 3, "--output-dir", pre_out],
   "/tmp/preflight.log", show=lambda l: "Error" in l or "Traceback" in l)
print("Pre-chequeo OK: datos y Chemprop funcionan.")
'''),
    ("code", r'''
# ==== Entrenamiento completo (en GPU ~15-30 min con ~10k filas) ====
MODEL_DIR = ROOT / "models/chemprop_solvation"
shutil.rmtree(MODEL_DIR, ignore_errors=True)
sh([CHEMPROP, "train", "--data-path", train_csv, "--smiles-columns", "solvent", "solute",
    "--target-columns", "dG", "--task-type", "regression", "--epochs", EPOCHS,
    "--ensemble-size", ENSEMBLE, "--metrics", "rmse", "mae", "r2", "--output-dir", MODEL_DIR],
   ROOT / "chemprop_train.log", show=lambda l: "INFO" in l or "Error" in l)
print("\nMETRICAS DE TEST (anotalas):")
print("".join(l for l in open(ROOT / "chemprop_train.log") if "chemprop.cli.train - test/" in l))
'''),
    ("code", r'''
# ==== Chequeo de cordura con valores conocidos ====
test = pd.DataFrame({"solvent": ["O", "CO", "O"], "solute": ["O", "O", "[OH-]"],
                     "referencia": ["agua en agua: exp. ~ -6.3", "agua en metanol",
                                    "OH- en agua: exp. ~ -100 (casi no hay iones en los datos)"]})
test[["solvent", "solute"]].to_csv("/tmp/test.csv", index=False)
sh([CHEMPROP, "predict", "--test-path", "/tmp/test.csv", "--preds-path", "/tmp/tp.csv",
    "--model-paths", MODEL_DIR, "--smiles-columns", "solvent", "solute"],
   "/tmp/predict.log", show=lambda l: "Error" in l or "Traceback" in l)
test["pred_kcal_mol"] = pd.read_csv("/tmp/tp.csv")["dG"].round(2).values
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
sh([sys.executable, "-m", "alumina_pipeline.orchestrator", "--config", PKG / "configs/pipeline.yaml",
    "--overlay", ROOT / "overlay_loop.yaml", "--rounds", N_ROUNDS, *resume_flag.split()], ROOT / "loop.log")
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
