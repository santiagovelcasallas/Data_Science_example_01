# Correr el pipeline en Kaggle

Tres notebooks que encadenan las 4 fases. Suben el código desde este repo, así que
no hay que empaquetar nada: solo subir los `.ipynb` a Kaggle (o importarlos desde GitHub).

| Notebook | Fase(s) | Corre | Tiempo aprox. (GPU) |
|---|---|---|---|
| `01_target_al2o3.ipynb` | 1 (objetivo Al₂O₃, MACE) | una vez | 10–40 min (P100) |
| `02_train_chemprop.ipynb` | 3 — entrenar Chemprop ΔG | una vez | minutos–1 h |
| `03_active_learning_loop.ipynb` | 2 + 3 + 4 (bucle) | por experimento | 1–4 h / 10 rondas (T4) |

## Orden

1. **01** → Commit. Produce `phase1_target.json` + `al2o3_slab.extxyz`.
2. **02** → Add Input: tu dataset de solvatación (CSV). Commit. Produce `chemprop_solvation/`.
3. **03** → Add Input: salidas de 01 y 02, + un dataset con `reinvent.prior`. Commit.

Cada notebook lista al final **qué guardar** de su salida.

## Ajustes del entorno

- **Accelerator:** GPU. P100 para el 01 (float64); T4 basta para el 03.
- **Internet:** On (Settings). Requiere cuenta verificada por teléfono.
- **Secrets** (Add-ons → Secrets): `MP_API_KEY` opcional (Materials Project);
  `GITHUB_TOKEN` solo si este repo es privado.
- El código va a `/tmp` (efímero); el estado a `/kaggle/working/alumina` (se guarda con el Commit).

## Insumos que aportas tú

- **Prior de REINVENT** (`reinvent.prior`): del Zenodo enlazado en el README de REINVENT4. Súbelo como Kaggle Dataset.
- **Datos de solvatación** para Chemprop: CombiSolv-Exp, MNSol o propios. Súbelos como Kaggle Dataset.

## Reanudar una corrida cortada (12 h de límite)

El notebook 03 detecta la salida de una corrida previa (Add Input → Notebook Output
de esa versión), recarga historial + estado de BoTorch + checkpoint de REINVENT, y
sigue desde la última ronda. Sube `N_ROUNDS` y vuelve a hacer Commit.
