# Datos de arranque

`solvation_starter_freesolv.csv` — CSV listo para el notebook 02 (columnas `solvent`,
`solute`, `dG` en kcal/mol). Derivado de **FreeSolv v0.52** (Mobley et al.), 642 valores
experimentales de energía libre de **hidratación**. Licencia CC-BY; cita FreeSolv si lo usas.

⚠️ **Solo agua como solvente.** Entrena el pipeline de punta a punta con datos reales, pero
el modelo no puede distinguir solventes. Para descubrimiento real usa `DATA_SOURCE="zenodo"`
(CombiSolv-Exp, multi-solvente) o tus propios datos de ILs/DES en el notebook 02.

Fuente: https://github.com/MobleyLab/FreeSolv
