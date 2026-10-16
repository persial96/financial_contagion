# Financial Contagion as a Self-Exciting Point Process

Code to reproduce the results of the paper by Luca Persia (2026), Università della Svizzera italiana.

## Structure
- `src/python/01_data_extraction.ipynb`: CRSP sector ETF returns (WRDS), daily stress events and VIX.
- `src/matlab/MAIN_CONTAGION.m`: Poisson, univariate and multivariate Hawkes models (Tables 1-3, Figures 1-4).
- `src/python/02_vix_dcc_benchmark.ipynb`: VIX calibration and DCC-GARCH benchmark (Table 4, Figures 5-6).
- `data/results`: exported tables and figure data.

## Run order
1. `01_data_extraction.ipynb` (requires a WRDS subscription with CRSP access).
2. `MAIN_CONTAGION.m` with `WINDOW = "GFC"` and then `WINDOW = "COVID"` (MATLAB with the Optimization and Statistics toolboxes).
3. `02_vix_dcc_benchmark.ipynb`.

CRSP data are proprietary and are not included in the repository.
