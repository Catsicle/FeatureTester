# Advanced Formula Tester

A Streamlit-based laboratory for experimenting with technical indicators, parameterized formulas, and correlation analytics against CSV datasets. It supports batching indicator generation, constant optimisation, downloadable plots (JSON/CSV/PNG/JPG), and automatic persistence of engineered features between sessions.

## Features

- Upload any CSV and limit processing to the first *N* rows (top-N, not random) for rapid iteration.
- Queue multiple technical indicators (powered by `pandas-ta`) and generate them in one batch; derived features persist across sessions via `<uploaded_name>_features.csv`.
- Evaluate formulas using coefficients (`a`, `b`, `c`, `d`, `k`) with safe divide-by-zero handling.
- Optimise any subset of coefficients sequentially, with correlation tracking and optimisation history logging.
- Plot target vs. derived series or any group of numeric columns, then download the data, Vega/Vega-Lite JSON spec, or rendered PNG/JPG images.
- Compute correlation matrices and pairwise statistics for selected columns.
- Searchable selectors for targets, indicator sources, and plotting columns.

## Quick Start

1. **Clone or copy the repository**
   ```powershell
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment (recommended)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```powershell
   python -m streamlit run .\formula_testing_tool.py
   ```

5. **Upload data and explore**
   - Choose a CSV (ensure numeric columns exist for indicators and formulas).
   - Optionally adjust the "Rows to use" slider to work with the first *N* rows.
   - Queue indicators, generate them, craft formulas, and review correlation metrics.
   - Download charts as JSON/CSV/PNG/JPG for reporting or sharing.

## Dependency Notes

- `pandas-ta` supplies technical indicators; requires pandas ≥ 1.5.
- `altair` renders interactive charts. The app enables the `vegafusion` transformer automatically; ensure `vegafusion` and `vl-convert-python>=1.6` are installed (both listed in `requirements.txt`). Without Altair, it falls back to Streamlit line charts.
- `matplotlib` enables exporting plots as PNG/JPG. If unavailable, the app still offers JSON/CSV downloads.
- `vegafusion` lifts Altair's default 5k-row cap; included in `requirements.txt`.

## Persisted Indicators

Generated indicators are stored in `./<uploaded_filename>_features.csv`. Re-uploading the same dataset will automatically merge those columns back into the working DataFrame, keeping your engineered features available between sessions.

If you prefer a clean slate, delete the corresponding `*_features.csv` file before restarting the app.

## Publishing to GitHub

1. Initialise the local repository (if you have not already):
   ```powershell
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a new repository on GitHub and copy the remote URL.

3. Connect and push:
   ```powershell
   git remote add origin <git-remote-url>
   git push -u origin main
   ```

4. Update the README with project-specific context (datasets, screenshots, etc.) before sharing.

## Additional Scripts

The workspace also contains supporting utilities (e.g., `Extract.py`, `browser_scraper.py`). Review each script for usage instructions or integrate them into your documentation as needed.

## Support

If dependencies fail to install, ensure you are using Python 3.10+ and upgrade pip:
```powershell
python -m pip install --upgrade pip
```
Then rerun the installation step.
