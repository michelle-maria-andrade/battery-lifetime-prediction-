# Battery state-of-health estimation

Estimates lithium-ion battery state of health (SoH) from charge-phase features, using classical machine learning, and serves the result in a local dashboard. Trained and validated on four NASA batteries and four CALCE cells (leave-one-battery-out).

## Setup

Run the setup script first. It creates `.venv` with Python 3.13 and installs the pinned packages from `requirements.txt`.

Linux / macOS:
```bash
bash setup.sh
source .venv/bin/activate
```

Windows:
```bat
setup.bat
.venv\Scripts\activate
```

Python 3.13 must already be installed.

## Data

Place the datasets under `data/`:

```
data/NASA/B0005.mat, B0006.mat, B0007.mat, B0018.mat
data/CALCE/CS2_35/ ... CS2_38/   (.xlsx files)
```

## Usage

Train the final model (writes `models/final_pipeline.joblib` and `models/model_card.json`):
```bash
python src/train_final.py
```

Run the dashboard:
```bash
streamlit run app/streamlit_app.py
```

Experiments (all five models, four training arms, sensitivity checks): open `src/battery_soh_loo.ipynb` and run all cells, selecting the `.venv` kernel.

## Layout

```
src/features.py            shared loading, feature and interval logic
src/train_final.py         trains and saves the final model
src/battery_soh_loo.ipynb  experiments and results
app/streamlit_app.py       dashboard
models/                    saved pipeline and model card
reports/figures/           figures from the notebook
```

## Limitations

Trained on eight laboratory cells of two cell types; it does not generalise to arbitrary cells. No remaining-useful-life forecast is provided; the dashboard reports where estimated SoH crosses 0.80.