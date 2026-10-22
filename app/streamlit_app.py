"""Battery state-of-health dashboard.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import io
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import features as ft  # noqa: E402

DEMO_CSV = ROOT / "data" / "processed" / "cycle_features_nasa_calce.csv"
MODEL_PATH = ROOT / "models" / "final_pipeline.joblib"
CARD_PATH = ROOT / "models" / "model_card.json"
IMPORTANCE_PNG = ROOT / "reports" / "figures" / "permutation_importance.png"

UPLOAD_REQUIRED = ["cycle_no", "cc_time", "cv_time", "t40", "win_cap", "dis_time"]
UPLOAD_OPTIONAL = ["capacity"]
RATIO_COLS = ["cc_ratio", "cv_ratio", "t40_ratio", "win_ratio"]

C_EST, C_TRUE, C_EOL = "#1f6feb", "#e8710a", "#6b7280"

st.set_page_config(page_title="Battery SoH dashboard", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_PATH)
    card = json.load(open(CARD_PATH))
    return pipe, card


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO_CSV)


@st.cache_data
def read_upload(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def prepare_upload(df: pd.DataFrame):
    """Validate an uploaded table and add ratio features.

    Returns (table, warnings). Raises ValueError with a readable message.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in UPLOAD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected columns: " + ", ".join(UPLOAD_REQUIRED + UPLOAD_OPTIONAL)
            + " (capacity is optional)."
        )
    has_capacity = "capacity" in df.columns
    if not has_capacity:
        df["capacity"] = np.nan
    if "battery" not in df.columns:
        df["battery"] = "uploaded"

    num_cols = UPLOAD_REQUIRED + ["capacity"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    warnings = []
    n0 = len(df)
    df = df.sort_values("cycle_no")
    check = [c for c in UPLOAD_REQUIRED if c != "cycle_no"] + (["capacity"] if has_capacity else [])
    df = df.dropna(subset=["cycle_no"] + check)
    df = df[(df[check] > 0).all(axis=1)].reset_index(drop=True)
    if n0 - len(df):
        warnings.append(f"{n0 - len(df)} row(s) with empty, non-numeric or non-positive values were skipped.")

    if len(df) < ft.n_early:
        raise ValueError(
            f"Only {len(df)} usable row(s) found. At least {ft.n_early} are needed, because the "
            f"ratios use the battery's own first {ft.n_early} cycles as the early-life reference."
        )
    if df["cycle_no"].duplicated().any():
        raise ValueError("`cycle_no` contains duplicates. Use one row per discharge cycle.")

    table = ft.add_ratios(df)
    if not has_capacity:
        table["soh"] = np.nan
    return table, warnings


def example_csv() -> bytes:
    demo = load_demo()
    one = demo[demo["battery"] == "B0005"][["cycle_no", "capacity", "cc_time", "cv_time", "t40", "win_cap", "dis_time"]]
    return one.to_csv(index=False).encode()


# ---------------------------------------------------------------- plots
def gauge_figure(soh, lo, hi):
    """Half-circle gauge. SoH is clipped to 0..1 for drawing only."""
    v, lo_c, hi_c = (float(np.clip(x, 0, 1)) for x in (soh, lo, hi))
    ang = lambda x: 180 - 180 * x  # 1.0 -> 0 deg (right), 0.0 -> 180 deg (left)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.28, facecolor="#e5e7eb", edgecolor="none"))
    colour = "#2e9d57" if v >= 0.9 else ("#e0a100" if v >= ft.eol else "#d64545")
    ax.add_patch(Wedge((0, 0), 1.0, ang(v), 180, width=0.28, facecolor=colour, edgecolor="none"))
    ax.add_patch(Wedge((0, 0), 1.14, ang(hi_c), ang(lo_c), width=0.07, facecolor="#374151", edgecolor="none"))
    t = np.deg2rad(ang(ft.eol))  # end-of-life tick
    ax.plot([0.68 * np.cos(t), 1.0 * np.cos(t)], [0.68 * np.sin(t), 1.0 * np.sin(t)], color="black", lw=1.5)
    ax.text(1.06 * np.cos(t), 1.06 * np.sin(t) + 0.04, "80%", ha="right", va="bottom", fontsize=8)
    ax.text(0, 0.18, f"{soh * 100:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.02, "estimated SoH", ha="center", va="top", fontsize=9, color="#4b5563")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.12, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def curve_figure(table, mu, lo, hi, crossing, selected_cycle, show_true, title):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = table["cycle_no"].values
    ax.fill_between(x, lo, hi, color=C_EST, alpha=0.18, label="Calibrated 95% band")
    ax.plot(x, mu, color=C_EST, lw=1.6, label="Model estimate")
    if show_true:
        ax.plot(x, table["soh"].values, color=C_TRUE, lw=1.2, alpha=0.85, label="True SoH (from measured capacity)")
    ax.axhline(ft.eol, color=C_EOL, ls="--", lw=1.2, label="End-of-life threshold (0.80)")
    if not np.isnan(crossing):
        ax.axvline(crossing, color="#d64545", ls=":", lw=1.4, label=f"Estimate reaches 0.80 (cycle {int(crossing)})")
    ax.axvline(selected_cycle, color="black", lw=0.8, alpha=0.5, label="Selected cycle")
    ax.set_xlabel("Discharge cycle number")
    ax.set_ylabel("State of health (fraction of early-life capacity)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- page
pipe, card = load_model()
half = card["interval_half_width_soh"]
rmse = card["loo_rmse_by_dataset"]
cover = card["nested_coverage_by_dataset"]

st.title("Lithium-ion battery state-of-health estimator")
st.caption(
    "Estimates SoH (current capacity as a fraction of early-life capacity) from charging-phase "
    "features and one slice of the discharge, using an RBF support vector regressor."
)

# ---- sidebar
with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode", ["Demo battery", "Upload cycle table"])
    table, warnings, is_calce, show_true, label = None, [], 0, False, ""

    if mode == "Demo battery":
        demo = load_demo()
        names = list(ft.nasa_batteries + ft.calce_batteries)
        names = [n for n in names if n in set(demo["battery"])]
        battery = st.selectbox("Battery", names, format_func=lambda b: f"{b} ({ft.ds_of[b]}, in training data)")
        table = demo[demo["battery"] == battery].sort_values("cycle_no").reset_index(drop=True)
        is_calce = int(table["is_calce"].iloc[0])
        show_true, label = True, battery
    else:
        cell = st.radio("Which cell type does your data resemble?", ["NASA-type", "CALCE-type"])
        is_calce = 1 if cell == "CALCE-type" else 0
        up = st.file_uploader("CSV, one row per discharge cycle", type=["csv"])
        st.caption(
            "Required columns: " + ", ".join(UPLOAD_REQUIRED) + ". Optional: capacity (draws the true-SoH line), "
            "battery. Times in seconds, capacity in Ah. Include the first 5 cycles."
        )
        st.download_button("Download example file", example_csv(), "example_upload.csv", "text/csv")
        if up is not None:
            try:
                table, warnings = prepare_upload(read_upload(up.getvalue()))
                show_true = table["soh"].notna().any()
                label = up.name
            except ValueError as e:
                st.error(str(e))
            except Exception:
                st.error("That file could not be read as a CSV table. Please check the format and try again.")

# ---- main
if table is None:
    st.info("Upload a cycle table in the sidebar, or switch to a demo battery.")
else:
    for w in warnings:
        st.warning(w)

    cell_name = "CALCE" if is_calce else "NASA"
    mu, lo, hi = ft.predict_with_band(pipe, card, table, is_calce)
    crossing = ft.first_crossing(table["cycle_no"].values, mu)

    if mode == "Demo battery":
        st.warning(
            f"**{label} is included in the model's training data.** The saved model has seen this battery, "
            "so these estimates look better than they would on a new battery."
        )

    cycles = table["cycle_no"].astype(int).tolist()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("State of health at a chosen cycle")
        cyc = st.select_slider("Cycle", options=cycles, value=cycles[min(len(cycles) - 1, len(cycles) // 2)])
        i = cycles.index(cyc)
        fig = gauge_figure(mu[i], lo[i], hi[i])
        st.pyplot(fig)
        plt.close(fig)
        st.markdown(
            f"**{mu[i] * 100:.1f} percent**, 95 percent band {lo[i] * 100:.1f} to {hi[i] * 100:.1f} percent"
        )
        st.caption(
            f"Band half-width for {cell_name}-type cells: {half[cell_name] * 100:.2f} percentage points "
            f"(from the model card; estimated from only 3 other batteries, so approximate). Coverage on unseen "
            f"batteries was {cover[cell_name]:.3f} against a nominal 0.95. Leave-one-battery-out RMSE: "
            f"{rmse[cell_name]:.4f} SoH units."
        )

    with right:
        st.subheader("End-of-life detection")
        if np.isnan(crossing):
            st.metric("Estimated SoH first reaches 0.80", "not reached in this data")
        else:
            st.metric("Estimated SoH first reaches 0.80", f"cycle {int(crossing)}")
        st.caption("Detection from estimated SoH, not a forecast.")
        st.caption(
            "This marks the point where degradation has become visible in the estimate. It is not an early "
            "warning, and the app makes no remaining-life prediction. Across the project's batteries, the "
            "detected cycle landed on average within about 3 percent of the true life."
        )
        if mode == "Demo battery":
            st.caption(f"For reference, the true end of life of {label} in this data is cycle {int(table['eol_cycle'].iloc[0])}.")

    st.subheader("Degradation curve")
    fig = curve_figure(table, mu, lo, hi, crossing, cyc, show_true,
                       f"Estimated state of health vs cycle: {label} ({cell_name}-type model setting)")
    st.pyplot(fig)
    plt.close(fig)

# ---- reliance panel
st.subheader("Which features the model relies on overall")
c1, c2 = st.columns([3, 2])
with c1:
    if IMPORTANCE_PNG.exists():
        st.image(str(IMPORTANCE_PNG), caption="Permutation importance (increase in error when a feature is shuffled)")
    else:
        st.info("Importance chart not found at reports/figures/permutation_importance.png.")
with c2:
    st.write(
        "Shuffling `win_ratio` (capacity discharged between 3.8 V and 3.4 V) hurts the model most, followed by "
        "the cell-type flag, `cc_ratio` and `t40_ratio`. `cv_ratio` matters least."
    )
    st.write(
        "This is a global summary of the whole model. It does not explain why any single estimate came out as it did."
    )

# ---- limitations (always visible)
with st.expander("Limitations", expanded=True):
    st.markdown(
        """
1. **No remaining-useful-life forecast.** Extrapolating the SoH trend failed before late life, giving absurd forecasts. The app only shows where the *estimated* SoH crosses 0.80, which is detection once degradation is visible, not early warning.
2. **Eight laboratory cells, two cell types.** The model does not apply to arbitrary cells, other chemistries or real-world drive cycles.
3. **A reference is required.** Ratios use the battery's own first 5 cycles, so any data you upload must include those early cycles. You must also say which cell type your data resembles.
4. **Feature importance is global**, not an explanation of individual predictions.
5. **The 8 demo batteries are in the training data.** Estimates on them look better than on a new battery.
"""
    )
