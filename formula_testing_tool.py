from __future__ import annotations

import io
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:  # pragma: no cover - graceful downgrade
    alt = None

try:
    import pandas_ta as ta  # type: ignore
except ImportError:  # pragma: no cover - runtime dependency hint
    ta = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - graceful downgrade
    plt = None

st.set_page_config(page_title="Formula Tester", layout="wide")
st.title("Formula Tester")

if ta is None:
    st.error(
        "The pandas-ta package is required for technical indicators. "
        "Install it with `pip install pandas-ta` and rerun the app."
    )
    st.stop()

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "df" not in st.session_state:
    st.session_state.df = None
if "sample_size" not in st.session_state:
    st.session_state.sample_size = 100
if "applied_sample_size" not in st.session_state:
    st.session_state.applied_sample_size = None
if "added_features" not in st.session_state:
    st.session_state.added_features = []
if "feature_definitions" not in st.session_state:
    st.session_state.feature_definitions = []
if "k_value" not in st.session_state:
    st.session_state.k_value = 0.0
if "pending_coeff_updates" not in st.session_state:
    st.session_state.pending_coeff_updates = {}
if "optimization_log" not in st.session_state:
    st.session_state.optimization_log = []
if "features_file" not in st.session_state:
    st.session_state.features_file = None
if "persisted_features" not in st.session_state:
    st.session_state.persisted_features = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "pending_indicators" not in st.session_state:
    st.session_state.pending_indicators = []

if st.session_state.pending_coeff_updates:
    for const_name, const_value in list(st.session_state.pending_coeff_updates.items()):
        if const_name == "k":
            st.session_state["k_value"] = float(const_value)
        else:
            st.session_state[f"coef_{const_name}"] = float(const_value)
    st.session_state.pending_coeff_updates = {}


def append_optimization_log(message: str) -> None:
    """Keep a short rolling history of optimization results."""
    log = st.session_state.optimization_log
    log.append(message)
    st.session_state.optimization_log = log[-12:]


def update_available_features() -> None:
    """Refresh the list of available feature columns for display."""
    feature_names: List[str] = []
    for definition in st.session_state.feature_definitions:
        feature_names.extend(definition.get("columns", []))
    persisted = st.session_state.get("persisted_features")
    if isinstance(persisted, pd.DataFrame):
        feature_names.extend(list(persisted.columns))
    # Preserve insertion order while removing duplicates
    deduped = list(dict.fromkeys(feature_names))
    st.session_state.added_features = deduped


def sanitize_basename(name: Optional[str]) -> str:
    if not name:
        return "dataset"
    stem = Path(name).stem
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    return safe or "dataset"


def build_plot_images(dataframe: pd.DataFrame, columns: List[str], title: str) -> Tuple[Optional[bytes], Optional[bytes]]:
    if plt is None or dataframe.empty or not columns:
        return None, None
    clean_df = dataframe[columns].copy()
    if clean_df.dropna(how="all").empty:
        return None, None
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        clean_df.plot(ax=ax)
        ax.set_title(title or "Plot")
        ax.set_xlabel("Row Index")
        ax.set_ylabel("Value")
        ax.legend(loc="best")
        fig.tight_layout()
        buffer_png = io.BytesIO()
        fig.savefig(buffer_png, format="png", dpi=200)
        buffer_png.seek(0)
        buffer_jpg = io.BytesIO()
        fig.savefig(buffer_jpg, format="jpeg", dpi=200)
        buffer_jpg.seek(0)
        plt.close(fig)
        return buffer_png.getvalue(), buffer_jpg.getvalue()
    except Exception:
        plt.close("all")
        return None, None


def get_features_path() -> Path:
    stored = st.session_state.get("features_file")
    if stored:
        return Path(stored)
    return Path.cwd() / "generated_features.csv"


def load_persisted_features(path: Path, expected_len: int) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        saved = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - user feedback
        st.warning(f"Failed to load stored indicators: {exc}")
        return None
    if "__row_id" in saved.columns:
        saved = saved.set_index("__row_id")
    saved = saved.reindex(range(expected_len))
    return saved


def persist_feature_columns(full_df: pd.DataFrame, columns: List[str]) -> None:
    if not columns:
        return
    path = get_features_path()
    subset = full_df[columns].copy()
    subset.index = range(len(subset))
    persisted = st.session_state.get("persisted_features")
    if not isinstance(persisted, pd.DataFrame):
        persisted = pd.DataFrame(index=subset.index)
    else:
        persisted = persisted.reindex(subset.index)
    for col in columns:
        persisted[col] = subset[col]
    st.session_state.persisted_features = persisted
    output = persisted.reset_index().rename(columns={"index": "__row_id"})
    try:
        output.to_csv(path, index=False)
    except Exception as exc:  # pragma: no cover - user feedback
        st.warning(f"Failed to store generated indicators: {exc}")


def compute_indicator(
    df: pd.DataFrame,
    definition: Dict[str, Any],
    *,
    show_feedback: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply a single indicator definition to the dataframe."""
    indicator_type: str = definition.get("type", "")
    source_column: Optional[str] = definition.get("source")
    window_int = int(definition.get("window", 1))
    std_dev = definition.get("std_dev")
    lag_periods = int(definition.get("lag", 1))

    # Remove previously generated columns for idempotent re-computation
    prior_columns = definition.get("columns", [])
    if prior_columns:
        df = df.drop(columns=prior_columns, errors="ignore")

    if source_column and source_column not in df.columns:
        if show_feedback:
            st.sidebar.warning(
                f"Source column '{source_column}' not found. Skipping feature generation."
            )
        return df, []

    new_columns: List[str] = []

    try:
        if indicator_type == "Rolling Mean (SMA)":
            col_name = f"SMA_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).mean()
            new_columns.append(col_name)
        elif indicator_type == "EMA":
            col_name = f"EMA_{source_column}_{window_int}"
            result = ta.ema(df[source_column], length=window_int)
            if result is None:
                if show_feedback:
                    st.sidebar.error("Unable to compute EMA with the current settings.")
                return df, []
            df[col_name] = result
            new_columns.append(col_name)
        elif indicator_type == "Rolling Median":
            col_name = f"MEDIAN_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).median()
            new_columns.append(col_name)
        elif indicator_type == "Rolling Std Dev":
            col_name = f"STD_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).std(ddof=0)
            new_columns.append(col_name)
        elif indicator_type == "Rolling Std Dev (Sample)":
            col_name = f"STD_SAMPLE_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).std(ddof=1)
            new_columns.append(col_name)
        elif indicator_type == "Rolling Max":
            col_name = f"MAX_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).max()
            new_columns.append(col_name)
        elif indicator_type == "Rolling Min":
            col_name = f"MIN_{source_column}_{window_int}"
            df[col_name] = df[source_column].rolling(window=window_int, min_periods=1).min()
            new_columns.append(col_name)
        elif indicator_type == "Lag (T-n)":
            col_name = f"LAG_{source_column}_{lag_periods}"
            df[col_name] = df[source_column].shift(lag_periods)
            new_columns.append(col_name)
        elif indicator_type == "Difference (T-n)":
            col_name = f"DIFF_{source_column}_{lag_periods}"
            df[col_name] = df[source_column].shift(lag_periods) - df[source_column]
            new_columns.append(col_name)
        elif indicator_type == "Bollinger Bands":
            bbands = ta.bbands(close=df[source_column], length=window_int, std=float(std_dev or 2.0))
            if bbands is None:
                if show_feedback:
                    st.sidebar.error("Unable to compute Bollinger Bands with the current settings.")
                return df, []
            df = df.drop(columns=prior_columns, errors="ignore")
            for col in bbands.columns:
                df[col] = bbands[col]
            new_columns.extend(list(bbands.columns))
        else:
            if show_feedback:
                st.sidebar.warning(f"Indicator type '{indicator_type}' is not recognized.")
            return df, []
    except Exception as exc:  # pragma: no cover - user-facing feedback
        if show_feedback:
            st.sidebar.error(f"Failed to compute feature: {exc}")
        return df, []

    return df, new_columns


def reapply_feature_definitions(base_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild all saved features on top of a fresh dataframe."""
    working_df = base_df.copy()
    for definition in st.session_state.feature_definitions:
        working_df, new_cols = compute_indicator(working_df, definition, show_feedback=False)
        definition["columns"] = new_cols
    update_available_features()
    return working_df


def format_correlation(value: float) -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def compute_correlation(
    df: pd.DataFrame,
    target_column: str,
    formula: str,
    coeffs: Dict[str, float],
) -> Tuple[float, Optional[pd.Series]]:
    derived = evaluate_formula(df, formula, coeffs)
    merged = pd.concat([df[target_column], derived], axis=1).dropna()
    if merged.empty:
        return np.nan, derived
    correlation_value = merged.iloc[:, 0].corr(merged.iloc[:, 1])
    return correlation_value, derived


def optimize_constants_sequential(
    constants: List[str],
    base_coeffs: Dict[str, float],
    df: pd.DataFrame,
    target_column: str,
    formula: str,
    passes: int = 1,
) -> Tuple[Dict[str, float], float, List[Dict[str, float]]]:
    """Coordinate-descent style search for multiple constants."""
    best_coeffs = base_coeffs.copy()
    history: List[Dict[str, float]] = []

    try:
        best_corr, _ = compute_correlation(df, target_column, formula, best_coeffs)
    except Exception:
        best_corr = np.nan
    if pd.isna(best_corr):
        best_corr = -np.inf

    for _ in range(max(1, passes)):
        improved = False
        for constant in constants:
            start_value = best_coeffs.get(constant, 0.0)
            scale = max(abs(start_value), 1.0)
            span = 3.0 * scale
            candidate_values = np.linspace(start_value - span, start_value + span, 41)

            best_const_value = start_value
            best_const_corr = best_corr

            for candidate in candidate_values:
                trial_coeffs = best_coeffs.copy()
                trial_coeffs[constant] = float(candidate)
                try:
                    corr, _ = compute_correlation(df, target_column, formula, trial_coeffs)
                except Exception:
                    corr = np.nan
                if pd.notna(corr) and corr > best_const_corr:
                    best_const_corr = corr
                    best_const_value = float(candidate)

            if best_const_value != start_value:
                improved = True
            best_coeffs[constant] = best_const_value
            best_corr = best_const_corr
            if best_const_value != start_value:
                history.append(
                    {
                        "constant": constant,
                        "value": best_const_value,
                        "correlation": best_const_corr,
                    }
                )
        if not improved:
            break

    return best_coeffs, best_corr, history

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file).reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - user feedback
        st.error(f"Unable to read the uploaded file: {exc}")
        st.stop()
    st.session_state.uploaded_filename = uploaded_file.name
    features_path = Path.cwd() / f"{sanitize_basename(uploaded_file.name)}_features.csv"
    st.session_state.features_file = str(features_path)
    persisted = load_persisted_features(features_path, len(uploaded_df))
    if persisted is not None:
        st.session_state.persisted_features = persisted
        combined_df = pd.concat([uploaded_df, persisted], axis=1)
        st.session_state.added_features = list(persisted.columns)
    else:
        st.session_state.persisted_features = None
        combined_df = uploaded_df
        st.session_state.added_features = []
    st.session_state.raw_df = combined_df.copy()
    st.session_state.df = combined_df.copy()
    st.session_state.sample_size = min(100, len(uploaded_df))
    st.session_state.applied_sample_size = None
    st.session_state.feature_definitions = []
    st.session_state.optimization_log = []

raw_df = st.session_state.raw_df if st.session_state.raw_df is not None else st.session_state.df
state_df = st.session_state.df
if state_df is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

if raw_df is None:
    raw_df = state_df

max_rows = len(raw_df)
if (
    st.session_state.applied_sample_size is None
    and st.session_state.sample_size == 0
    and max_rows > 0
):
    st.session_state.sample_size = min(100, max_rows)
elif st.session_state.sample_size > max_rows:
    st.session_state.sample_size = max_rows

sample_step = max(1, max_rows // 50) if max_rows else 1
sample_size = st.sidebar.number_input(
    "Rows to use (0 = all rows)",
    min_value=0,
    max_value=max_rows,
    value=int(st.session_state.sample_size),
    step=sample_step,
    key="sample_size",
    help=(
        "Limit the working dataset to the first N rows to speed up calculations. "
        "Changing this resets generated indicators."
    ),
)

sampling_changed = False

if st.session_state.applied_sample_size != sample_size and raw_df is not None:
    if 0 < sample_size < len(raw_df):
        subset = raw_df.iloc[: int(sample_size)].copy()
        sampled_df = subset.reset_index(drop=True)
    else:
        sampled_df = raw_df.copy().reset_index(drop=True)

    st.session_state.applied_sample_size = int(sample_size)
    working_df = sampled_df
    sampling_changed = True
elif st.session_state.df is not None:
    working_df = st.session_state.df.copy()
elif raw_df is not None:
    working_df = raw_df.copy().reset_index(drop=True)
else:
    st.error("Unable to prepare data sample.")
    st.stop()

if sampling_changed:
    if st.session_state.feature_definitions:
        rebuilt_df = reapply_feature_definitions(working_df.copy())
        st.session_state.df = rebuilt_df
        st.sidebar.info("Sampling change reapplied existing indicators to the new sample.")
    else:
        st.session_state.df = working_df
        update_available_features()
else:
    if st.session_state.df is None:
        st.session_state.df = working_df
    update_available_features()

state_df = st.session_state.df

if state_df is None:
    st.error("No data available after sampling.")
    st.stop()

base_df = state_df.drop(columns=["Derived Formula"], errors="ignore")

numeric_columns = base_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_columns:
    st.error("No numeric columns found in the uploaded CSV.")
    st.stop()

st.sidebar.header("Formula Settings")
target_search_term = st.sidebar.text_input(
    "Search target column",
    value="",
    key="target_column_search",
    placeholder="Type to filter columns...",
)
filtered_target_columns = [
    col for col in numeric_columns if target_search_term.lower() in col.lower()
]
if not filtered_target_columns:
    filtered_target_columns = numeric_columns
target_column = st.sidebar.selectbox(
    "Target Column",
    filtered_target_columns,
    key="target_column_select",
)

a = st.sidebar.number_input("a", value=1.0, key="coef_a")
b = st.sidebar.number_input("b", value=0.0, key="coef_b")
c = st.sidebar.number_input("c", value=0.0, key="coef_c")
d = st.sidebar.number_input("d", value=0.0, key="coef_d")
k = st.sidebar.number_input("k", value=0.0, key="k_value")

coefficients: Dict[str, float] = {"a": a, "b": b, "c": c, "d": d, "k": k}

default_formula = "k"
if len(numeric_columns) == 1:
    default_formula = f"a * {numeric_columns[0]} + k"
elif len(numeric_columns) >= 2:
    default_formula = f"a * {numeric_columns[0]} + b * {numeric_columns[1]} + k"

if "formula_input" not in st.session_state:
    st.session_state.formula_input = default_formula

formula_string = st.sidebar.text_input(
    "Enter your formula",
    value=st.session_state.formula_input,
    key="formula_input",
    help=(
        "Use column names and coefficients a, b, c, d, k. "
        "Example: a * Close + b * Volume + k"
    ),
)

st.sidebar.subheader("Optimize Constants")
find_best_k_clicked = st.sidebar.button("Find Best Constant (k)")
constants_to_optimize = st.sidebar.multiselect(
    "Constants to tune",
    options=["a", "b", "c", "d", "k"],
    default=[],
)
optimize_constants_clicked = st.sidebar.button("Optimize Selected Constants")

st.sidebar.subheader("Add Technical Indicators")
indicator_type = st.sidebar.selectbox(
    "Indicator Type",
    (
        "Rolling Mean (SMA)",
        "EMA",
        "Rolling Median",
        "Rolling Std Dev",
    "Rolling Std Dev (Sample)",
        "Rolling Max",
        "Rolling Min",
        "Lag (T-n)",
        "Difference (T-n)",
        "Bollinger Bands",
    ),
)
source_search_term = st.sidebar.text_input(
    "Search source column",
    value="",
    key="source_column_search",
    placeholder="Type to filter columns...",
)
filtered_source_columns = [
    col for col in numeric_columns if source_search_term.lower() in col.lower()
]
if not filtered_source_columns:
    filtered_source_columns = numeric_columns
source_column = st.sidebar.selectbox(
    "Source Column",
    filtered_source_columns,
    key="source_column_select",
)
window = st.sidebar.number_input("Window Period", min_value=1, value=20, step=1)

std_dev: Optional[float] = None
if indicator_type == "Bollinger Bands":
    std_dev = st.sidebar.number_input(
        "Standard Deviations", min_value=1.0, value=2.0, step=0.5
    )

lag_periods = 1
if indicator_type in {"Lag (T-n)", "Difference (T-n)"}:
    lag_periods = int(
        st.sidebar.number_input(
            "Lag periods (n)", min_value=1, value=1, step=1, key="lag_periods"
        )
    )

queue_feature_clicked = st.sidebar.button("Add To Feature Queue")
generate_features_clicked = st.sidebar.button("Generate Queued Features")

feature_definition = {
    "type": indicator_type,
    "source": source_column,
    "window": int(window),
    "std_dev": float(std_dev) if std_dev is not None else None,
    "lag": lag_periods,
}

if queue_feature_clicked:
    st.session_state.pending_indicators.append(feature_definition.copy())
    st.sidebar.success("Feature queued. Add more or generate when ready.")

if st.session_state.pending_indicators:
    st.sidebar.markdown("**Queued features:**")
    for idx, definition in enumerate(st.session_state.pending_indicators, start=1):
        src = definition.get("source", "?")
        kind = definition.get("type", "?")
        window_str = f", window={definition.get('window')}" if definition.get("window") else ""
        extra = ""
        if definition.get("type") == "Bollinger Bands" and definition.get("std_dev"):
            extra = f", std={definition.get('std_dev')}"
        if definition.get("type") in {"Lag (T-n)", "Difference (T-n)"}:
            extra = f", lag={definition.get('lag')}"
        st.sidebar.caption(f"{idx}. {kind} on {src}{window_str}{extra}")

if generate_features_clicked:
    if not st.session_state.pending_indicators:
        st.sidebar.info("No features queued yet.")
    else:
        raw_df_state = st.session_state.raw_df
        if raw_df_state is None:
            st.sidebar.warning("Load data before generating features.")
        else:
            previous_features = set(st.session_state.added_features)
            base_for_feature = raw_df_state.drop(columns=["Derived Formula"], errors="ignore").copy()
            updated_df = base_for_feature
            successful_definitions: List[Dict[str, Any]] = []
            all_new_columns: List[str] = []

            for definition in st.session_state.pending_indicators:
                updated_df, new_columns = compute_indicator(
                    updated_df,
                    definition,
                    show_feedback=True,
                )
                if new_columns:
                    definition = definition.copy()
                    definition["columns"] = new_columns
                    successful_definitions.append(definition)
                    all_new_columns.extend(new_columns)
                else:
                    st.sidebar.warning(
                        "Skipping a queued feature because it did not generate any columns."
                    )

            if successful_definitions and all_new_columns:
                st.session_state.feature_definitions.extend(successful_definitions)
                st.session_state.raw_df = updated_df.copy()
                st.session_state.df = updated_df.copy()
                persist_feature_columns(updated_df, all_new_columns)
                st.session_state.applied_sample_size = None
                update_available_features()
                added_now = [
                    col for col in st.session_state.added_features if col not in previous_features
                ]
                if added_now:
                    st.sidebar.success(f"Generated feature(s): {', '.join(added_now)}")
                else:
                    st.sidebar.info("Generated features updated existing columns.")
            else:
                st.sidebar.warning("Queued features did not produce any new columns.")

        st.session_state.pending_indicators = []
        st.sidebar.markdown("---")

update_available_features()
st.sidebar.write(
    "Available features:",
    st.session_state.added_features if st.session_state.added_features else "None yet.",
)


def evaluate_formula(dataframe: pd.DataFrame, formula: str, locals_dict: Dict[str, float]) -> pd.Series:
    """Evaluate the user formula against the provided dataframe."""
    safe_locals = {name: float(value) for name, value in locals_dict.items()}
    namespace: Dict[str, object] = {col: dataframe[col] for col in dataframe.columns}
    namespace.update(safe_locals)
    namespace["np"] = np

    with np.errstate(divide="ignore", invalid="ignore"):
        evaluated = pd.eval(formula, local_dict=namespace, engine="python")

    def _replace_infinite(obj: Any) -> Any:
        if isinstance(obj, pd.Series):
            return obj.replace([np.inf, -np.inf], np.nan)
        if isinstance(obj, pd.DataFrame):
            return obj.replace([np.inf, -np.inf], np.nan)
        if isinstance(obj, np.ndarray):
            arr = np.array(obj, dtype=float, copy=True)
            arr[np.isinf(arr)] = np.nan
            return arr
        if np.isscalar(obj):
            return np.nan if np.isinf(obj) else obj
        return obj

    evaluated = _replace_infinite(evaluated)

    if isinstance(evaluated, pd.Series):
        return evaluated
    if isinstance(evaluated, pd.DataFrame):
        if evaluated.shape[1] != 1:
            raise ValueError("Formula produced multiple columns; please return a single series.")
        return evaluated.iloc[:, 0]
    if np.isscalar(evaluated):
        return pd.Series([evaluated] * len(dataframe), index=dataframe.index)
    return pd.Series(evaluated, index=dataframe.index)


if find_best_k_clicked:
    eval_df = st.session_state.df.drop(columns=["Derived Formula"], errors="ignore")
    if not formula_string.strip():
        st.sidebar.warning("Enter a formula before searching for k.")
    else:
        target_series = eval_df[target_column]
        target_mean = float(target_series.mean())
        target_std = float(target_series.std())
        if np.isnan(target_std) or target_std == 0.0:
            target_std = max(1.0, abs(target_mean))
        k_values = np.linspace(target_mean - 3 * target_std, target_mean + 3 * target_std, 181)

        best_k: Optional[float] = None
        best_corr = -np.inf
        formula_error: Optional[Exception] = None

        for candidate_k in k_values:
            trial_coeffs = {**coefficients, "k": float(candidate_k)}
            try:
                corr_value, _ = compute_correlation(eval_df, target_column, formula_string, trial_coeffs)
            except Exception as err:  # pragma: no cover - runtime formula errors
                formula_error = err
                break
            if pd.notna(corr_value) and corr_value > best_corr:
                best_corr = corr_value
                best_k = float(candidate_k)

        if formula_error is not None:
            st.sidebar.error(
                f"Formula evaluation failed during optimization: {formula_error}"
            )
        elif best_k is None:
            st.sidebar.warning("Unable to determine an improved constant for k.")
        else:
            coefficients["k"] = float(best_k)
            pending_updates = st.session_state.pending_coeff_updates
            pending_updates["k"] = float(best_k)
            st.session_state.pending_coeff_updates = pending_updates
            append_optimization_log(
                f"k optimized to {best_k:.4f} (corr {format_correlation(best_corr)})"
            )
            st.sidebar.success(
                f"Found best k: {best_k:.4f} (corr {format_correlation(best_corr)})"
            )

if optimize_constants_clicked:
    eval_df = st.session_state.df.drop(columns=["Derived Formula"], errors="ignore")
    if not formula_string.strip():
        st.sidebar.warning("Enter a formula before optimizing constants.")
    elif not constants_to_optimize:
        st.sidebar.warning("Select at least one constant to optimize.")
    else:
        constants_present: List[str] = []
        missing_constants: List[str] = []
        for constant in constants_to_optimize:
            pattern = rf"\b{re.escape(constant)}\b"
            if re.search(pattern, formula_string):
                constants_present.append(constant)
            else:
                missing_constants.append(constant)

        if missing_constants:
            st.sidebar.warning(
                "Ignoring constants not used in the formula: "
                + ", ".join(missing_constants)
            )

        if not constants_present:
            st.sidebar.warning(
                "None of the selected constants appear in the formula."
            )
        else:
            try:
                optimized_coeffs, optimized_corr, history = optimize_constants_sequential(
                    constants_present,
                    coefficients,
                    eval_df,
                    target_column,
                    formula_string,
                    passes=2,
                )
            except Exception as exc:  # pragma: no cover - user feedback
                st.sidebar.error(f"Optimization failed: {exc}")
            else:
                if not history:
                    st.sidebar.info(
                        f"Optimization finished. Best correlation remains {format_correlation(optimized_corr)}."
                    )
                else:
                    pending_updates = st.session_state.pending_coeff_updates
                    summary_parts = []
                    latest_record: Dict[str, Dict[str, float]] = {}
                    for record in history:
                        latest_record[record["constant"]] = record
                    for constant in constants_present:
                        if constant in optimized_coeffs:
                            value = float(optimized_coeffs[constant])
                            coefficients[constant] = value
                            pending_updates[constant] = value
                            summary_parts.append(f"{constant}={value:.4f}")
                            record = latest_record.get(constant)
                            if record:
                                append_optimization_log(
                                    f"{constant} optimized to {value:.4f} (corr {format_correlation(record['correlation'])})"
                                )
                    st.session_state.pending_coeff_updates = pending_updates
                    summary_text = ", ".join(summary_parts)
                    st.sidebar.success(
                        f"Updated constants: {summary_text} (corr {format_correlation(optimized_corr)})"
                    )

working_df = st.session_state.df.drop(columns=["Derived Formula"], errors="ignore")
result_df = working_df.copy()
derived_series: Optional[pd.Series] = None

if formula_string.strip():
    try:
        derived_series = evaluate_formula(working_df, formula_string, coefficients)
        result_df["Derived Formula"] = derived_series
        st.session_state.df = result_df
    except Exception as exc:  # pragma: no cover - user feedback
        st.session_state.df = working_df
        st.error(
            f"Error: {exc}. Check your formula. Did you spell a column name correctly, "
            "or forget to generate an indicator?"
        )
else:
    st.session_state.df = working_df
    st.info("Enter a formula to evaluate against the target column.")

current_df = st.session_state.df

st.sidebar.subheader("Plot & Correlate Columns")
plot_numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
plot_search_term = st.sidebar.text_input(
    "Search plot columns",
    value="",
    key="plot_column_search",
    placeholder="Type to filter columns...",
)
filtered_plot_columns = [
    col for col in plot_numeric_columns if plot_search_term.lower() in col.lower()
]
if not filtered_plot_columns:
    filtered_plot_columns = plot_numeric_columns
selected_plot_columns = st.sidebar.multiselect(
    "Columns to plot",
    filtered_plot_columns,
    default=[],
    key="plot_column_select",
)

if "Derived Formula" in current_df.columns:
    merged = pd.concat(
        [current_df[target_column], current_df["Derived Formula"]], axis=1
    ).dropna()
    if merged.empty:
        st.warning("Not enough valid rows to compute correlation.")
    else:
        correlation_value = merged.iloc[:, 0].corr(merged.iloc[:, 1])
        st.metric("Pearson Correlation", format_correlation(correlation_value))

        plot_source = (
            current_df[[target_column, "Derived Formula"]]
            .reset_index()
            .melt("index", var_name="Series", value_name="Value")
        )

        if alt is not None:
            chart = (
                alt.Chart(plot_source)
                .mark_line()
                .encode(
                    x=alt.X("index:Q", title="Row Index"),
                    y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                    color="Series:N",
                    tooltip=["index:Q", "Series:N", alt.Tooltip("Value:Q", format=".4f")],
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
            chart_json = chart.to_json(indent=2)
            plot_source_csv = plot_source.to_csv(index=False)
            st.download_button(
                "Download plot (Vega-Lite JSON)",
                data=chart_json,
                file_name="target_vs_formula_chart.json",
                mime="application/json",
                key="download_main_chart_json",
            )
            st.download_button(
                "Download plot data (CSV)",
                data=plot_source_csv,
                file_name="target_vs_formula_chart.csv",
                mime="text/csv",
                key="download_main_chart_csv",
            )
        else:  # pragma: no cover - fallback when Altair unavailable
            st.line_chart(current_df[[target_column, "Derived Formula"]])
            plot_source_csv = plot_source.to_csv(index=False)
            st.download_button(
                "Download plot data (CSV)",
                data=plot_source_csv,
                file_name="target_vs_formula_chart.csv",
                mime="text/csv",
                key="download_main_chart_csv_fallback",
            )

        png_bytes, jpg_bytes = build_plot_images(
            current_df,
            [target_column, "Derived Formula"],
            "Target vs Derived Formula",
        )
        if png_bytes:
            st.download_button(
                "Download plot (PNG)",
                data=png_bytes,
                file_name="target_vs_formula_chart.png",
                mime="image/png",
                key="download_main_chart_png",
            )
        if jpg_bytes:
            st.download_button(
                "Download plot (JPG)",
                data=jpg_bytes,
                file_name="target_vs_formula_chart.jpg",
                mime="image/jpeg",
                key="download_main_chart_jpg",
            )
else:
    st.warning("Formula results are unavailable. Check the formula and try again.")

if selected_plot_columns:
    st.markdown("**Custom Column Plot**")
    columns_to_plot = current_df[selected_plot_columns]
    chart_source = columns_to_plot.reset_index().melt(
        "index", var_name="Series", value_name="Value"
    )
    if chart_source.dropna().empty:
        st.info("Selected columns do not contain enough data to plot.")
    else:
        if alt is not None:
            custom_chart = (
                alt.Chart(chart_source)
                .mark_line()
                .encode(
                    x=alt.X("index:Q", title="Row Index"),
                    y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                    color="Series:N",
                    tooltip=[
                        "index:Q",
                        "Series:N",
                        alt.Tooltip("Value:Q", format=".4f"),
                    ],
                )
                .interactive()
            )
            st.altair_chart(custom_chart, use_container_width=True)
            custom_chart_json = custom_chart.to_json(indent=2)
            chart_source_csv = chart_source.to_csv(index=False)
            st.download_button(
                "Download custom plot (Vega-Lite JSON)",
                data=custom_chart_json,
                file_name="custom_columns_chart.json",
                mime="application/json",
                key="download_custom_chart_json",
            )
            st.download_button(
                "Download plot data (CSV)",
                data=chart_source_csv,
                file_name="custom_columns_chart.csv",
                mime="text/csv",
                key="download_custom_chart_csv",
            )
        else:  # pragma: no cover - fallback when Altair unavailable
            st.line_chart(columns_to_plot)
            chart_source_csv = chart_source.to_csv(index=False)
            st.download_button(
                "Download plot data (CSV)",
                data=chart_source_csv,
                file_name="custom_columns_chart.csv",
                mime="text/csv",
                key="download_custom_chart_csv_fallback",
            )

        png_bytes, jpg_bytes = build_plot_images(
            current_df,
            selected_plot_columns,
            "Custom Column Plot",
        )
        if png_bytes:
            st.download_button(
                "Download custom plot (PNG)",
                data=png_bytes,
                file_name="custom_columns_chart.png",
                mime="image/png",
                key="download_custom_chart_png",
            )
        if jpg_bytes:
            st.download_button(
                "Download custom plot (JPG)",
                data=jpg_bytes,
                file_name="custom_columns_chart.jpg",
                mime="image/jpeg",
                key="download_custom_chart_jpg",
            )

    corr_matrix = columns_to_plot.corr()
    if corr_matrix.empty:
        st.info("Unable to compute correlation for the selected columns.")
    else:
        st.markdown("**Correlation Matrix**")
        st.dataframe(corr_matrix.round(4))
        if len(selected_plot_columns) >= 2:
            st.markdown("**Pairwise Correlations**")
            for col_a, col_b in combinations(selected_plot_columns, 2):
                corr_value = corr_matrix.loc[col_a, col_b]
                st.write(f"{col_a} vs {col_b}: {format_correlation(corr_value)}")

if st.session_state.optimization_log:
    st.markdown("**Optimization History**")
    for entry in reversed(st.session_state.optimization_log):
        st.markdown(f"- {entry}")

st.dataframe(current_df)
