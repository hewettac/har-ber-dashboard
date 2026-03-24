import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Hudl Predictor V2", layout="wide")

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0d0d0d;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #111111;
}
.metric-card {
    background: linear-gradient(135deg, #0A2342, #12355B);
    padding: 16px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}
.section-header {
    background: #0A2342;
    color: white;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 700;
    margin-top: 16px;
    margin-bottom: 12px;
}
.small-note {
    color: #bbbbbb;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
HUDL_ALIAS_MAP = {
    "quarter": ["quarter", "qtr", "q"],
    "clock": ["clock", "time", "game clock", "time left"],
    "score_diff": ["score_diff", "score differential", "score margin", "margin"],
    "down": ["down", "dn"],
    "distance": ["distance", "dist", "togo", "yards to go", "ydstogo"],
    "yardline": ["yardline", "yard ln", "spot", "ball on"],
    "hash": ["hash", "ball hash"],
    "play_type": ["play_type", "play type", "playtype", "type"],
    "gain_loss": ["gain_loss", "gn/ls", "gain/loss", "yards gained"],
    "formation": ["formation", "off form", "offense formation", "off form."],
    "off_play": ["off play", "concept", "play call", "offense play"],
    "off_str": ["off_str", "off str", "strength", "off strength"],
    "play_direction": ["play_direction", "play dir", "direction"],
    "personnel": ["personnel", "off personnel", "pers", "grouping"],
    "back_alignment": ["back_alignment", "back align", "rb align", "backfield"],
    "motion": ["motion", "jet motion", "shift/motion"],
    "result": ["result"],
    "def_front": ["def_front", "def front"],
    "coverage": ["coverage"],
    "blitz": ["blitz"],
    "opponent": ["opponent", "opp"],
}

CORE_FEATURE_COLUMNS = [
    "quarter",
    "minutes_left",
    "score_diff",
    "down",
    "distance",
    "yardline",
    "hash_num",
    "personnel_num",
    "formation_num",
    "strength_num",
    "back_align_num",
    "motion_num",
    "distance_bucket",
    "field_zone",
    "short_yardage",
    "passing_down",
]

# =========================================================
# HELPERS
# =========================================================
def safe_lower_strip_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower().str.strip()
    return df

def first_existing_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None

def standardize_columns(df):
    df = df.copy()
    df = safe_lower_strip_columns(df)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    rename_dict = {}
    for standard_name, aliases in HUDL_ALIAS_MAP.items():
        found = first_existing_column(df, aliases)
        if found is not None:
            rename_dict[found] = standard_name

    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

def parse_clock_to_minutes(clock_value):
    if pd.isna(clock_value):
        return np.nan
    s = str(clock_value).strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            try:
                mins = float(parts[0])
                secs = float(parts[1])
                return mins + secs / 60.0
            except:
                return np.nan
    try:
        return float(s)
    except:
        return np.nan

def normalize_hash(val):
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower()
    if s in ["l", "left"]:
        return "left"
    if s in ["m", "mid", "middle", "center"]:
        return "middle"
    if s in ["r", "right"]:
        return "right"
    return s

def normalize_play_type(val):
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower()
    if "run" in s:
        return "run"
    if "pass" in s:
        return "pass"
    return s

def normalize_personnel(val):
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower().replace(" personnel", "").replace("pers", "").strip()
    s = s.replace("-", "").replace(" ", "")
    return s if s else "unknown"

def normalize_text(val):
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower()
    return s if s else "unknown"

def load_default_mapping():
    return pd.DataFrame(columns=["raw_off_play", "concept_group", "off_play_clean"])

@st.cache_data
def load_mapping_file(mapping_file):
    if mapping_file is None:
        return load_default_mapping()

    if mapping_file.name.lower().endswith(".csv"):
        mapping = pd.read_csv(mapping_file)
    else:
        mapping = pd.read_excel(mapping_file)

    mapping.columns = mapping.columns.astype(str).str.lower().str.strip()

    for col in ["raw_off_play", "concept_group", "off_play_clean"]:
        if col not in mapping.columns:
            mapping[col] = ""

    mapping["raw_off_play"] = mapping["raw_off_play"].astype(str).str.lower().str.strip()
    mapping["concept_group"] = mapping["concept_group"].astype(str).str.lower().str.strip()
    mapping["off_play_clean"] = mapping["off_play_clean"].astype(str).str.lower().str.strip()

    mapping = mapping.drop_duplicates(subset=["raw_off_play"], keep="first")
    return mapping

def apply_mapping(df, mapping):
    df = df.copy()
    df.columns = df.columns.astype(str).str.lower().str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "off_play" not in df.columns:
        df["off_play"] = "unknown"

    df["off_play"] = df["off_play"].astype(str).str.lower().str.strip()

    if mapping.empty:
        df["concept_group"] = df["off_play"]
        df["off_play_clean"] = df["off_play"]
        return df

    mapping_dict = mapping.set_index("raw_off_play")[["concept_group", "off_play_clean"]].to_dict("index")

    def mapper(x):
        if x in mapping_dict:
            return mapping_dict[x]["concept_group"], mapping_dict[x]["off_play_clean"]
        return "unknown", x

    mapped = df["off_play"].apply(mapper)
    df["concept_group"] = mapped.apply(lambda x: x[0])
    df["off_play_clean"] = mapped.apply(lambda x: x[1])
    return df

def add_missing_core_columns(df):
    df = df.copy()
    defaults = {
        "quarter": np.nan,
        "clock": np.nan,
        "minutes_left": np.nan,
        "score_diff": 0,
        "down": np.nan,
        "distance": np.nan,
        "yardline": np.nan,
        "hash": "unknown",
        "play_type": "unknown",
        "gain_loss": np.nan,
        "formation": "unknown",
        "off_play": "unknown",
        "off_str": "unknown",
        "play_direction": "unknown",
        "personnel": "unknown",
        "back_alignment": "unknown",
        "motion": "unknown",
        "opponent": "unknown",
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

def engineer_features(df):
    df = df.copy()
    df = add_missing_core_columns(df)

    df["quarter"] = pd.to_numeric(df["quarter"], errors="coerce")
    df["minutes_left"] = df["clock"].apply(parse_clock_to_minutes)
    df["score_diff"] = pd.to_numeric(df["score_diff"], errors="coerce").fillna(0)
    df["down"] = pd.to_numeric(df["down"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["yardline"] = pd.to_numeric(df["yardline"], errors="coerce")
    df["gain_loss"] = pd.to_numeric(df["gain_loss"], errors="coerce")

    df["hash"] = df["hash"].apply(normalize_hash)
    df["play_type"] = df["play_type"].apply(normalize_play_type)
    df["personnel"] = df["personnel"].apply(normalize_personnel)
    df["formation"] = df["formation"].apply(normalize_text)
    df["off_str"] = df["off_str"].apply(normalize_text)
    df["back_alignment"] = df["back_alignment"].apply(normalize_text)
    df["motion"] = df["motion"].apply(normalize_text)
    df["play_direction"] = df["play_direction"].apply(normalize_text)
    df["concept_group"] = df["concept_group"].apply(normalize_text)
    df["off_play_clean"] = df["off_play_clean"].apply(normalize_text)
    df["opponent"] = df["opponent"].apply(normalize_text)

    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[-1, 3, 6, 10, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)

    df["field_zone"] = pd.cut(
        df["yardline"],
        bins=[-51, -20, 20, 51],
        labels=[0, 1, 2]
    ).astype(float)

    df["short_yardage"] = (df["distance"] <= 3).astype(int)
    df["passing_down"] = ((df["down"] >= 3) & (df["distance"] >= 6)).astype(int)

    hash_num_map = {"left": 0, "middle": 1, "right": 2, "unknown": 3}
    df["hash_num"] = df["hash"].map(hash_num_map).fillna(3).astype(int)

    return df

def encode_categories_from_reference(df, ref_df, column, new_col):
    combined_values = pd.concat(
        [ref_df[column].astype(str), df[column].astype(str)],
        ignore_index=True
    ).fillna("unknown")

    categories = sorted(combined_values.unique().tolist())
    mapping = {cat: i for i, cat in enumerate(categories)}

    df[new_col] = df[column].astype(str).map(mapping).fillna(-1).astype(int)
    ref_df[new_col] = ref_df[column].astype(str).map(mapping).fillna(-1).astype(int)

    return df, ref_df, mapping

@st.cache_data
def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def build_training_weights(df, weekly_opponents=None, default_weight=1.0, opponent_weight=4.0):
    weights = np.ones(len(df), dtype=float) * default_weight
    if weekly_opponents:
        weekly_opponents = [str(x).lower().strip() for x in weekly_opponents if str(x).strip()]
        if "opponent" in df.columns:
            mask = df["opponent"].astype(str).str.lower().isin(weekly_opponents)
            weights[mask] = opponent_weight
    return weights

@st.cache_resource(show_spinner=False)
def train_xgb_model(train_df, target, feature_cols, weekly_opponents_tuple, opponent_weight):
    df = train_df.copy()

    required_cols = feature_cols + [target]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, None, None, None

    df = df.dropna(subset=feature_cols + [target])

    if df.empty:
        return None, None, None, None

    df[target] = df[target].astype(str)
    counts = df[target].value_counts()
    valid_classes = counts[counts >= 2].index.tolist()
    df = df[df[target].isin(valid_classes)]

    if df.empty or df[target].nunique() < 2:
        return None, None, None, None

    classes = sorted(df[target].unique().tolist())
    class_to_int = {c: i for i, c in enumerate(classes)}
    int_to_class = {i: c for c, i in class_to_int.items()}
    df["target_y"] = df[target].map(class_to_int)

    X = df[feature_cols]
    y = df["target_y"]
    sample_weights = build_training_weights(
        df,
        weekly_opponents=list(weekly_opponents_tuple),
        default_weight=1.0,
        opponent_weight=opponent_weight
    )

    try:
        X_train, X_temp, y_train, y_temp, w_train, w_temp = train_test_split(
            X, y, sample_weights, test_size=0.30, random_state=42, stratify=y
        )
        X_valid, X_test, y_valid, y_test, w_valid, w_test = train_test_split(
            X_temp, y_temp, w_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    except Exception:
        return None, None, None, None

    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1
    )

    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    except Exception:
        model.fit(X_train, y_train, sample_weight=w_train)

    pred_labels = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)

    exact_acc = accuracy_score(y_test, pred_labels)

    try:
        top3_acc = top_k_accuracy_score(
            y_test,
            pred_proba,
            k=min(3, pred_proba.shape[1]),
            labels=np.arange(pred_proba.shape[1])
        )
    except Exception:
        top3_acc = None

    metrics = {
        "exact_accuracy": exact_acc,
        "top3_accuracy": top3_acc
    }

    return model, int_to_class, metrics, classes

def get_top_predictions(model, int_to_class, input_df, feature_cols, top_n=3):
    if model is None:
        return pd.DataFrame(columns=["label", "probability"])
    probs = model.predict_proba(input_df[feature_cols])[0]
    order = np.argsort(probs)[::-1][:top_n]
    rows = []
    for idx in order:
        rows.append({
            "label": int_to_class[idx],
            "probability": probs[idx]
        })
    return pd.DataFrame(rows)

def build_situation_filter(df, down, distance, yardline, hash_val, personnel, formation, opponent_mode=None):
    cond = pd.Series(True, index=df.index)

    if "down" in df.columns:
        cond &= (df["down"] == down)
    if "distance" in df.columns:
        cond &= df["distance"].between(max(distance - 2, 1), distance + 2)
    if "yardline" in df.columns:
        cond &= df["yardline"].between(yardline - 10, yardline + 10)
    if "hash" in df.columns:
        cond &= (df["hash"] == hash_val)
    if "personnel" in df.columns and personnel != "unknown":
        cond &= (df["personnel"] == personnel)
    if "formation" in df.columns and formation != "unknown":
        cond &= (df["formation"] == formation)
    if opponent_mode and "opponent" in df.columns and opponent_mode != "all opponents":
        cond &= (df["opponent"] == opponent_mode)

    return df[cond].copy()

def format_pct(x):
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1%}"

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Uploads")

base_file = st.sidebar.file_uploader(
    "Upload Historical/Base Training File",
    type=["csv", "xlsx"],
    key="base_file"
)

weekly_file = st.sidebar.file_uploader(
    "Upload Weekly Hudl Export",
    type=["csv", "xlsx"],
    key="weekly_file"
)

mapping_file = st.sidebar.file_uploader(
    "Optional Off Play Mapping File",
    type=["csv", "xlsx"],
    key="mapping_file"
)

show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
opponent_weight = st.sidebar.slider("Weekly Opponent Weight", 1, 10, 4)

# =========================================================
# HEADER
# =========================================================
st.title("Hudl-to-Predictor Dashboard V2")
st.markdown("Upload Hudl-style data, weight the weekly opponent more heavily, and use the app for prediction + live gameday logging.")

if weekly_file is None:
    st.info("Upload your Weekly Hudl Export in the sidebar to begin.")
    st.stop()

# =========================================================
# LOAD FILES
# =========================================================
with st.spinner("Loading files and preparing model-ready data..."):
    weekly_raw = read_uploaded_file(weekly_file)
    weekly_df = standardize_columns(weekly_raw)

    if base_file is not None:
        base_raw = read_uploaded_file(base_file)
        base_df = standardize_columns(base_raw)
    else:
        base_df = weekly_df.copy()

    mapping_df = load_mapping_file(mapping_file)

    weekly_df = apply_mapping(weekly_df, mapping_df)
    base_df = apply_mapping(base_df, mapping_df)

    weekly_df = engineer_features(weekly_df)
    base_df = engineer_features(base_df)

    weekly_df, base_df, personnel_map = encode_categories_from_reference(
        weekly_df, base_df, "personnel", "personnel_num"
    )
    weekly_df, base_df, formation_map = encode_categories_from_reference(
        weekly_df, base_df, "formation", "formation_num"
    )
    weekly_df, base_df, strength_map = encode_categories_from_reference(
        weekly_df, base_df, "off_str", "strength_num"
    )
    weekly_df, base_df, back_map = encode_categories_from_reference(
        weekly_df, base_df, "back_alignment", "back_align_num"
    )
    weekly_df, base_df, motion_map = encode_categories_from_reference(
        weekly_df, base_df, "motion", "motion_num"
    )

    combined_df = pd.concat([base_df, weekly_df], ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep="first")]

weekly_opponents = sorted(
    weekly_df["opponent"].dropna().astype(str).unique().tolist()
) if "opponent" in weekly_df.columns else []

if not weekly_opponents:
    weekly_opponents = ["unknown"]

# =========================================================
# DEBUG
# =========================================================
if show_debug:
    with st.expander("Debug Info", expanded=False):
        st.write("Weekly raw columns:", list(weekly_raw.columns))
        st.write("Weekly standardized columns:", list(weekly_df.columns))
        st.write("Base standardized columns:", list(base_df.columns))
        st.write("Weekly opponents:", weekly_opponents)
        st.write("Weekly rows:", len(weekly_df))
        st.write("Base rows:", len(base_df))
        st.write("Combined rows:", len(combined_df))

# =========================================================
# TRAIN MODELS
# =========================================================
with st.spinner("Training weighted models..."):
    play_type_model, play_type_map, play_type_metrics, _ = train_xgb_model(
        combined_df,
        "play_type",
        CORE_FEATURE_COLUMNS,
        tuple(weekly_opponents),
        opponent_weight
    )
    concept_group_model, concept_group_map, concept_group_metrics, _ = train_xgb_model(
        combined_df,
        "concept_group",
        CORE_FEATURE_COLUMNS,
        tuple(weekly_opponents),
        opponent_weight
    )
    off_play_model, off_play_map, off_play_metrics, _ = train_xgb_model(
        combined_df,
        "off_play_clean",
        CORE_FEATURE_COLUMNS,
        tuple(weekly_opponents),
        opponent_weight
    )

# =========================================================
# STATUS
# =========================================================
st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        "Run/Pass Exact",
        format_pct(play_type_metrics["exact_accuracy"]) if play_type_metrics else "N/A"
    )
    st.caption(f"Top-3: {format_pct(play_type_metrics['top3_accuracy']) if play_type_metrics else 'N/A'}")

with m2:
    st.metric(
        "Concept Group Exact",
        format_pct(concept_group_metrics["exact_accuracy"]) if concept_group_metrics else "N/A"
    )
    st.caption(f"Top-3: {format_pct(concept_group_metrics['top3_accuracy']) if concept_group_metrics else 'N/A'}")

with m3:
    st.metric(
        "Clean Concept Exact",
        format_pct(off_play_metrics["exact_accuracy"]) if off_play_metrics else "N/A"
    )
    st.caption(f"Top-3: {format_pct(off_play_metrics['top3_accuracy']) if off_play_metrics else 'N/A'}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Prediction Panel", "Situation Explorer", "Live Gameday Log"])

# =========================================================
# TAB 1: PREDICTION PANEL
# =========================================================
with tab1:
    st.markdown('<div class="section-header">Gameday Prediction Panel</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pred_quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=0, key="pred_quarter")
    with c2:
        pred_clock = st.text_input("Clock", value="10:00", key="pred_clock")
    with c3:
        pred_score_diff = st.number_input("Score Differential", value=0, step=1, key="pred_score_diff")
    with c4:
        pred_down = st.selectbox("Down", [1, 2, 3, 4], index=0, key="pred_down")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        pred_distance = st.number_input("Distance", min_value=1, max_value=50, value=6, key="pred_distance")
    with c6:
        pred_yardline = st.slider("Yardline", min_value=-50, max_value=50, value=-35, key="pred_yardline")
    with c7:
        pred_hash = st.selectbox("Hash", ["left", "middle", "right"], index=0, key="pred_hash")
    with c8:
        personnel_options = sorted(combined_df["personnel"].dropna().astype(str).unique().tolist()) or ["unknown"]
        default_personnel_idx = personnel_options.index("11") if "11" in personnel_options else 0
        pred_personnel = st.selectbox("Personnel", personnel_options, index=default_personnel_idx, key="pred_personnel")

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        formation_options = sorted(combined_df["formation"].dropna().astype(str).unique().tolist()) or ["unknown"]
        pred_formation = st.selectbox("Formation", formation_options, key="pred_formation")
    with c10:
        strength_options = sorted(combined_df["off_str"].dropna().astype(str).unique().tolist()) or ["unknown"]
        pred_strength = st.selectbox("Strength", strength_options, key="pred_strength")
    with c11:
        back_options = sorted(combined_df["back_alignment"].dropna().astype(str).unique().tolist()) or ["unknown"]
        pred_back = st.selectbox("Back Alignment", back_options, key="pred_back")
    with c12:
        motion_options = sorted(combined_df["motion"].dropna().astype(str).unique().tolist()) or ["unknown"]
        pred_motion = st.selectbox("Motion", motion_options, key="pred_motion")

    opponent_filter_options = ["all opponents"] + sorted(combined_df["opponent"].dropna().astype(str).unique().tolist())
    pred_opponent_mode = st.selectbox("Prediction Context", opponent_filter_options, index=0, key="pred_opp_mode")

    predict_now = st.button("Generate Prediction", use_container_width=True)

    input_row = pd.DataFrame([{
        "quarter": pred_quarter,
        "clock": pred_clock,
        "score_diff": pred_score_diff,
        "down": pred_down,
        "distance": pred_distance,
        "yardline": pred_yardline,
        "hash": pred_hash,
        "personnel": pred_personnel,
        "formation": pred_formation,
        "off_str": pred_strength,
        "back_alignment": pred_back,
        "motion": pred_motion,
        "play_type": "unknown",
        "off_play": "unknown",
        "concept_group": "unknown",
        "off_play_clean": "unknown",
        "opponent": pred_opponent_mode if pred_opponent_mode != "all opponents" else "unknown",
    }])

    input_row = engineer_features(input_row)
    input_row["personnel_num"] = input_row["personnel"].astype(str).map(personnel_map).fillna(-1).astype(int)
    input_row["formation_num"] = input_row["formation"].astype(str).map(formation_map).fillna(-1).astype(int)
    input_row["strength_num"] = input_row["off_str"].astype(str).map(strength_map).fillna(-1).astype(int)
    input_row["back_align_num"] = input_row["back_alignment"].astype(str).map(back_map).fillna(-1).astype(int)
    input_row["motion_num"] = input_row["motion"].astype(str).map(motion_map).fillna(-1).astype(int)

    if predict_now:
        top_play_type = get_top_predictions(play_type_model, play_type_map, input_row, CORE_FEATURE_COLUMNS, top_n=2)
        top_groups = get_top_predictions(concept_group_model, concept_group_map, input_row, CORE_FEATURE_COLUMNS, top_n=3)
        top_plays = get_top_predictions(off_play_model, off_play_map, input_row, CORE_FEATURE_COLUMNS, top_n=5)

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown("### Run / Pass")
            view = top_play_type.copy()
            if not view.empty:
                view["probability"] = (view["probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(view, use_container_width=True, hide_index=True)

        with p2:
            st.markdown("### Concept Group")
            view = top_groups.copy()
            if not view.empty:
                view["probability"] = (view["probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(view, use_container_width=True, hide_index=True)

        with p3:
            st.markdown("### Top Concepts")
            view = top_plays.copy()
            if not view.empty:
                view["probability"] = (view["probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(view, use_container_width=True, hide_index=True)

        if not top_play_type.empty:
            best_label = top_play_type.iloc[0]["label"]
            best_prob = top_play_type.iloc[0]["probability"]
            st.success(f"Most likely play type: {best_label} ({best_prob:.1%})")

# =========================================================
# TAB 2: SITUATION EXPLORER
# =========================================================
with tab2:
    st.markdown('<div class="section-header">Situation Explorer</div>', unsafe_allow_html=True)

    explorer_opponent = st.selectbox(
        "Filter to Opponent",
        ["all opponents"] + sorted(combined_df["opponent"].dropna().astype(str).unique().tolist()),
        index=0,
        key="explorer_opponent"
    )

    situation_df = build_situation_filter(
        combined_df,
        down=st.session_state.get("pred_down", 1),
        distance=st.session_state.get("pred_distance", 6),
        yardline=st.session_state.get("pred_yardline", -35),
        hash_val=st.session_state.get("pred_hash", "left"),
        personnel=st.session_state.get("pred_personnel", "unknown"),
        formation=st.session_state.get("pred_formation", "unknown"),
        opponent_mode=explorer_opponent
    )

    st.write(f"Matching historical plays: **{len(situation_df)}**")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown("### Run / Pass Tendency")
        if not situation_df.empty and "play_type" in situation_df.columns:
            tmp = situation_df["play_type"].value_counts(normalize=True).reset_index()
            tmp.columns = ["play_type", "rate"]
            tmp["rate"] = (tmp["rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(tmp, use_container_width=True, hide_index=True)
        else:
            st.info("No matching run/pass data.")

    with s2:
        st.markdown("### Top Concept Groups")
        if not situation_df.empty and "concept_group" in situation_df.columns:
            tmp = situation_df["concept_group"].value_counts(normalize=True).head(8).reset_index()
            tmp.columns = ["concept_group", "rate"]
            tmp["rate"] = (tmp["rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(tmp, use_container_width=True, hide_index=True)
        else:
            st.info("No matching concept-group data.")

    with s3:
        st.markdown("### Top Clean Concepts")
        if not situation_df.empty and "off_play_clean" in situation_df.columns:
            tmp = situation_df["off_play_clean"].value_counts(normalize=True).head(8).reset_index()
            tmp.columns = ["off_play_clean", "rate"]
            tmp["rate"] = (tmp["rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(tmp, use_container_width=True, hide_index=True)
        else:
            st.info("No matching concept data.")

# =========================================================
# TAB 3: LIVE GAMEDAY LOG
# =========================================================
with tab3:
    st.markdown('<div class="section-header">Live Gameday Log</div>', unsafe_allow_html=True)

    if "game_log" not in st.session_state:
        st.session_state.game_log = pd.DataFrame()

    log_c1, log_c2, log_c3 = st.columns(3)

    with log_c1:
        actual_play_type = st.selectbox(
            "Actual Play Type",
            sorted(combined_df["play_type"].dropna().astype(str).unique().tolist()) or ["unknown"],
            key="actual_play_type"
        )

    with log_c2:
        actual_concept_group = st.selectbox(
            "Actual Concept Group",
            sorted(combined_df["concept_group"].dropna().astype(str).unique().tolist()) or ["unknown"],
            key="actual_concept_group"
        )

    with log_c3:
        actual_concept = st.selectbox(
            "Actual Clean Concept",
            sorted(combined_df["off_play_clean"].dropna().astype(str).unique().tolist()) or ["unknown"],
            key="actual_concept"
        )

    note = st.text_input("Optional Note", key="log_note")

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Log This Play", use_container_width=True):
            new_row = pd.DataFrame([{
                "quarter": st.session_state.get("pred_quarter", np.nan),
                "clock": st.session_state.get("pred_clock", ""),
                "score_diff": st.session_state.get("pred_score_diff", 0),
                "down": st.session_state.get("pred_down", np.nan),
                "distance": st.session_state.get("pred_distance", np.nan),
                "yardline": st.session_state.get("pred_yardline", np.nan),
                "hash": st.session_state.get("pred_hash", "unknown"),
                "personnel": st.session_state.get("pred_personnel", "unknown"),
                "formation": st.session_state.get("pred_formation", "unknown"),
                "off_str": st.session_state.get("pred_strength", "unknown"),
                "back_alignment": st.session_state.get("pred_back", "unknown"),
                "motion": st.session_state.get("pred_motion", "unknown"),
                "actual_play_type": actual_play_type,
                "actual_concept_group": actual_concept_group,
                "actual_concept": actual_concept,
                "note": note
            }])

            st.session_state.game_log = pd.concat(
                [st.session_state.game_log, new_row],
                ignore_index=True
            )
            st.success("Play logged.")

    with col_b:
        if st.button("Clear Log", use_container_width=True):
            st.session_state.game_log = pd.DataFrame()
            st.warning("Log cleared.")

    if not st.session_state.game_log.empty:
        st.markdown("### Logged Plays")
        st.dataframe(st.session_state.game_log, use_container_width=True)

        csv = st.session_state.game_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Gameday Log CSV",
            data=csv,
            file_name="gameday_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No plays logged yet.")

# =========================================================
# CLEANED DATA PREVIEW
# =========================================================
st.markdown('<div class="section-header">Cleaned Weekly Data Preview</div>', unsafe_allow_html=True)
preview_cols = [
    c for c in [
        "opponent", "quarter", "clock", "score_diff", "down", "distance", "yardline",
        "hash", "personnel", "formation", "off_str", "back_alignment", "motion",
        "play_type", "concept_group", "off_play_clean", "gain_loss"
    ] if c in weekly_df.columns
]
st.dataframe(weekly_df[preview_cols].head(50), use_container_width=True)
