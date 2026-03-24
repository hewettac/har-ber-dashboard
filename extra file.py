import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Har-Ber Elite Dashboard", layout="wide")

# -------------------------
# LOAD MAPPING FILE
# -------------------------
@st.cache_data
def load_mapping():
    try:
        mapping = pd.read_csv("off_play_mapping_template.csv")
        mapping.columns = mapping.columns.astype(str).str.lower().str.strip()

        # safety checks
        expected_cols = ["raw_off_play", "concept_group", "off_play_clean"]
        for col in expected_cols:
            if col not in mapping.columns:
                mapping[col] = ""

        mapping["raw_off_play"] = mapping["raw_off_play"].astype(str).str.lower().str.strip()
        mapping["concept_group"] = mapping["concept_group"].astype(str).str.strip()
        mapping["off_play_clean"] = mapping["off_play_clean"].astype(str).str.strip()

        mapping = mapping.drop_duplicates(subset=["raw_off_play"], keep="first")
        return mapping

    except Exception:
        return pd.DataFrame(columns=["raw_off_play", "concept_group", "off_play_clean"])


# -------------------------
# STANDARDIZE COLUMNS
# -------------------------
def standardize_columns(df):
    df = df.copy()

    # normalize raw headers
    df.columns = df.columns.astype(str).str.lower().str.strip()

    # remove duplicate raw columns before mapping
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    alias_map = {
        "down": ["down", "dn"],
        "distance": ["distance", "dist", "togo", "yards to go", "ydstogo"],
        "hash": ["hash"],
        "yardline": ["yardline", "yard ln", "spot", "ball on"],
        "play_type": ["play_type", "play type", "playtype", "type"],
        "result": ["result"],
        "gain_loss": ["gain_loss", "gn/ls"],
        "formation": ["formation", "off form"],
        "off play": ["off play", "concept"],
        "off_str": ["off_str", "off str"],
        "play_direction": ["play_direction", "play dir"],
        "gap": ["gap"],
        "pass_zone": ["pass_zone", "pass zone"],
        "def_front": ["def_front", "def front"],
        "coverage": ["coverage"],
        "blitz": ["blitz"],
        "quarter": ["quarter", "qtr"]
    }

    rename_dict = {}
    for standard_name, aliases in alias_map.items():
        for alias in aliases:
            if alias in df.columns:
                rename_dict[alias] = standard_name
                break

    df = df.rename(columns=rename_dict)

    # if multiple raw aliases collapse into one standardized name, keep first
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df


# -------------------------
# APPLY MAPPING
# -------------------------
def apply_mapping(df, mapping):
    df = df.copy()

    # ensure unique columns
    df.columns = df.columns.astype(str).str.lower().str.strip()
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # make sure off play exists
    if "off play" not in df.columns:
        df["off play"] = "unknown"

    df["off play"] = df["off play"].astype(str).str.lower().str.strip()

    if mapping.empty or "raw_off_play" not in mapping.columns:
        df["concept_group"] = "Unknown"
        df["off_play_clean"] = df["off play"]
        return df

    mapping_dict = mapping.set_index("raw_off_play")[["concept_group", "off_play_clean"]].to_dict("index")

    def map_row(x):
        if x in mapping_dict:
            return mapping_dict[x]["concept_group"], mapping_dict[x]["off_play_clean"]
        return "Unknown", x

    mapped = df["off play"].apply(map_row)
    df["concept_group"] = mapped.apply(lambda x: x[0])
    df["off_play_clean"] = mapped.apply(lambda x: x[1])

    return df


# -------------------------
# FEATURE ENGINEERING
# -------------------------
def add_features(df):
    df = df.copy()

    if "distance" not in df.columns:
        df["distance"] = np.nan
    if "yardline" not in df.columns:
        df["yardline"] = np.nan
    if "down" not in df.columns:
        df["down"] = np.nan

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["yardline"] = pd.to_numeric(df["yardline"], errors="coerce")
    df["down"] = pd.to_numeric(df["down"], errors="coerce")

    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[-1, 3, 7, 100],
        labels=[0, 1, 2]
    ).astype(float)

    df["field_zone"] = pd.cut(
        df["yardline"],
        bins=[-51, -20, 20, 51],
        labels=[0, 1, 2]
    ).astype(float)

    df["short_yardage"] = (df["distance"] <= 3).astype(int)
    df["passing_down"] = ((df["down"] >= 3) & (df["distance"] >= 6)).astype(int)

    return df


# -------------------------
# LOAD BASE DATA
# -------------------------
@st.cache_data
def load_base():
    base = pd.read_csv("AllPlaysTrainData.csv")
    base = standardize_columns(base)
    return base


# -------------------------
# TRAIN MODEL
# -------------------------
@st.cache_resource(show_spinner=False)
def train_stage_model(df, target, features):
    df = df.copy()

    required_cols = features + [target]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return None, None, None

    df = df.dropna(subset=required_cols)

    if df.empty:
        return None, None, None

    counts = df[target].value_counts()
    valid = counts[counts >= 2].index
    df = df[df[target].isin(valid)]

    if df.empty or df[target].nunique() < 2:
        return None, None, None

    classes = sorted(df[target].astype(str).unique())
    class_to_int = {c: i for i, c in enumerate(classes)}
    int_to_class = {i: c for c, i in class_to_int.items()}

    df[target] = df[target].astype(str)
    df["y"] = df[target].map(class_to_int)

    X = df[features]
    y = df["y"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        return None, None, None

    model = XGBClassifier(
        n_estimators=100,        # lowered from 400
        max_depth=4,             # lowered from 6
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, int_to_class, acc

# -------------------------
# SIDEBAR UPLOAD
# -------------------------
uploaded = st.sidebar.file_uploader("Upload Weekly Hudl File", type=["xlsx"])

if not uploaded:
    st.stop()

df = pd.read_excel(uploaded)
df = standardize_columns(df)

# optional debug section
with st.sidebar.expander("Debug Uploaded Columns", expanded=False):
    st.write("Columns:", list(df.columns))
    st.write("Duplicate columns:", df.columns[df.columns.duplicated()].tolist())

mapping = load_mapping()
df = apply_mapping(df, mapping)
df = add_features(df)

base_df = load_base()
base_df = apply_mapping(base_df, mapping)
base_df = add_features(base_df)

# transfer learning weighting
base_df["weight"] = 1
df["weight"] = 10

combined = pd.concat([base_df, df], ignore_index=True)

# -------------------------
# FEATURES
# -------------------------
features = [
    "down", "distance", "yardline",
    "distance_bucket", "field_zone",
    "short_yardage", "passing_down"
]

# -------------------------
# TRAIN MODELS
# -------------------------
st.title("Har-Ber Elite Dashboard")
st.info("Weekly file uploaded. Training models now...")

with st.spinner("Training models... this may take a moment on first run."):
    st1_model, st1_map, st1_acc = train_stage_model(combined, "play_type", features)
    st2_model, st2_map, st2_acc = train_stage_model(combined, "concept_group", features)
    st3_model, st3_map, st3_acc = train_stage_model(combined, "off_play_clean", features)

st.write("✅ Weekly file uploaded")
st.write("Weekly rows:", len(df))
st.write("Weekly columns:", list(df.columns))

st.write("✅ Mapping loaded")
st.write("Mapping rows:", len(mapping))

st.write("✅ Base file loaded")
st.write("Base rows:", len(base_df))

st.write("✅ Combined rows:", len(combined))
# -------------------------
# HEADER
# -------------------------
st.title("Har-Ber Elite Dashboard")

acc_c1, acc_c2, acc_c3 = st.columns(3)
with acc_c1:
    st.metric("Stage 1 Accuracy", f"{st1_acc:.1%}" if st1_acc is not None else "N/A")
with acc_c2:
    st.metric("Stage 2 Accuracy", f"{st2_acc:.1%}" if st2_acc is not None else "N/A")
with acc_c3:
    st.metric("Stage 3 Accuracy", f"{st3_acc:.1%}" if st3_acc is not None else "N/A")

# -------------------------
# INPUT
# -------------------------
st.markdown("## 🎯 Predict Next Play")

c1, c2, c3 = st.columns(3)

with c1:
    down = st.selectbox("Down", [1, 2, 3, 4])
with c2:
    distance = st.number_input("Distance", min_value=1, max_value=30, value=5)
with c3:
    yardline = st.slider("Yardline", min_value=-50, max_value=50, value=0)

input_df = pd.DataFrame([{
    "down": down,
    "distance": distance,
    "yardline": yardline
}])

input_df = add_features(input_df)

# -------------------------
# PREDICTIONS
# -------------------------
if st1_model is not None:
    p1 = st1_model.predict_proba(input_df[features])[0]
    idx1 = np.argmax(p1)
    run_prob = p1[idx1]
    run_label = st1_map[idx1]

    st.markdown("### Stage 1")
    st.write(f"{run_label}: {run_prob:.1%}")

    if st2_model is not None:
        p2 = st2_model.predict_proba(input_df[features])[0]
        idx2 = np.argmax(p2)
        group = st2_map[idx2]
        prob2 = p2[idx2]

        st.markdown("### Stage 2")
        st.write(f"{group} given {run_label}: {prob2:.1%}")

        if st3_model is not None:
            p3 = st3_model.predict_proba(input_df[features])[0]
            idx3 = np.argmax(p3)
            concept = st3_map[idx3]
            prob3 = p3[idx3]

            overall = run_prob * prob2 * prob3

            st.markdown("### Stage 3")
            st.write(f"{concept} given {group}: {prob3:.1%}")

            st.markdown("### 🔥 Final Prediction")
            st.write(f"{concept} overall: {overall:.1%}")
else:
    st.warning("Not enough clean data to train Stage 1 model.")

# -------------------------
# LIVE LOGGING
# -------------------------
st.markdown("## 📝 Live Game Logger")

if "game_log" not in st.session_state:
    st.session_state.game_log = pd.DataFrame()

log_c1, log_c2 = st.columns(2)

play_type_options = (
    sorted(df["play_type"].dropna().astype(str).unique().tolist())
    if "play_type" in df.columns else []
)

concept_options = (
    sorted(df["off_play_clean"].dropna().astype(str).unique().tolist())
    if "off_play_clean" in df.columns else []
)

with log_c1:
    actual_type = st.selectbox(
        "Actual Play Type",
        play_type_options if play_type_options else ["Unknown"]
    )

with log_c2:
    actual_concept = st.selectbox(
        "Actual Concept",
        concept_options if concept_options else ["Unknown"]
    )

if st.button("Log Play"):
    new_row = {
        "down": down,
        "distance": distance,
        "yardline": yardline,
        "play_type": actual_type,
        "off_play_clean": actual_concept
    }

    st.session_state.game_log = pd.concat(
        [st.session_state.game_log, pd.DataFrame([new_row])],
        ignore_index=True
    )

# -------------------------
# LIVE TENDENCIES
# -------------------------
if not st.session_state.game_log.empty:
    st.markdown("## 📊 Live Tendencies")

    log = st.session_state.game_log

    run_rate = (log["play_type"].astype(str).str.lower() == "run").mean()
    pass_rate = (log["play_type"].astype(str).str.lower() == "pass").mean()

    t1, t2 = st.columns(2)
    with t1:
        st.metric("Run Rate", f"{run_rate:.1%}")
    with t2:
        st.metric("Pass Rate", f"{pass_rate:.1%}")

    st.dataframe(log, use_container_width=True)
