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
        mapping["raw_off_play"] = mapping["raw_off_play"].str.lower().str.strip()
        return mapping
    except:
        return pd.DataFrame(columns=["raw_off_play","concept_group","off_play_clean"])

# -------------------------
# APPLY MAPPING
# -------------------------
def apply_mapping(df, mapping):
    df = df.copy()
    df["off play"] = df["off play"].astype(str).str.lower().str.strip()

    mapping_dict = mapping.set_index("raw_off_play").to_dict("index")

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

    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["yardline"] = pd.to_numeric(df["yardline"], errors="coerce")
    df["down"] = pd.to_numeric(df["down"], errors="coerce")

    df["distance_bucket"] = pd.cut(df["distance"], [-1,3,7,100], labels=[0,1,2]).astype(float)
    df["field_zone"] = pd.cut(df["yardline"], [-51,-20,20,51], labels=[0,1,2]).astype(float)

    df["short_yardage"] = (df["distance"] <= 3).astype(int)
    df["passing_down"] = ((df["down"]>=3) & (df["distance"]>=6)).astype(int)

    return df

# -------------------------
# LOAD BASE DATA
# -------------------------
@st.cache_data
def load_base():
    base = pd.read_csv("AllPlaysTrainData.csv")
    base.columns = base.columns.str.lower().str.strip()
    return base

# -------------------------
# TRAIN MODEL
# -------------------------
def train_stage_model(df, target, features):

    df = df.dropna(subset=features+[target])

    # remove rare classes
    counts = df[target].value_counts()
    valid = counts[counts>=2].index
    df = df[df[target].isin(valid)]

    if df[target].nunique() < 2:
        return None, None

    classes = sorted(df[target].unique())
    mapping = {c:i for i,c in enumerate(classes)}
    inv_map = {i:c for c,i in mapping.items()}

    df["y"] = df[target].map(mapping)

    X = df[features]
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )

    model.fit(X_train,y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, inv_map, acc

# -------------------------
# SIDEBAR UPLOAD
# -------------------------
uploaded = st.sidebar.file_uploader("Upload Weekly Hudl File", type=["xlsx"])

if not uploaded:
    st.stop()

df = pd.read_excel(uploaded)
df.columns = df.columns.str.lower().str.strip()

# rename key columns
rename_map = {
    "dn":"down",
    "dist":"distance",
    "yard ln":"yardline",
    "play type":"play_type",
    "off play":"off play"
}
df = df.rename(columns=rename_map)

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
    "down","distance","yardline",
    "distance_bucket","field_zone",
    "short_yardage","passing_down"
]

# -------------------------
# TRAIN MODELS
# -------------------------
st1_model, st1_map, st1_acc = train_stage_model(combined,"play_type",features)
st2_model, st2_map, st2_acc = train_stage_model(combined,"concept_group",features)
st3_model, st3_map, st3_acc = train_stage_model(combined,"off_play_clean",features)

# -------------------------
# INPUT
# -------------------------
st.markdown("## 🎯 Predict Next Play")

c1,c2,c3 = st.columns(3)

with c1:
    down = st.selectbox("Down",[1,2,3,4])
with c2:
    distance = st.number_input("Distance",1,30,5)
with c3:
    yardline = st.slider("Yardline",-50,50,0)

input_df = pd.DataFrame([{
    "down":down,
    "distance":distance,
    "yardline":yardline
}])

input_df = add_features(input_df)

# -------------------------
# PREDICTIONS
# -------------------------
if st1_model:

    p1 = st1_model.predict_proba(input_df[features])[0]
    run_idx = np.argmax(p1)
    run_prob = p1[run_idx]
    run_label = st1_map[run_idx]

    st.markdown("### Stage 1")
    st.write(f"{run_label}: {run_prob:.1%}")

    if st2_model:
        p2 = st2_model.predict_proba(input_df[features])[0]
        idx2 = np.argmax(p2)
        group = st2_map[idx2]
        prob2 = p2[idx2]

        st.markdown("### Stage 2")
        st.write(f"{group} given {run_label}: {prob2:.1%}")

        if st3_model:
            p3 = st3_model.predict_proba(input_df[features])[0]
            idx3 = np.argmax(p3)
            concept = st3_map[idx3]
            prob3 = p3[idx3]

            overall = run_prob * prob2 * prob3

            st.markdown("### Stage 3")
            st.write(f"{concept} given {group}: {prob3:.1%}")

            st.markdown("### 🔥 Final Prediction")
            st.write(f"{concept} overall: {overall:.1%}")

# -------------------------
# LIVE LOGGING
# -------------------------
st.markdown("## 📝 Live Game Logger")

if "game_log" not in st.session_state:
    st.session_state.game_log = pd.DataFrame()

log_c1, log_c2 = st.columns(2)

with log_c1:
    actual_type = st.selectbox("Actual Play Type", df["play_type"].dropna().unique())

with log_c2:
    actual_concept = st.selectbox("Actual Concept", df["off_play_clean"].dropna().unique())

if st.button("Log Play"):
    new_row = {
        "down":down,
        "distance":distance,
        "yardline":yardline,
        "play_type":actual_type,
        "off_play_clean":actual_concept
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

    run_rate = (log["play_type"]=="Run").mean()
    pass_rate = (log["play_type"]=="Pass").mean()

    st.write(f"Run Rate: {run_rate:.1%}")
    st.write(f"Pass Rate: {pass_rate:.1%}")

    st.dataframe(log)
