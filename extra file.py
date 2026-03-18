import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har-Ber Basic Dashboard", layout="wide")

# -------------------------
# Styles
# -------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #0A2342;
    padding: 10px 15px;
    border-radius: 8px;
    text-align: center;
    color: white;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.4);
    transition: transform 0.15s;
    margin-bottom: 8px;
}
.metric-card:hover {
    transform: scale(1.03);
}
.metric-number {
    font-size: 22px;
    font-weight: 700;
    color: #7FDBFF;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 12px;
    color: #AAAAAA;
    font-weight: 500;
}
.section-header {
    font-size: 26px;
    font-weight: 700;
    color: #7FDBFF;
    margin: 10px 0 15px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
st.sidebar.image("logo_har-ber-high-school.png", width=150)
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.title("Har-Ber Basic Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Hudl Excel File", type=["xlsx", "xls"])

# -------------------------
# Helpers
# -------------------------
COLUMN_MAP = {
    "down": ["down", "dn"],
    "distance": ["distance", "dist", "togo", "yards to go", "ydstogo"],
    "hash": ["hash"],
    "yardline": ["yardline", "yard ln", "spot", "ball on"],
    "play_type": ["play_type", "play type", "playtype", "type"],
    "result": ["result"],
    "gain_loss": ["gain_loss", "gn/ls"],
    "formation": ["formation", "off form"],
    "concept": ["concept", "off play"],
    "off_str": ["off str"],
    "play_direction": ["play_direction", "play dir"],
    "gap": ["gap"],
    "pass_zone": ["pass zone"],
    "def_front": ["def front"],
    "coverage": ["coverage"],
    "blitz": ["blitz"],
    "quarter": ["quarter", "qtr"]
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    rename_dict = {}
    for standard_name, variants in COLUMN_MAP.items():
        for col in df.columns:
            if col in variants:
                rename_dict[col] = standard_name

    df = df.rename(columns=rename_dict)
    return df


def ensure_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def custom_yard_group(yardline):
    if pd.isna(yardline):
        return "Unknown"
    if 0 >= yardline >= -9:
        return "0 - -9"
    elif -10 >= yardline >= -19:
        return "-10 - -19"
    elif -20 >= yardline >= -29:
        return "-20 - -29"
    elif -30 >= yardline >= -39:
        return "-30 - -39"
    elif -40 >= yardline >= -50:
        return "-40 - -50"
    elif 50 >= yardline >= 40:
        return "+50 - +40"
    elif 39 >= yardline >= 30:
        return "+39 - +30"
    elif 29 >= yardline >= 20:
        return "+29 - +20"
    elif 19 >= yardline >= 10:
        return "+19 - +10"
    elif 9 >= yardline >= 0:
        return "+9 - 0"
    return "Other"


def add_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_numeric_columns(df, ["down", "distance", "yardline"])

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

    return df


@st.cache_data
def load_base_data() -> pd.DataFrame:
    base = pd.read_csv("AllPlaysTrainData.csv")
    base = standardize_columns(base)
    base = ensure_numeric_columns(base, ["down", "distance", "yardline", "gain_loss", "quarter"])
    return base


@st.cache_resource
def train_prediction_model(base_df: pd.DataFrame, weekly_df: pd.DataFrame):
    base_df = base_df.copy()
    weekly_df = weekly_df.copy()

    required_cols = ["down", "distance", "yardline", "play_type"]
    for col in required_cols:
        if col not in base_df.columns:
            raise ValueError(f"Base data missing required column: {col}")
        if col not in weekly_df.columns:
            raise ValueError(f"Weekly uploaded data missing required column: {col}")

    base_df = add_prediction_features(base_df)
    weekly_df = add_prediction_features(weekly_df)

    base_df = base_df.dropna(subset=["down", "distance", "yardline", "play_type", "distance_bucket", "field_zone"])
    weekly_df = weekly_df.dropna(subset=["down", "distance", "yardline", "play_type", "distance_bucket", "field_zone"])

    if weekly_df.empty:
        raise ValueError("Weekly dataset has no usable rows after cleaning.")

    # Transfer-learning style weighting
    base_df["weight"] = 1
    weekly_df["weight"] = 6

    combined = pd.concat([base_df, weekly_df], ignore_index=True)

    le = LabelEncoder()
    combined["play_type_encoded"] = le.fit_transform(combined["play_type"].astype(str))

    features = ["down", "distance", "yardline", "distance_bucket", "field_zone"]

    X = combined[features].reset_index(drop=True)
    y = combined["play_type_encoded"].reset_index(drop=True)
    weights = combined["weight"].reset_index(drop=True)

    # Remove ultra-rare classes that break stratified split
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index

    mask = y.isin(valid_classes)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    weights = weights.loc[mask].reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError("Not enough play types with at least 2 samples to train the model.")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=w_train)

    preds = model.predict(X_test)
    overall_accuracy = accuracy_score(y_test, preds)

    return model, le, overall_accuracy, X_test, y_test, features


# -------------------------
# Main App
# -------------------------
if not uploaded_file:
    st.info("Upload a Hudl Excel file in the sidebar to open the dashboard.")
    st.stop()

df = pd.read_excel(uploaded_file)
df = standardize_columns(df)
df = ensure_numeric_columns(df, ["down", "distance", "yardline", "gain_loss", "quarter"])

required_main_cols = ["down", "yardline"]
missing_main = [col for col in required_main_cols if col not in df.columns]
if missing_main:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing_main)}")
    st.stop()

df["yard_group"] = df["yardline"].apply(custom_yard_group)

yard_order = [
    "0 - -9", "-10 - -19", "-20 - -29", "-30 - -39", "-40 - -50",
    "+50 - +40", "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0"
]

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters")

down_choices = sorted([x for x in df["down"].dropna().unique()])
if not down_choices:
    st.error("No valid down values found in uploaded file.")
    st.stop()

down_selected = st.sidebar.selectbox("Down", down_choices)

df_down = df[df["down"] == down_selected]
yard_choices = [yg for yg in yard_order if yg in df_down["yard_group"].unique()]

if not yard_choices:
    st.error("No valid yard groups found for the selected down.")
    st.stop()

yard_choice = st.sidebar.selectbox("Yard Group", yard_choices)

selected = df_down[df_down["yard_group"] == yard_choice]
if selected.empty:
    st.warning("No plays for this selection.")
    st.stop()

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overall Snapshot",
    "Filtered by Down/Yardline",
    "Success Heatmaps",
    "Concept Effectiveness",
    "Play Prediction"
])

# -------------------------
# Tab 1
# -------------------------
with tab1:
    st.markdown("<div class='section-header'>Overall Snapshot</div>", unsafe_allow_html=True)

    if "gain_loss" in df.columns and not df["gain_loss"].dropna().empty:
        avg_gain_all = round(df["gain_loss"].mean(), 1)
        max_gain_all = df["gain_loss"].max()
        min_gain_all = df["gain_loss"].min()

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-card"><div class="metric-number">{avg_gain_all}</div><div class="metric-label">Average Gain</div></div>',
            unsafe_allow_html=True
        )
        c2.markdown(
            f'<div class="metric-card"><div class="metric-number">{max_gain_all}</div><div class="metric-label">Max Gain</div></div>',
            unsafe_allow_html=True
        )
        c3.markdown(
            f'<div class="metric-card"><div class="metric-number">{min_gain_all}</div><div class="metric-label">Min Gain</div></div>',
            unsafe_allow_html=True
        )

        gain_summary_all = (
            df.groupby("gain_loss")
            .size()
            .reset_index(name="plays")
            .sort_values("gain_loss")
        )

        gain_fig_all = px.bar(
            gain_summary_all,
            x="gain_loss",
            y="plays",
            labels={"gain_loss": "Yards Gained", "plays": "Number of Plays"},
            title="Gain / Loss Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )

        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig_all, use_container_width=True)
    else:
        st.warning("No gain/loss column available for overall metrics.")

    if {"concept", "play_direction"}.issubset(df.columns):
        top_concepts_all = (
            df.groupby(["concept", "play_direction"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(8)
        )

        concept_fig_all = px.bar(
            top_concepts_all,
            x="count",
            y="concept",
            color="play_direction",
            orientation="h",
            title="Most Frequent Concepts by Play Direction",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF", "#0A2342", "#AAAAAA"]
        )

        if "r1c2" in locals():
            r1c2.plotly_chart(concept_fig_all, use_container_width=True)
        else:
            st.plotly_chart(concept_fig_all, use_container_width=True)

    if "play_type" in df.columns:
        play_type_summary_all = df["play_type"].value_counts().reset_index()
        play_type_summary_all.columns = ["play_type", "count"]

        run_pass_fig_all = px.pie(
            play_type_summary_all,
            names="play_type",
            values="count",
            title="Play Type %",
            color="play_type",
            template="plotly_dark"
        )

        st.markdown('<div class="section-header">Play Type & Concept Distribution</div>', unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig_all, use_container_width=True)

        if "concept" in df.columns:
            concept_summary_all = df["concept"].value_counts().head(6).reset_index()
            concept_summary_all.columns = ["concept", "count"]

            concept_pie_fig_all = px.pie(
                concept_summary_all,
                names="concept",
                values="count",
                title="Most Frequent Concepts",
                color_discrete_sequence=px.colors.sequential.Blues,
                template="plotly_dark"
            )
            r2c2.plotly_chart(concept_pie_fig_all, use_container_width=True)

# -------------------------
# Tab 2
# -------------------------
with tab2:
    st.markdown("<div class='section-header'>Filtered by Down / Yardline</div>", unsafe_allow_html=True)

    if "gain_loss" in selected.columns and not selected["gain_loss"].dropna().empty:
        avg_gain = round(selected["gain_loss"].mean(), 1)
        max_gain = selected["gain_loss"].max()
        min_gain = selected["gain_loss"].min()

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-card"><div class="metric-number">{avg_gain}</div><div class="metric-label">Average Gain</div></div>',
            unsafe_allow_html=True
        )
        c2.markdown(
            f'<div class="metric-card"><div class="metric-number">{max_gain}</div><div class="metric-label">Max Gain</div></div>',
            unsafe_allow_html=True
        )
        c3.markdown(
            f'<div class="metric-card"><div class="metric-number">{min_gain}</div><div class="metric-label">Min Gain</div></div>',
            unsafe_allow_html=True
        )

        gain_summary = (
            selected.groupby("gain_loss")
            .size()
            .reset_index(name="plays")
            .sort_values("gain_loss")
        )

        gain_fig = px.bar(
            gain_summary,
            x="gain_loss",
            y="plays",
            labels={"gain_loss": "Yards Gained", "plays": "Number of Plays"},
            title="Gain / Loss Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )
    else:
        gain_fig = None
        st.warning("No gain/loss data available for filtered metrics.")

    if {"concept", "play_direction"}.issubset(selected.columns):
        top_concepts = (
            selected.groupby(["concept", "play_direction"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(8)
        )

        concept_fig = px.bar(
            top_concepts,
            x="count",
            y="concept",
            color="play_direction",
            orientation="h",
            title="Most Frequent Concepts by Play Direction",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF", "#0A2342", "#AAAAAA"]
        )
    else:
        concept_fig = None

    r1c1, r1c2 = st.columns(2)
    if gain_fig is not None:
        r1c1.plotly_chart(gain_fig, use_container_width=True)
    if concept_fig is not None:
        r1c2.plotly_chart(concept_fig, use_container_width=True)

    st.markdown('<div class="section-header">Play Type & Concept Distribution</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)

    if "play_type" in selected.columns:
        play_type_summary = selected["play_type"].value_counts().reset_index()
        play_type_summary.columns = ["play_type", "count"]

        run_pass_fig = px.pie(
            play_type_summary,
            names="play_type",
            values="count",
            title="Play Type %",
            color="play_type",
            template="plotly_dark"
        )
        r2c1.plotly_chart(run_pass_fig, use_container_width=True)

    if "concept" in selected.columns:
        concept_summary = selected["concept"].value_counts().head(6).reset_index()
        concept_summary.columns = ["concept", "count"]

        concept_pie_fig = px.pie(
            concept_summary,
            names="concept",
            values="count",
            title="Most Frequent Concepts",
            color_discrete_sequence=px.colors.sequential.Blues,
            template="plotly_dark"
        )
        r2c2.plotly_chart(concept_pie_fig, use_container_width=True)

    st.markdown('<div class="section-header">Raw Play Data</div>', unsafe_allow_html=True)
    st.dataframe(selected, use_container_width=True)

# -------------------------
# Tab 3
# -------------------------
with tab3:
    st.markdown("<div class='section-header'>Play Success Heatmap</div>", unsafe_allow_html=True)

    with st.expander("How to read this chart"):
        st.write("""
        This heatmap shows the success rate of plays by down and field position.

        Success is defined as gaining 4 or more yards.
        Darker cells indicate a higher success rate.
        """)

    if "gain_loss" not in df.columns:
        st.warning("The uploaded file does not contain GN/LS / gain_loss.")
    else:
        df["success"] = df["gain_loss"] >= 4

        heatmap_df = (
            df.groupby(["down", "yard_group"])
            .agg(
                success_rate=("success", "mean"),
                num_plays=("success", "count")
            )
            .reset_index()
        )

        if not heatmap_df.empty:
            success_pivot = heatmap_df.pivot(index="down", columns="yard_group", values="success_rate")
            plays_pivot = heatmap_df.pivot(index="down", columns="yard_group", values="num_plays")

            success_pivot = success_pivot.reindex(columns=[yg for yg in yard_order if yg in success_pivot.columns])
            plays_pivot = plays_pivot.reindex(columns=[yg for yg in yard_order if yg in plays_pivot.columns])

            heatmap_fig = px.imshow(
                success_pivot,
                text_auto=".0%",
                aspect="auto",
                labels=dict(x="Yard Group", y="Down", color="Success Rate"),
                color_continuous_scale="Blues"
            )

            heatmap_fig.update_traces(
                hovertemplate="<b>Down:</b> %{y}<br>"
                              "<b>Yard Group:</b> %{x}<br>"
                              "<b>Success Rate:</b> %{z:.0%}<br>"
                              "<b>Number of Plays:</b> %{customdata}",
                customdata=plays_pivot.values
            )

            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Not enough data to build the heatmap.")

# -------------------------
# Tab 4
# -------------------------
with tab4:
    st.markdown("<div class='section-header'>Concept Effectiveness</div>", unsafe_allow_html=True)

    with st.expander("How to read this chart"):
        st.write("""
        This chart evaluates concepts using:

        X-axis: Success Rate
        Y-axis: Average Yards Gained
        Bubble Size: Number of Plays
        """)

    if not {"concept", "gain_loss"}.issubset(df.columns):
        st.warning("The uploaded file needs OFF PLAY / concept and GN/LS / gain_loss for this chart.")
    else:
        concept_stats = (
            df.groupby("concept")
            .agg(
                avg_gain=("gain_loss", "mean"),
                success_rate=("gain_loss", lambda x: (x >= 4).mean()),
                plays=("gain_loss", "count")
            )
            .reset_index()
            .sort_values("avg_gain", ascending=False)
        )

        bubble = px.scatter(
            concept_stats,
            x="success_rate",
            y="avg_gain",
            size="plays",
            hover_name="concept",
            title="Concept Effectiveness",
            template="plotly_dark"
        )

        st.plotly_chart(bubble, use_container_width=True)
        st.dataframe(concept_stats, use_container_width=True)

# ------------------------
# Tab 5 - Play Prediction
# ------------------------
with tab5:

    st.markdown("<div class='section-header'>ELITE Play Prediction System</div>", unsafe_allow_html=True)

    # -------------------------
    # Feature Engineering
    # -------------------------
    def add_features(df):
        df = df.copy()

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

        return df

    # -------------------------
    # Load Base Dataset
    # -------------------------
    @st.cache_data
    def load_base_data():
        base = pd.read_csv("AllPlaysTrainData.csv")
        base.columns = base.columns.str.lower().str.strip()
        return base

    # -------------------------
    # Train Model
    # -------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder

    @st.cache_resource
    def train_model(base_df, weekly_df):

        base_df = base_df.copy()
        weekly_df = weekly_df.copy()

        base_df.columns = base_df.columns.str.lower().str.strip()
        weekly_df.columns = weekly_df.columns.str.lower().str.strip()

        required_cols = ["down", "distance", "yardline", "play_type"]
        for col in required_cols:
            if col not in base_df.columns:
                raise ValueError(f"Base data missing column: {col}")
            if col not in weekly_df.columns:
                raise ValueError(f"Weekly data missing column: {col}")

        base_df = add_features(base_df)
        weekly_df = add_features(weekly_df)

        base_df = base_df.dropna(subset=["down","distance","yardline","play_type","distance_bucket","field_zone"])
        weekly_df = weekly_df.dropna(subset=["down","distance","yardline","play_type","distance_bucket","field_zone"])

        if weekly_df.empty:
            raise ValueError("Weekly dataset has no usable rows.")

        # 🔥 Transfer Learning Weights
        base_df["weight"] = 1
        weekly_df["weight"] = 6

        combined = pd.concat([base_df, weekly_df], ignore_index=True)

        features = ["down","distance","yardline","distance_bucket","field_zone"]

        # 🔥 REMOVE RARE CLASSES BEFORE ENCODING (fixes your error)
        combined["play_type"] = combined["play_type"].astype(str)
        play_counts = combined["play_type"].value_counts()
        valid_play_types = play_counts[play_counts >= 2].index

        combined = combined[combined["play_type"].isin(valid_play_types)].reset_index(drop=True)

        if combined["play_type"].nunique() < 2:
            raise ValueError("Not enough play types to train model.")

        # 🔥 NOW encode (ensures labels = 0..n-1)
        le = LabelEncoder()
        combined["play_type_encoded"] = le.fit_transform(combined["play_type"])

        X = combined[features].reset_index(drop=True)
        y = combined["play_type_encoded"].reset_index(drop=True)
        weights = combined["weight"].reset_index(drop=True)

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )

        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )

        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        return model, le, accuracy, X_test, y_test, features

    # -------------------------
    # Load Data
    # -------------------------
    try:
        base_df = load_base_data()
    except Exception as e:
        st.error(f"Error loading base dataset: {e}")
        st.stop()

    weekly_df = df.copy()

    try:
        model, le, accuracy, X_test, y_test, features = train_model(base_df, weekly_df)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    st.success(f"Model Accuracy: {accuracy:.2%}")

    # -------------------------
    # User Inputs
    # -------------------------
    st.markdown("### Predict Play")

    col1, col2, col3 = st.columns(3)

    with col1:
        pred_down = st.selectbox("Down", [1,2,3,4])

    with col2:
        pred_dist = st.number_input("Distance", 1, 30, 5)

    with col3:
        pred_yard = st.number_input("Yardline", -50, 50, 0)

    input_df = pd.DataFrame([{
        "down": pred_down,
        "distance": pred_dist,
        "yardline": pred_yard
    }])

    input_df = add_features(input_df)

    probs = model.predict_proba(input_df[features])[0]

    pred_class = np.argmax(probs)
    predicted_play = le.inverse_transform([pred_class])[0]
    confidence = probs[pred_class]

    # -------------------------
    # Situation Accuracy
    # -------------------------
    mask = (
        (X_test["down"] == pred_down) &
        (abs(X_test["distance"] - pred_dist) <= 2) &
        (abs(X_test["yardline"] - pred_yard) <= 10)
    )

    similar_X = X_test[mask]
    similar_y = y_test[mask]

    if len(similar_X) >= 10:
        situation_acc = accuracy_score(similar_y, model.predict(similar_X))
    else:
        situation_acc = None

    # -------------------------
    # Display Results
    # -------------------------
    st.markdown("### Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Play", predicted_play)
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Model Accuracy", f"{accuracy:.1%}")

    if situation_acc:
        st.metric("Situation Accuracy", f"{situation_acc:.1%}")
    else:
        st.info("Not enough similar plays for situation accuracy")

    # -------------------------
    # Top 3 Plays
    # -------------------------
    st.markdown("### Top 3 Plays")

    top_idx = np.argsort(probs)[::-1][:3]

    for i in top_idx:
        st.write(f"{le.inverse_transform([i])[0]} — {probs[i]:.1%}")

    # -------------------------
    # Probability Table
    # -------------------------
    st.markdown("### Full Probabilities")

    prob_df = pd.DataFrame({
        "Play": le.inverse_transform(np.arange(len(probs))),
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    prob_df["Probability"] = prob_df["Probability"].map(lambda x: f"{x:.2%}")

    st.dataframe(prob_df, use_container_width=True)

    # -------------------------
    # Insight
    # -------------------------
    st.markdown("### Situation Insight")

    if pred_down == 3 and pred_dist >= 7:
        st.info("Likely PASS situation")
    elif pred_down == 1 and pred_dist <= 3:
        st.info("Likely RUN situation")
