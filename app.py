import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from io import BytesIO


st.set_page_config(
    page_title="Har-Ber Football Analytics",
    page_icon="🏈",
    layout="wide",
)


# =========================================================
# GLOBAL CHART THEME
# =========================================================
pio.templates.default = "plotly_dark"

pio.templates["plotly_dark"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font={"color": "#f5fbff"},
    title={"font": {"color": "#7FDBFF"}},
    legend={
        "bgcolor": "rgba(10,35,66,0.72)",
        "bordercolor": "rgba(127,219,255,0.28)",
        "borderwidth": 1,
    },
    xaxis={
        "gridcolor": "rgba(127,219,255,0.12)",
        "linecolor": "rgba(127,219,255,0.28)",
        "zerolinecolor": "rgba(127,219,255,0.22)",
    },
    yaxis={
        "gridcolor": "rgba(127,219,255,0.12)",
        "linecolor": "rgba(127,219,255,0.28)",
        "zerolinecolor": "rgba(127,219,255,0.22)",
    },
)


def themed_chart(figure):
    """Apply the Har-Ber theme immediately before rendering."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f5fbff",
        title_font_color="#7FDBFF",
        legend_bgcolor="rgba(10,35,66,0.72)",
        legend_bordercolor="rgba(127,219,255,0.28)",
        legend_borderwidth=1,
    )
    figure.update_xaxes(
        gridcolor="rgba(127,219,255,0.12)",
        linecolor="rgba(127,219,255,0.28)",
        zerolinecolor="rgba(127,219,255,0.22)",
    )
    figure.update_yaxes(
        gridcolor="rgba(127,219,255,0.12)",
        linecolor="rgba(127,219,255,0.28)",
        zerolinecolor="rgba(127,219,255,0.22)",
    )
    return figure


def show_chart(figure, key=None):
    st.plotly_chart(
        themed_chart(figure),
        use_container_width=True,
        theme=None,
        key=key,
    )


# =========================================================
# SHARED DATA HELPERS
# =========================================================
COLUMN_MAP = {
    "down": ["down", "dn"],
    "distance": [
        "distance", "dist", "togo", "yards to go", "ydstogo"
    ],
    "hash": ["hash"],
    "yardline": ["yardline", "yard ln", "spot", "ball on"],
    "play_type": ["play_type", "play type", "playtype", "type"],
    "result": ["result"],
    "gain_loss": [
        "gain_loss", "gn/ls", "gain loss", "gain/loss"
    ],
    "formation": ["formation", "off form", "off formation"],
    "concept": [
        "concept", "off play", "play concept", "play", "play call"
    ],
    "off_str": ["off str"],
    "play_direction": ["play_direction", "play direction", "play dir"],
    "gap": ["gap"],
    "pass_zone": ["pass_zone", "pass zone"],
    "def_front": ["def_front", "def front"],
    "coverage": ["coverage"],
    "blitz": ["blitz"],
    "quarter": ["quarter", "qtr"],
}


def standardize_columns(df):
    """Rename recognized headers while avoiding duplicate standard names."""
    df = df.copy()
    df.columns = [
        str(column).lower().strip() for column in df.columns
    ]

    rename_dict = {}
    for standard_name, variants in COLUMN_MAP.items():
        if standard_name in df.columns:
            continue

        match = next(
            (column for column in df.columns if column in variants),
            None,
        )
        if match is not None:
            rename_dict[match] = standard_name

    return df.rename(columns=rename_dict)


def ensure_numeric_columns(df, columns):
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )
    return df


def read_uploaded_excel(uploaded_file):
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)


def apply_global_play_settings(df):
    """Apply sidebar grouping rules without changing source data."""
    df = df.copy()

    if "concept" not in df.columns:
        return df

    cleaned_concepts = df["concept"].astype("string")

    if st.session_state.get("ignore_rpo_tag", True):
        cleaned_concepts = cleaned_concepts.str.replace(
            r"\s*\(\s*rpo\s*\)\s*$",
            "",
            regex=True,
            case=False,
        )

    if st.session_state.get("group_lt_plays_v2", True):
        cleaned_concepts = cleaned_concepts.str.replace(
            r"^\s*lt\b[\s_-]*",
            "",
            regex=True,
            case=False,
        )

    if st.session_state.get("group_read_plays_v2", True):
        cleaned_concepts = cleaned_concepts.str.replace(
            r"\bread\b",
            "",
            regex=True,
            case=False,
        )

    if st.session_state.get("group_numbered_plays_v2", True):
        cleaned_concepts = cleaned_concepts.str.replace(
            r"\d+",
            "",
            regex=True,
        )

    df["concept"] = (
        cleaned_concepts
        .str.replace(r"\s+", " ", regex=True)
        .str.strip(" -_/")
    )
    return df


def apply_custom_breakdown_groups(df):
    """Apply enabled custom groups to formation and concept columns."""
    df = df.copy()
    active_names = []

    for group in st.session_state.get(
        "custom_breakdown_groups", []
    ):
        enabled = st.session_state.get(
            f"custom_group_enabled_{group['id']}",
            group.get("enabled", True),
        )

        if enabled and group["column"] in df.columns:
            df.loc[
                df[group["column"]].isin(group["items"]),
                group["column"],
            ] = group["name"]
            active_names.append(group["name"])

    return df, active_names


def sidebar_logo():
    try:
        st.sidebar.image(
            "logo_har-ber-high-school.png", width=150
        )
    except Exception:
        pass


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


def add_prediction_features(df):
    df = ensure_numeric_columns(
        df, ["distance", "yardline", "down"]
    )

    df["distance_bucket"] = pd.cut(
        df["distance"],
        bins=[-1, 3, 7, 100],
        labels=[0, 1, 2],
    ).astype(float)

    df["field_zone"] = pd.cut(
        df["yardline"],
        bins=[-51, -20, 20, 51],
        labels=[0, 1, 2],
    ).astype(float)

    return df


@st.cache_data
def load_base_data():
    base = pd.read_csv("AllPlaysTrainData.csv")
    base = standardize_columns(base)
    return ensure_numeric_columns(
        base,
        ["down", "distance", "yardline", "gain_loss", "quarter"],
    )


@st.cache_resource(show_spinner=False)
def train_model(base_df, weekly_df):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    base_df = standardize_columns(base_df)
    weekly_df = standardize_columns(weekly_df)

    required = ["down", "distance", "yardline", "play_type"]

    for column in required:
        if column not in base_df.columns:
            raise ValueError(
                f"Base data missing column: {column}"
            )
        if column not in weekly_df.columns:
            raise ValueError(
                f"Weekly data missing column: {column}"
            )

    base_df = add_prediction_features(base_df)
    weekly_df = add_prediction_features(weekly_df)

    usable_columns = [
        "down", "distance", "yardline", "play_type",
        "distance_bucket", "field_zone",
    ]

    base_df = base_df.dropna(subset=usable_columns).copy()
    weekly_df = weekly_df.dropna(subset=usable_columns).copy()

    if weekly_df.empty:
        raise ValueError(
            "Weekly dataset has no usable rows."
        )

    base_df["weight"] = 1
    weekly_df["weight"] = 6

    combined = pd.concat(
        [base_df, weekly_df], ignore_index=True
    )

    features = [
        "down", "distance", "yardline",
        "distance_bucket", "field_zone",
    ]

    combined["play_type"] = (
        combined["play_type"].astype(str).str.strip()
    )
    combined = combined[
        combined["play_type"] != ""
    ].copy()

    counts = combined["play_type"].value_counts()
    valid_types = counts[counts >= 2].index
    combined = combined[
        combined["play_type"].isin(valid_types)
    ].copy()

    n_classes = combined["play_type"].nunique()

    if n_classes < 2:
        raise ValueError(
            "Not enough play types with at least 2 samples "
            "to train the model."
        )

    play_types = sorted(
        combined["play_type"].unique()
    )
    play_to_int = {
        play: index for index, play in enumerate(play_types)
    }
    int_to_play = {
        index: play for play, index in play_to_int.items()
    }

    combined["y"] = (
        combined["play_type"].map(play_to_int).astype(int)
    )

    X = combined[features].reset_index(drop=True)
    y = combined["y"].reset_index(drop=True)
    weights = combined["weight"].reset_index(drop=True)

    if len(X) == 0 or y.nunique() < 2:
        raise ValueError(
            "No usable multi-class training data remain."
        )

    (
        X_train, X_test,
        y_train, y_test,
        w_train, w_test,
    ) = train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training split contains fewer than 2 classes."
        )

    model_settings = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
    }

    if n_classes == 2:
        model = XGBClassifier(
            **model_settings,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    else:
        model = XGBClassifier(
            **model_settings,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
        )

    model.fit(
        X_train, y_train, sample_weight=w_train
    )

    accuracy = accuracy_score(
        y_test, model.predict(X_test)
    )

    return (
        model, int_to_play, accuracy,
        X_test, y_test, features, n_classes,
    )


# =========================================================
# GLOBAL STYLING
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --hb-bg: #07182d;
        --hb-panel: rgba(10, 35, 66, 0.92);
        --hb-panel-soft: rgba(12, 43, 79, 0.72);
        --hb-accent: #7FDBFF;
        --hb-accent-hover: #a8e7ff;
        --hb-text: #f5fbff;
        --hb-muted: #b8cfdf;
        --hb-border: rgba(127, 219, 255, 0.28);
    }

    html, body, [class*="css"] {
        color: var(--hb-text);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at top right,
                rgba(127,219,255,0.14),
                transparent 35%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(10,67,112,0.28),
                transparent 38%
            ),
            var(--hb-bg);
        color: var(--hb-text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #061426 0%, #0a2342 100%
        );
        border-right: 1px solid var(--hb-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--hb-text);
    }

    h1, h2, h3, h4 {
        color: var(--hb-accent) !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--hb-text);
    }

    hr {
        border-color: var(--hb-border) !important;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: var(--hb-accent);
        border: 1px solid transparent;
        border-radius: 10px;
        color: var(--hb-bg) !important;
        font-weight: 750;
        min-height: 2.8rem;
        transition: all 0.15s ease;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: var(--hb-accent-hover);
        border-color: white;
        color: var(--hb-bg) !important;
        transform: translateY(-1px);
    }

    [data-testid="stFileUploaderDropzone"],
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: var(--hb-panel-soft) !important;
        border-color: var(--hb-border) !important;
        color: var(--hb-text) !important;
        border-radius: 10px;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: var(--hb-accent);
        color: var(--hb-bg);
        border: 0;
    }

    [data-testid="stFileUploaderFile"] {
        background: rgba(7,24,45,0.94) !important;
        border: 1px solid var(--hb-border) !important;
        border-radius: 10px !important;
        color: var(--hb-text) !important;
    }

    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFile"] small {
        color: var(--hb-text) !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploaderFile"] button {
        background: transparent !important;
        border: 0 !important;
        color: var(--hb-accent) !important;
    }

    [data-testid="stFileUploaderFile"] button svg {
        fill: var(--hb-accent) !important;
        stroke: var(--hb-accent) !important;
    }

    [data-baseweb="tab-list"] {
        gap: 0.3rem;
        border-bottom: 1px solid var(--hb-border);
    }

    [data-baseweb="tab"] {
        background: rgba(10,35,66,0.55);
        border-radius: 9px 9px 0 0;
        color: var(--hb-muted);
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background: var(--hb-panel);
        color: var(--hb-accent);
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--hb-accent);
    }

    [data-testid="stDataFrame"],
    [data-testid="stMetric"],
    [data-testid="stAlert"] {
        border: 1px solid var(--hb-border);
        border-radius: 12px;
        overflow: hidden;
    }

    .metric-card {
        background: var(--hb-panel);
        border: 1px solid var(--hb-border);
        padding: 14px 16px;
        border-radius: 14px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25);
        transition: transform 0.15s;
        margin-bottom: 8px;
    }

    .metric-card:hover {
        transform: scale(1.03);
    }

    .metric-number {
        font-size: 24px;
        font-weight: 700;
        color: #7FDBFF;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 12px;
        color: #b8cfdf;
        font-weight: 500;
    }

    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #7FDBFF;
        background: rgba(10,35,66,0.76);
        border: 1px solid rgba(127,219,255,0.24);
        border-radius: 12px;
        padding: 12px 15px;
        margin: 10px 0 14px;
    }

    .hero {
        padding: 4rem 0 2rem;
        text-align: center;
    }

    .hero h1 {
        color: #7FDBFF;
        font-size: clamp(2.4rem, 6vw, 4.6rem);
        line-height: 1;
        margin-bottom: 0.8rem;
    }

    .hero p {
        color: #d7e8f5;
        font-size: 1.15rem;
        margin: 0 auto;
        max-width: 680px;
    }

    .tool-card {
        background: rgba(10,35,66,0.92);
        border: 1px solid rgba(127,219,255,0.28);
        border-radius: 18px;
        min-height: 190px;
        padding: 1.5rem;
        box-shadow: 0 12px 35px rgba(0,0,0,0.28);
    }

    .tool-card h3 {
        color: #7FDBFF;
        margin-top: 0;
    }

    .tool-card p {
        color: #d7e8f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(text):
    st.markdown(
        f'<div class="section-header">{text}</div>',
        unsafe_allow_html=True,
    )


def metric_card(container, value, label):
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HOME
# =========================================================
def go_to(page_name):
    st.session_state.active_page = page_name


def render_home():
    st.markdown(
        """
        <div class="hero">
            <h1>Har-Ber Football Analytics</h1>
            <p>Choose the dashboard you want to open.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tools = [
        (
            "Basic Dashboard",
            "Review tendencies, game situations, and play "
            "predictions from uploaded Hudl data.",
        ),
        (
            "Advanced Analytics",
            "Explore explosive plays, success rates, formations, "
            "concepts, and field position.",
        ),
        (
            "Play Breakdown",
            "Break down plays by formation and formations "
            "by play selection.",
        ),
        (
            "Formation Tendencies",
            "Explore play direction, gap or zone scheme, launch "
            "point, RB path, and play call by formation.",
        ),
    ]

    # Two rows of cards keep all four dashboards easy to read.
    for start in range(0, len(tools), 2):
        card_columns = st.columns(2, gap="large")

        for container, (name, description) in zip(
            card_columns, tools[start:start + 2]
        ):
            with container:
                st.markdown(
                    f"""
                    <div class="tool-card">
                        <h3>{name}</h3>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Open {name}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(name,),
                    key=f"home_open_{name}",
                )

        st.write("")


# =========================================================
# BASIC DASHBOARD
# =========================================================
def render_basic_dashboard(uploaded_file):
    sidebar_logo()
    st.sidebar.title("Har-Ber Dashboard")

    if uploaded_file is None:
        st.info(
            "Upload a Hudl Excel file in the sidebar "
            "to open the dashboard."
        )
        return

    df = read_uploaded_excel(uploaded_file)
    df = standardize_columns(df)
    df = apply_global_play_settings(df)
    df = ensure_numeric_columns(
        df,
        ["down", "distance", "yardline", "gain_loss", "quarter"],
    )

    missing = [
        column for column in ["down", "yardline"]
        if column not in df.columns
    ]
    if missing:
        st.error(
            "Uploaded file is missing required columns: "
            + ", ".join(missing)
        )
        return

    df["yard_group"] = df["yardline"].apply(
        custom_yard_group
    )

    yard_order = [
        "0 - -9", "-10 - -19", "-20 - -29",
        "-30 - -39", "-40 - -50", "+50 - +40",
        "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0",
    ]

    st.sidebar.header("Filters")

    down_choices = sorted(df["down"].dropna().unique())
    if not down_choices:
        st.error("No valid down values found in uploaded file.")
        return

    down_selected = st.sidebar.selectbox(
        "Down", down_choices, key="basic_filter_down"
    )

    df_down = df[df["down"] == down_selected]

    yard_choices = [
        group for group in yard_order
        if group in df_down["yard_group"].unique()
    ]
    if not yard_choices:
        st.error(
            "No valid yard groups found for the selected down."
        )
        return

    yard_choice = st.sidebar.selectbox(
        "Yard Group", yard_choices, key="basic_filter_yard"
    )

    selected = df_down[
        df_down["yard_group"] == yard_choice
    ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overall Snapshot",
        "Filtered by Down/Yardline",
        "Success Heatmaps",
        "Concept Effectiveness",
        "Play Prediction",
    ])

    def render_snapshot(data, title, key_prefix):
        section_header(title)

        if (
            "gain_loss" in data.columns
            and data["gain_loss"].notna().any()
        ):
            c1, c2, c3 = st.columns(3)
            metric_card(
                c1, f"{data['gain_loss'].mean():.1f}",
                "Average Gain",
            )
            metric_card(
                c2, data["gain_loss"].max(), "Max Gain"
            )
            metric_card(
                c3, data["gain_loss"].min(), "Min Gain"
            )

            gain_summary = (
                data.groupby("gain_loss")
                .size()
                .reset_index(name="plays")
                .sort_values("gain_loss")
            )
            gain_fig = px.bar(
                gain_summary,
                x="gain_loss",
                y="plays",
                labels={
                    "gain_loss": "Yards Gained",
                    "plays": "Number of Plays",
                },
                title="Gain / Loss Distribution",
                color_discrete_sequence=["#7FDBFF"],
            )
        else:
            gain_fig = None

        if {"concept", "play_direction"}.issubset(data.columns):
            concepts = (
                data.groupby(["concept", "play_direction"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(8)
            )
            concept_fig = px.bar(
                concepts,
                x="count",
                y="concept",
                color="play_direction",
                orientation="h",
                title="Most Frequent Concepts by Play Direction",
                color_discrete_sequence=[
                    "#7FDBFF", "#0A2342", "#AAAAAA"
                ],
            )
        else:
            concept_fig = None

        c1, c2 = st.columns(2)
        with c1:
            if gain_fig is not None:
                show_chart(
                    gain_fig, key=f"{key_prefix}_gain"
                )
        with c2:
            if concept_fig is not None:
                show_chart(
                    concept_fig, key=f"{key_prefix}_direction"
                )

        section_header("Play Type & Concept Distribution")
        c1, c2 = st.columns(2)

        with c1:
            if "play_type" in data.columns:
                summary = (
                    data["play_type"].value_counts()
                    .rename_axis("play_type")
                    .reset_index(name="count")
                )
                fig = px.pie(
                    summary,
                    names="play_type",
                    values="count",
                    title="Play Type %",
                )
                show_chart(
                    fig, key=f"{key_prefix}_play_type"
                )

        with c2:
            if "concept" in data.columns:
                summary = (
                    data["concept"].value_counts()
                    .head(6)
                    .rename_axis("concept")
                    .reset_index(name="count")
                )
                fig = px.pie(
                    summary,
                    names="concept",
                    values="count",
                    title="Most Frequent Concepts",
                    color_discrete_sequence=(
                        px.colors.sequential.Blues
                    ),
                )
                show_chart(
                    fig, key=f"{key_prefix}_concept_pie"
                )

    with tab1:
        render_snapshot(
            df, "Overall Snapshot", "basic_overall"
        )

    with tab2:
        if selected.empty:
            st.warning("No plays for this selection.")
        else:
            render_snapshot(
                selected,
                "Filtered by Down / Yardline",
                "basic_filtered",
            )
            section_header("Raw Play Data")
            st.dataframe(
                selected, use_container_width=True
            )

    with tab3:
        section_header("Play Success Heatmap")

        with st.expander("How to read this chart"):
            st.write(
                "This heatmap shows success rate by down and "
                "field position. Success means gaining 4 or "
                "more yards. Use the color scale to compare rates."
            )

        if "gain_loss" not in df.columns:
            st.warning("No gain/loss data found.")
        else:
            success_data = df.dropna(
                subset=["gain_loss"]
            ).copy()
            success_data["success"] = (
                success_data["gain_loss"] >= 4
            )

            summary = (
                success_data.groupby(["down", "yard_group"])
                .agg(
                    success_rate=("success", "mean"),
                    num_plays=("success", "size"),
                )
                .reset_index()
            )

            if summary.empty:
                st.info("Not enough data to display heatmap.")
            else:
                rates = summary.pivot(
                    index="down",
                    columns="yard_group",
                    values="success_rate",
                )
                counts = summary.pivot(
                    index="down",
                    columns="yard_group",
                    values="num_plays",
                )
                existing_order = [
                    group for group in yard_order
                    if group in rates.columns
                ]
                rates = rates.reindex(columns=existing_order)
                counts = counts.reindex(
                    index=rates.index, columns=existing_order
                )

                fig = px.imshow(
                    rates,
                    text_auto=".0%",
                    aspect="auto",
                    labels={
                        "x": "Yard Group",
                        "y": "Down",
                        "color": "Success Rate",
                    },
                    color_continuous_scale="Blues",
                )
                fig.update_traces(
                    customdata=counts.values,
                    hovertemplate=(
                        "<b>Down:</b> %{y}<br>"
                        "<b>Yard Group:</b> %{x}<br>"
                        "<b>Success Rate:</b> %{z:.0%}<br>"
                        "<b>Number of Plays:</b> %{customdata}"
                        "<extra></extra>"
                    ),
                )
                show_chart(fig, key="basic_success_heatmap")

    with tab4:
        section_header("Concept Effectiveness")

        with st.expander("How to read this chart"):
            st.write(
                "X-axis: Success Rate. "
                "Y-axis: Average yards gained. "
                "Bubble size: Number of plays."
            )

        if not {"concept", "gain_loss"}.issubset(df.columns):
            st.warning(
                "Need concept and gain/loss columns "
                "for this chart."
            )
        else:
            concept_stats = (
                df.dropna(subset=["gain_loss"])
                .groupby("concept")
                .agg(
                    avg_gain=("gain_loss", "mean"),
                    success_rate=(
                        "gain_loss",
                        lambda values: (values >= 4).mean(),
                    ),
                    plays=("gain_loss", "size"),
                )
                .reset_index()
                .sort_values("avg_gain", ascending=False)
            )

            if concept_stats.empty:
                st.info("No usable concept gain data found.")
            else:
                fig = px.scatter(
                    concept_stats,
                    x="success_rate",
                    y="avg_gain",
                    size="plays",
                    hover_name="concept",
                    title="Concept Effectiveness",
                )
                show_chart(fig, key="basic_concept_bubble")
                st.dataframe(
                    concept_stats, use_container_width=True
                )

    with tab5:
        render_play_prediction(df)


def render_play_prediction(df):
    from sklearn.metrics import accuracy_score

    section_header("ELITE Play Prediction System")

    try:
        base_df = load_base_data()
    except Exception as error:
        st.error(
            f"Error loading AllPlaysTrainData.csv: {error}"
        )
        return

    try:
        (
            model, int_to_play, accuracy,
            X_test, y_test, features, n_classes,
        ) = train_model(base_df, df.copy())
    except Exception as error:
        st.error(f"Model training failed: {error}")
        return

    st.success(f"Overall Model Accuracy: {accuracy:.2%}")

    st.markdown("### Predict Play")
    c1, c2, c3 = st.columns(3)

    with c1:
        pred_down = st.selectbox(
            "Down", [1, 2, 3, 4], key="prediction_down"
        )
    with c2:
        pred_dist = st.number_input(
            "Distance",
            min_value=1,
            max_value=30,
            value=5,
            key="prediction_distance",
        )
    with c3:
        pred_yard = st.number_input(
            "Yardline",
            min_value=-50,
            max_value=50,
            value=0,
            key="prediction_yardline",
        )

    input_df = pd.DataFrame([{
        "down": pred_down,
        "distance": pred_dist,
        "yardline": pred_yard,
    }])
    input_df = add_prediction_features(input_df)

    probabilities = model.predict_proba(
        input_df[features]
    )[0]

    if n_classes == 2 and len(probabilities) == 1:
        probabilities = np.array([
            1 - probabilities[0], probabilities[0]
        ])

    predicted_class = int(np.argmax(probabilities))
    predicted_play = int_to_play[predicted_class]
    confidence = float(np.max(probabilities))

    mask = (
        (X_test["down"] == pred_down)
        & ((X_test["distance"] - pred_dist).abs() <= 2)
        & ((X_test["yardline"] - pred_yard).abs() <= 10)
    )

    similar_X = X_test[mask]
    similar_y = y_test[mask]

    situation_accuracy = None
    if len(similar_X) >= 10:
        situation_accuracy = accuracy_score(
            similar_y, model.predict(similar_X)
        )

    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Play", predicted_play)
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Model Accuracy", f"{accuracy:.1%}")

    if situation_accuracy is not None:
        st.metric(
            "Situation Accuracy",
            f"{situation_accuracy:.1%}",
        )
    else:
        st.info(
            "Not enough similar plays for situation accuracy."
        )

    st.markdown("### Top 3 Plays")
    for index in np.argsort(probabilities)[::-1][:3]:
        if index in int_to_play:
            st.write(
                f"{int_to_play[index]} — "
                f"{probabilities[index]:.1%}"
            )

    st.markdown("### Full Probabilities")
    probability_df = pd.DataFrame([
        {
            "Play": int_to_play[index],
            "Probability": probabilities[index],
        }
        for index in range(len(probabilities))
        if index in int_to_play
    ]).sort_values("Probability", ascending=False)

    probability_df["Probability"] = (
        probability_df["Probability"]
        .map(lambda value: f"{value:.2%}")
    )
    st.dataframe(
        probability_df, use_container_width=True
    )

    st.markdown("### Situation Insight")
    if pred_down == 3 and pred_dist >= 7:
        st.info(
            "Likely passing tendency: long-yardage third down."
        )
    elif pred_down == 1 and pred_dist <= 3:
        st.info(
            "Likely run tendency: short-yardage first down."
        )
    elif pred_down == 4:
        st.info(
            "Fourth down can be more volatile because "
            "of smaller sample sizes."
        )


# =========================================================
# ADVANCED ANALYTICS
# =========================================================
def render_advanced_analytics(uploaded_file):
    sidebar_logo()
    st.sidebar.title("Har-Ber Advanced Analytics")

    if uploaded_file is None:
        st.info(
            "Upload a Hudl Excel file in the sidebar "
            "to open the dashboard."
        )
        return

    df = read_uploaded_excel(uploaded_file)
    df = standardize_columns(df)
    df = apply_global_play_settings(df)
    df = ensure_numeric_columns(
        df, ["gain_loss", "down", "yardline"]
    )

    if "gain_loss" not in df.columns:
        st.error(
            "No gain/loss column found in uploaded file."
        )
        return

    def advanced_yard_group(yardline):
        if pd.isna(yardline):
            return "Unknown"
        if 0 <= yardline <= 9:
            return "0 - 9"
        elif 10 <= yardline <= 19:
            return "10 - 19"
        elif 20 <= yardline <= 29:
            return "20 - 29"
        elif 30 <= yardline <= 39:
            return "30 - 39"
        elif 40 <= yardline <= 50:
            return "40 - 50"
        elif -9 <= yardline <= 0:
            return "-9 - 0"
        elif -19 <= yardline <= -10:
            return "-19 - -10"
        elif -29 <= yardline <= -20:
            return "-29 - -20"
        elif -39 <= yardline <= -30:
            return "-39 - -30"
        elif -50 <= yardline <= -40:
            return "-50 - -40"
        return "Other"

    if "yardline" in df.columns:
        df["yard_group"] = df["yardline"].apply(
            advanced_yard_group
        )
    else:
        df["yard_group"] = "Unknown"

    yard_order = [
        "-50 - -40", "-39 - -30", "-29 - -20",
        "-19 - -10", "-9 - 0", "0 - 9",
        "10 - 19", "20 - 29", "30 - 39", "40 - 50",
    ]

    # Missing gains remain missing rather than becoming zero-yard plays.
    df["success"] = (
        df["gain_loss"].ge(4)
        .astype(float)
        .where(df["gain_loss"].notna())
    )

    if "play_type" in df.columns:
        play_types = (
            df["play_type"].astype("string")
            .str.strip().str.lower()
        )
    else:
        play_types = pd.Series(
            pd.NA, index=df.index, dtype="string"
        )

    run_mask = play_types.eq("run").fillna(False)
    pass_mask = play_types.eq("pass").fillna(False)

    df["explosive"] = np.nan
    valid_gains = df["gain_loss"].notna()

    df.loc[run_mask & valid_gains, "explosive"] = (
        df.loc[run_mask & valid_gains, "gain_loss"]
        .ge(10).astype(float)
    )
    df.loc[pass_mask & valid_gains, "explosive"] = (
        df.loc[pass_mask & valid_gains, "gain_loss"]
        .ge(20).astype(float)
    )

    down_order = (
        sorted(df["down"].dropna().unique())
        if "down" in df.columns else []
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Explosive & Success Metrics",
        "Gain/Loss Breakdown",
        "Concept by Yardline",
        "Formation Breakdown",
        "Concept Breakdown",
    ])

    def rate_heatmap(data, value_column, title, key):
        if "down" not in data.columns:
            st.info("A down column is required for this heatmap.")
            return

        data = data.dropna(subset=[value_column])

        if data.empty:
            st.info(f"No usable data for {title}.")
            return

        summary = (
            data.groupby(["down", "yard_group"])
            .agg(
                num_plays=("gain_loss", "size"),
                avg_gain=("gain_loss", "mean"),
                rate=(value_column, "mean"),
            )
            .reset_index()
        )

        if summary.empty:
            st.info(f"No usable data for {title}.")
            return

        rates = summary.pivot(
            index="down", columns="yard_group", values="rate"
        ) * 100
        counts = summary.pivot(
            index="down",
            columns="yard_group",
            values="num_plays",
        )
        gains = summary.pivot(
            index="down",
            columns="yard_group",
            values="avg_gain",
        )

        ordered_columns = [
            group for group in yard_order
            if group in rates.columns
        ] + [
            group for group in rates.columns
            if group not in yard_order
        ]

        rates = rates.reindex(columns=ordered_columns)
        counts = counts.reindex(
            index=rates.index, columns=rates.columns
        )
        gains = gains.reindex(
            index=rates.index, columns=rates.columns
        )

        fig = px.imshow(
            rates,
            text_auto=".1f",
            aspect="auto",
            labels={
                "x": "Yard Group",
                "y": "Down",
                "color": title,
            },
            color_continuous_scale="Blues",
            title=title,
        )
        fig.update_traces(
            customdata=np.stack(
                [counts.values, gains.values], axis=-1
            ),
            hovertemplate=(
                "<b>Down:</b> %{y}<br>"
                "<b>Yard Group:</b> %{x}<br>"
                "<b>Rate:</b> %{z:.1f}%<br>"
                "<b>Plays:</b> %{customdata[0]:.0f}<br>"
                "<b>Avg Gain:</b> %{customdata[1]:.1f}y"
                "<extra></extra>"
            ),
        )
        show_chart(fig, key=key)

    with tab1:
        section_header("Explosive Play & Success Metrics")
        st.markdown(
            "**Definitions:**\n"
            "- **Explosive Plays:** Runs ≥ 10 yards, "
            "Passes ≥ 20 yards\n"
            "- **Success:** Plays gaining ≥ 4 yards"
        )

        explosive_runs = df.loc[
            run_mask, "explosive"
        ].mean() * 100
        explosive_passes = df.loc[
            pass_mask, "explosive"
        ].mean() * 100
        success_percent = df["success"].mean() * 100

        def rate_label(value):
            return "N/A" if pd.isna(value) else f"{value:.1f}%"

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, len(df), "Total Plays")
        metric_card(
            c2, rate_label(explosive_runs), "Explosive Runs"
        )
        metric_card(
            c3, rate_label(explosive_passes), "Explosive Passes"
        )
        metric_card(
            c4, rate_label(success_percent), "Overall Success %"
        )

        st.markdown("### Explosive Plays")
        c1, c2 = st.columns(2)
        with c1:
            rate_heatmap(
                df[run_mask],
                "explosive",
                "Run Explosive %",
                "advanced_run_explosive",
            )
        with c2:
            rate_heatmap(
                df[pass_mask],
                "explosive",
                "Pass Explosive %",
                "advanced_pass_explosive",
            )

        st.markdown("### Success Rate")
        rate_heatmap(
            df,
            "success",
            "Success Rate %",
            "advanced_success",
        )

    with tab2:
        if "down" not in df.columns:
            st.info("A down column is required for this chart.")
        else:
            summary = (
                df.groupby(["down", "yard_group"])
                .agg(
                    avg_gain=("gain_loss", "mean"),
                    num_plays=("gain_loss", "count"),
                )
                .reset_index()
            )

            if summary.empty:
                st.info("No usable gain/loss breakdown found.")
            else:
                gains = summary.pivot(
                    index="down",
                    columns="yard_group",
                    values="avg_gain",
                )
                counts = summary.pivot(
                    index="down",
                    columns="yard_group",
                    values="num_plays",
                ).reindex(
                    index=gains.index, columns=gains.columns
                )

                fig = px.imshow(
                    gains,
                    text_auto=".1f",
                    color_continuous_scale="Blues",
                    labels={
                        "x": "Yard Group",
                        "y": "Down",
                        "color": "Avg Gain",
                    },
                    title=(
                        "Average Gain / Loss by Down & Yard Group"
                    ),
                )
                fig.update_traces(
                    customdata=counts.values,
                    hovertemplate=(
                        "<b>Down:</b> %{y}<br>"
                        "<b>Yard Group:</b> %{x}<br>"
                        "<b>Avg Gain:</b> %{z:.1f} yards<br>"
                        "<b>Recorded Gains:</b> %{customdata:.0f}"
                        "<extra></extra>"
                    ),
                )
                show_chart(fig, key="advanced_gain_heatmap")

    with tab3:
        section_header("Concept Effectiveness by Field Zone")

        if "concept" not in df.columns:
            st.info("A concept column is required.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                min_plays = st.number_input(
                    "Min Plays to Show Concept",
                    min_value=1,
                    value=2,
                    step=1,
                    key="advanced_concept_min_plays",
                )
            with c2:
                metric = st.selectbox(
                    "Select Metric",
                    [
                        "Success Rate %",
                        "Average Gain",
                        "Explosive Play %",
                    ],
                    key="advanced_concept_metric",
                )

            summary = (
                df.groupby(["concept", "yard_group"])
                .agg(
                    plays=("gain_loss", "size"),
                    avg_gain=("gain_loss", "mean"),
                    success_rate=("success", "mean"),
                    explosive_rate=("explosive", "mean"),
                )
                .reset_index()
            )
            summary = summary[
                summary["plays"] >= min_plays
            ].copy()

            if summary.empty:
                st.warning("No concepts met the threshold.")
            else:
                if metric == "Success Rate %":
                    summary["display_val"] = (
                        summary["success_rate"] * 100
                    ).round(1)
                elif metric == "Explosive Play %":
                    summary["display_val"] = (
                        summary["explosive_rate"] * 100
                    ).round(1)
                else:
                    summary["display_val"] = (
                        summary["avg_gain"].round(1)
                    )

                values = summary.pivot(
                    index="concept",
                    columns="yard_group",
                    values="display_val",
                )
                counts = summary.pivot(
                    index="concept",
                    columns="yard_group",
                    values="plays",
                )
                order = [
                    group for group in yard_order
                    if group in values.columns
                ] + [
                    group for group in values.columns
                    if group not in yard_order
                ]
                values = values.reindex(columns=order)
                counts = counts.reindex(
                    index=values.index, columns=order
                )

                fig = px.imshow(
                    values,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels={
                        "x": "Field Zone",
                        "y": "Play Concept",
                        "color": metric,
                    },
                    title=f"{metric} by Concept and Yardline",
                )
                fig.update_traces(
                    customdata=counts.values,
                    hovertemplate=(
                        "<b>Concept:</b> %{y}<br>"
                        "<b>Yard Group:</b> %{x}<br>"
                        "<b>Value:</b> %{z}<br>"
                        "<b>Plays:</b> %{customdata:.0f}"
                        "<extra></extra>"
                    ),
                )
                show_chart(
                    fig, key="advanced_concept_yardline"
                )

            overall = (
                df.groupby("concept")
                .agg(
                    plays=("gain_loss", "size"),
                    avg_gain=("gain_loss", "mean"),
                    success_pct=("success", "mean"),
                    explosive_pct=("explosive", "mean"),
                )
                .reset_index()
            )
            overall = overall[overall["plays"] >= 3]
            st.dataframe(
                overall.sort_values(
                    "success_pct", ascending=False
                ),
                use_container_width=True,
            )

    with tab4:
        if not {"formation", "down"}.issubset(df.columns):
            st.info("Formation and down columns are required.")
        else:
            summary = (
                df.groupby(["formation", "down"])
                .agg(
                    avg_gain=("gain_loss", "mean"),
                    num_plays=("gain_loss", "count"),
                )
                .reset_index()
            )
            summary = summary[
                summary["num_plays"] >= 2
            ]

            if summary.empty:
                st.info(
                    "No formation/down combinations "
                    "have at least two recorded gains."
                )
            else:
                values = summary.pivot(
                    index="formation",
                    columns="down",
                    values="avg_gain",
                )
                counts = summary.pivot(
                    index="formation",
                    columns="down",
                    values="num_plays",
                )

                order = [
                    down for down in down_order
                    if down in values.columns
                ]
                values = values.reindex(columns=order)
                counts = counts.reindex(
                    index=values.index, columns=order
                )

                fig = px.imshow(
                    values,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels={
                        "x": "Down",
                        "y": "Formation",
                        "color": "Avg Gain",
                    },
                    title="Average Gain by Formation and Down",
                )
                fig.update_traces(
                    customdata=counts.values,
                    hovertemplate=(
                        "<b>Formation:</b> %{y}<br>"
                        "<b>Down:</b> %{x}<br>"
                        "<b>Avg Gain:</b> %{z:.1f}y<br>"
                        "<b>Recorded Gains:</b> %{customdata:.0f}"
                        "<extra></extra>"
                    ),
                )
                show_chart(
                    fig, key="advanced_formation_down"
                )

            overall = (
                df.groupby("formation")
                .agg(
                    plays=("gain_loss", "size"),
                    avg_gain=("gain_loss", "mean"),
                    success_pct=("success", "mean"),
                    explosive_pct=("explosive", "mean"),
                )
                .reset_index()
            )
            overall = overall[overall["plays"] >= 3]
            st.dataframe(
                overall.sort_values(
                    "success_pct", ascending=False
                ),
                use_container_width=True,
            )

    with tab5:
        section_header("Concept Breakdown")

        if "concept" not in df.columns:
            st.info("A concept column is required.")
        else:
            summary = (
                df.groupby("concept")
                .agg(
                    plays=("gain_loss", "size"),
                    avg_gain=("gain_loss", "mean"),
                    success_pct=("success", "mean"),
                    explosive_pct=("explosive", "mean"),
                )
                .reset_index()
            )
            summary = summary[summary["plays"] >= 3]

            st.dataframe(
                summary.sort_values(
                    "success_pct", ascending=False
                ),
                use_container_width=True,
            )

            for metric, title, key in [
                (
                    "success_pct",
                    "Concept Success Rate",
                    "advanced_concept_success",
                ),
                (
                    "explosive_pct",
                    "Concept Explosive Play %",
                    "advanced_concept_explosive",
                ),
            ]:
                chart_data = (
                    summary.dropna(subset=[metric])
                    .sort_values(metric)
                )
                if chart_data.empty:
                    st.info(f"No usable data for {title}.")
                    continue

                fig = px.bar(
                    chart_data,
                    x=metric,
                    y="concept",
                    orientation="h",
                    title=title,
                    labels={metric: "Rate"},
                    color=metric,
                    color_continuous_scale="Blues",
                    custom_data=["plays", "avg_gain"],
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>Concept:</b> %{y}<br>"
                        "<b>Rate:</b> %{x:.1%}<br>"
                        "<b>Plays:</b> %{customdata[0]}<br>"
                        "<b>Avg Gain:</b> %{customdata[1]:.1f}y"
                        "<extra></extra>"
                    )
                )
                show_chart(fig, key=key)


# =========================================================
# PLAY BREAKDOWN HELPERS
# =========================================================
def render_custom_grouping(df):
    if "custom_breakdown_groups" not in st.session_state:
        st.session_state.custom_breakdown_groups = []

    if "custom_breakdown_group_id" not in st.session_state:
        st.session_state.custom_breakdown_group_id = 1

    with st.expander("Custom Grouping", expanded=False):
        st.caption(
            "Create multiple temporary groups. Each group can "
            "be turned on or off independently, and your "
            "uploaded file is never changed."
        )

        grouping_type = st.radio(
            "What do you want to combine?",
            ["Plays", "Formations"],
            horizontal=True,
            key="new_custom_grouping_type",
        )
        column = (
            "concept" if grouping_type == "Plays"
            else "formation"
        )
        options = sorted(
            df[column].dropna().unique().tolist()
        )

        selected_items = st.multiselect(
            f"Select {grouping_type.lower()} to combine",
            options,
            key=f"new_custom_{column}_items",
            help="Select at least two items.",
        )

        group_name = st.text_input(
            "Combined group name",
            value="",
            placeholder="Example: Counter Family",
            key=f"new_custom_{column}_name",
        ).strip()

        if st.button(
            "Add custom group", use_container_width=True
        ):
            if len(selected_items) < 2:
                st.warning(
                    "Select at least two items "
                    "before adding the group."
                )
            elif not group_name:
                st.warning(
                    "Enter a name for the combined group."
                )
            else:
                used_items = {
                    item
                    for group in (
                        st.session_state.custom_breakdown_groups
                    )
                    if group["column"] == column
                    for item in group["items"]
                }
                overlaps = sorted(
                    set(selected_items) & used_items
                )

                if overlaps:
                    st.warning(
                        "These items are already assigned "
                        "to another custom group: "
                        + ", ".join(overlaps)
                        + ". Remove that group before "
                        "reassigning them."
                    )
                else:
                    group_id = (
                        st.session_state.custom_breakdown_group_id
                    )
                    st.session_state.custom_breakdown_groups.append({
                        "id": group_id,
                        "type": grouping_type,
                        "column": column,
                        "name": group_name,
                        "items": list(selected_items),
                        "enabled": True,
                    })
                    st.session_state.custom_breakdown_group_id += 1
                    st.rerun()

        st.divider()
        st.markdown("#### Saved custom groups")

        if not st.session_state.custom_breakdown_groups:
            st.info("No custom groups have been added yet.")
        else:
            for group in list(
                st.session_state.custom_breakdown_groups
            ):
                enabled_col, details_col, remove_col = (
                    st.columns([1, 4, 1])
                )

                with enabled_col:
                    st.toggle(
                        "On",
                        value=group.get("enabled", True),
                        key=f"custom_group_enabled_{group['id']}",
                        label_visibility="collapsed",
                    )

                with details_col:
                    st.markdown(
                        f"**{group['name']}** · {group['type']}"
                    )
                    st.caption(" + ".join(group["items"]))

                with remove_col:
                    if st.button(
                        "Remove",
                        key=f"remove_custom_group_{group['id']}",
                    ):
                        st.session_state.custom_breakdown_groups = [
                            saved
                            for saved in (
                                st.session_state.custom_breakdown_groups
                            )
                            if saved["id"] != group["id"]
                        ]
                        st.session_state.pop(
                            f"custom_group_enabled_{group['id']}",
                            None,
                        )
                        st.rerun()

            if st.button(
                "Clear all custom groups",
                use_container_width=True,
            ):
                for group in (
                    st.session_state.custom_breakdown_groups
                ):
                    st.session_state.pop(
                        f"custom_group_enabled_{group['id']}",
                        None,
                    )
                st.session_state.custom_breakdown_groups = []
                st.rerun()


def breakdown_table_config(category):
    return {
        category: (
            "Formation" if category == "formation" else "Play"
        ),
        "plays": st.column_config.NumberColumn(
            "Plays", format="%d"
        ),
        "percentage": st.column_config.NumberColumn(
            "Usage %", format="%.1f%%"
        ),
        "average_gain": st.column_config.NumberColumn(
            "Average Gain", format="%.1f yds"
        ),
        "longest_play": st.column_config.NumberColumn(
            "Longest Play", format="%.1f yds"
        ),
        "shortest_play": st.column_config.NumberColumn(
            "Shortest Play", format="%.1f yds"
        ),
    }


def render_selection_breakdown(
    df,
    select_column,
    group_column,
    select_label,
    select_key,
):
    selected_value = st.selectbox(
        select_label,
        sorted(df[select_column].unique().tolist()),
        key=select_key,
    )

    selected = df[
        df[select_column] == selected_value
    ]

    aggregations = {
        "plays": (group_column, "size"),
    }
    if "gain_loss" in selected.columns:
        aggregations.update({
            "average_gain": ("gain_loss", "mean"),
            "longest_play": ("gain_loss", "max"),
            "shortest_play": ("gain_loss", "min"),
        })

    summary = (
        selected.groupby(group_column, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("plays", ascending=False)
    )

    total = len(selected)
    summary["percentage"] = (
        summary["plays"] / total * 100
    ).round(1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Plays", f"{total:,}")
    c2.metric(
        "Formations Used"
        if group_column == "formation" else "Different Plays",
        summary[group_column].nunique(),
    )

    if (
        "gain_loss" in selected.columns
        and selected["gain_loss"].notna().any()
    ):
        gains = selected["gain_loss"]
        c3.metric("Average Gain", f"{gains.mean():.1f} yds")
        c4.metric("Longest Play", f"{gains.max():.1f} yds")
        c5.metric("Shortest Play", f"{gains.min():.1f} yds")
    else:
        c3.metric("Average Gain", "N/A")
        c4.metric("Longest Play", "N/A")
        c5.metric("Shortest Play", "N/A")

    title = (
        f"Formations Used for {selected_value}"
        if group_column == "formation"
        else f"Plays Run from {selected_value}"
    )
    category_label = (
        "Formation" if group_column == "formation" else "Play"
    )

    fig = px.bar(
        summary.sort_values("plays"),
        x="plays",
        y=group_column,
        orientation="h",
        text="percentage",
        title=title,
        labels={
            "plays": "Number of Plays",
            group_column: category_label,
        },
        color="plays",
        color_continuous_scale="Blues",
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            f"<b>{category_label}:</b> %{{y}}<br>"
            "<b>Plays:</b> %{x}<br>"
            "<b>Usage:</b> %{text:.1f}%"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title=None,
    )

    show_chart(fig, key=f"{select_key}_chart")
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config=breakdown_table_config(group_column),
    )


def render_averages(df, category):
    label = "Formation" if category == "formation" else "Play"

    st.subheader(f"{label} Averages")
    st.write(
        f"Compare every {label.lower()} by average gain, "
        "shortest gain, longest gain, and number of plays."
    )

    if (
        "gain_loss" not in df.columns
        or not df["gain_loss"].notna().any()
    ):
        st.warning(
            f"{label} averages require a gain/loss column "
            "such as GN/LS, Gain Loss, or Gain/Loss."
        )
        return

    averages = (
        df.groupby(category, dropna=False)
        .agg(
            average_gain=("gain_loss", "mean"),
            shortest_gain=("gain_loss", "min"),
            longest_gain=("gain_loss", "max"),
            number_of_plays=(category, "size"),
        )
        .reset_index()
        .sort_values(
            ["average_gain", "number_of_plays"],
            ascending=[False, False],
        )
    )

    for column in [
        "average_gain", "shortest_gain", "longest_gain"
    ]:
        averages[column] = averages[column].round(1)

    fig = px.bar(
        averages.sort_values("average_gain"),
        x="average_gain",
        y=category,
        orientation="h",
        color="average_gain",
        color_continuous_scale="Blues",
        text="average_gain",
        title=f"Average Gain by {label}",
        labels={
            category: label,
            "average_gain": "Average Gain",
        },
        custom_data=[
            "number_of_plays", "shortest_gain", "longest_gain"
        ],
    )
    fig.update_traces(
        texttemplate="%{text:.1f} yds",
        textposition="outside",
        hovertemplate=(
            f"<b>{label}:</b> %{{y}}<br>"
            "<b>Average Gain:</b> %{x:.1f} yds<br>"
            "<b>Number of Plays:</b> %{customdata[0]}<br>"
            "<b>Shortest Gain:</b> %{customdata[1]:.1f} yds<br>"
            "<b>Longest Gain:</b> %{customdata[2]:.1f} yds"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title=None,
    )

    show_chart(fig, key=f"breakdown_{category}_averages")

    st.dataframe(
        averages,
        use_container_width=True,
        hide_index=True,
        column_config={
            category: label,
            "average_gain": st.column_config.NumberColumn(
                "Average Gain", format="%.1f yds"
            ),
            "shortest_gain": st.column_config.NumberColumn(
                "Shortest Gain", format="%.1f yds"
            ),
            "longest_gain": st.column_config.NumberColumn(
                "Longest Gain", format="%.1f yds"
            ),
            "number_of_plays": st.column_config.NumberColumn(
                "Number of Plays", format="%d"
            ),
        },
    )


# =========================================================
# PLAY BREAKDOWN
# =========================================================
def render_play_breakdown(uploaded_file):
    st.title("Play Breakdown")
    st.caption(
        "Analyze which formations are used for each play "
        "and which plays are used from each formation."
    )

    if uploaded_file is None:
        st.info(
            "Upload a Hudl Excel file from the sidebar to begin."
        )
        return

    df = read_uploaded_excel(uploaded_file)
    df = standardize_columns(df)

    missing = [
        column for column in ["formation", "concept"]
        if column not in df.columns
    ]
    if missing:
        st.error(
            "The uploaded file is missing the following "
            "required columns: " + ", ".join(missing)
        )
        return

    for column in ["formation", "concept"]:
        df[column] = (
            df[column].astype("string")
            .str.strip().replace("", pd.NA)
        )

    df = ensure_numeric_columns(df, ["gain_loss"])
    df = df.dropna(subset=["formation", "concept"])
    df = apply_global_play_settings(df)

    # Grouping rules can leave a play name empty.
    df["concept"] = df["concept"].replace("", pd.NA)
    df = df.dropna(subset=["concept"])

    if df.empty:
        st.warning(
            "No usable formation and play data were found."
        )
        return

    if st.session_state.get("ignore_rpo_tag", True):
        st.success(
            "RPO-tagged plays are currently grouped "
            "with their base plays."
        )
    else:
        st.info(
            "RPO-tagged plays are currently shown separately."
        )

    render_custom_grouping(df)
    df, active_names = apply_custom_breakdown_groups(df)

    if active_names:
        st.success(
            "Active custom groups: " + ", ".join(active_names)
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Plays by Formation",
        "Formations by Play",
        "Formation Averages",
        "Play Averages",
    ])

    with tab1:
        st.subheader("Plays by Formation")
        st.write(
            "Select a play to see which formations "
            "it was run from."
        )
        render_selection_breakdown(
            df,
            select_column="concept",
            group_column="formation",
            select_label="Select a play",
            select_key="breakdown_selected_play",
        )

    with tab2:
        st.subheader("Formations by Play")
        st.write(
            "Select a formation to see which plays "
            "were run from it."
        )
        render_selection_breakdown(
            df,
            select_column="formation",
            group_column="concept",
            select_label="Select a formation",
            select_key="breakdown_selected_formation",
        )

    with tab3:
        render_averages(df, "formation")

    with tab4:
        render_averages(df, "concept")


# =========================================================
# NEW DASHBOARD: FORMATION TENDENCIES
# =========================================================
def render_formation_tendencies(uploaded_file):
    st.title("Formation Tendencies")

    if uploaded_file is None:
        st.info("Upload your Excel file in the sidebar.")
        return

    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")

    df.columns = [
        str(column).strip().upper()
        for column in df.columns
    ]

    required = [
        "OFF FORM",
        "OFF PLAY",
        "HASH",
        "PLAY TYPE",
        "OFF STR",
        "PLAY DIR",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    # Allow files without these charting fields.
    for column in ["GAP OR ZONE", "RB PATH", "PASS ZONE"]:
        if column not in df.columns:
            df[column] = pd.NA

    if "GN/LS" in df.columns:
        df["gain_loss"] = pd.to_numeric(
            df["GN/LS"], errors="coerce"
        )

    text_columns = required + ["GAP OR ZONE", "RB PATH"]

    for column in text_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    for column in [
        "OFF STR",
        "PLAY DIR",
        "PLAY TYPE",
        "GAP OR ZONE",
        "RB PATH",
    ]:
        df[column] = df[column].str.upper()

    df["PASS ZONE"] = pd.to_numeric(
        df["PASS ZONE"], errors="coerce"
    )

    df = df.rename(columns={
        "OFF FORM": "formation",
        "OFF PLAY": "concept",
    })

    # ---------------------------------------------
    # Automatic formation grouping
    # ---------------------------------------------
    col_lt, col_numbers = st.columns(2)

    with col_lt:
        group_lt = st.toggle(
            "Combine LT formations",
            value=True,
            key="tendencies_auto_group_lt",
            help="Combine LT DOS with DOS, for example.",
        )

    with col_numbers:
        group_numbers = st.toggle(
            "Combine numbered formations as EMPTY",
            value=True,
            key="tendencies_auto_group_numbers",
            help=(
                "Combine DING 1 and DING 2 as DING EMPTY. "
                "Unnumbered DING stays separate."
            ),
        )

    formations = (
        df["formation"]
        .astype("string")
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    if group_lt:
        formations = formations.str.replace(
            r"^LT\b[\s_-]*",
            "",
            regex=True,
        )

    if group_numbers:
        numbered = formations.str.contains(
            r"\b\d+\b",
            regex=True,
            na=False,
        )

        numbered_names = (
            formations.loc[numbered]
            .str.replace(r"\b\d+\b", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(" -_/")
            .str.replace(r"\s+EMPTY$", "", regex=True)
        )

        formations.loc[numbered] = numbered_names + " EMPTY"

    df["formation"] = formations.replace("", pd.NA)

    # ---------------------------------------------
    # Manual grouping
    # ---------------------------------------------
    render_custom_grouping(df)

    df, active_names = apply_custom_breakdown_groups(df)

    if active_names:
        st.caption(
            "Active custom groups: " + ", ".join(active_names)
        )

    # ---------------------------------------------
    # Filters
    # ---------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        hash_options = ["All"] + sorted(
            df["HASH"].dropna().unique().tolist()
        )

        hash_key = "simple_tendencies_hash"

        if (
            st.session_state.get(hash_key, "All")
            not in hash_options
        ):
            st.session_state.pop(hash_key, None)

        selected_hash = st.selectbox(
            "Hash",
            hash_options,
            key=hash_key,
        )

    if selected_hash != "All":
        df = df[df["HASH"] == selected_hash].copy()

    # Snapshot includes all formations within the selected hash.
    overall_data = df.copy()

    with col2:
        formation_options = ["All"] + sorted(
            df["formation"].dropna().unique().tolist()
        )

        formation_key = "simple_tendencies_formation"

        if (
            st.session_state.get(formation_key, "All")
            not in formation_options
        ):
            st.session_state.pop(formation_key, None)

        selected_formation = st.selectbox(
            "Formation",
            formation_options,
            key=formation_key,
        )

    if selected_formation != "All":
        df = df[
            df["formation"] == selected_formation
        ].copy()

    # ---------------------------------------------
    # Shared helpers
    # ---------------------------------------------
    def strength_percentages(plays):
        strength = plays["OFF STR"]
        direction = plays["PLAY DIR"]

        strength_total = int(
            strength.isin(["L", "R", "BAL"]).sum()
        )

        direction_total = int(
            direction.isin(["L", "R"]).sum()
        )

        result = {}

        for side, label in [
            ("L", "Strength L %"),
            ("R", "Strength R %"),
            ("BAL", "Balanced %"),
        ]:
            count = int(
                strength.eq(side).fillna(False).sum()
            )

            result[label] = (
                count / strength_total * 100
                if strength_total
                else float("nan")
            )

        for side in ["L", "R"]:
            count = int(
                direction.eq(side).fillna(False).sum()
            )

            result[f"Direction {side} %"] = (
                count / direction_total * 100
                if direction_total
                else float("nan")
            )

        comparable = (
            strength.isin(["L", "R"])
            & direction.isin(["L", "R"])
        )

        matched = (
            comparable
            & strength.eq(direction).fillna(False)
        )

        eligible_count = int(comparable.sum())
        match_count = int(matched.sum())

        result["At Strength %"] = (
            match_count / eligible_count * 100
            if eligible_count
            else float("nan")
        )

        result["Away From Strength %"] = (
            100 - result["At Strength %"]
            if eligible_count
            else float("nan")
        )

        return result

    def percentage_config(columns):
        return {
            column: st.column_config.NumberColumn(
                column,
                format="%.1f%%",
            )
            for column in columns
            if str(column).endswith("%")
        }

    def display_distribution(data, field, label):
        st.subheader(label)

        values = data[field].dropna()

        if values.empty:
            st.info(f"No recorded {label.lower()} data.")
            return

        summary = (
            values.value_counts()
            .rename_axis(label)
            .reset_index(name="Plays")
        )

        summary["Percentage %"] = (
            summary["Plays"] / summary["Plays"].sum() * 100
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Plays": st.column_config.NumberColumn(
                    "Plays", format="%d"
                ),
                "Percentage %": st.column_config.NumberColumn(
                    "Percentage %", format="%.1f%%"
                ),
            },
        )

    # ---------------------------------------------
    # Five tabs
    # ---------------------------------------------
    (
        snapshot_tab,
        gap_tab,
        path_tab,
        strength_tab,
        pass_tab,
    ) = st.tabs([
        "Overall Snapshot",
        "Gap or Zone",
        "RB Path",
        "Strength & Direction",
        "Pass Zones",
    ])

    # ---------------------------------------------
    # Tab 1: Overall Snapshot
    # ---------------------------------------------
    with snapshot_tab:
        st.subheader("Overall Snapshot")

        st.caption(
            "Includes all formations for the selected hash. "
            "The formation selector and four-run minimum "
            "do not limit this snapshot."
        )

        overall_strength = strength_percentages(overall_data)

        def percent_label(value):
            return (
                "N/A"
                if pd.isna(value)
                else f"{value:.1f}%"
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Plays",
            f"{len(overall_data):,}",
        )

        c2.metric(
            "At Strength",
            percent_label(
                overall_strength["At Strength %"]
            ),
        )

        c3.metric(
            "Away From Strength",
            percent_label(
                overall_strength["Away From Strength %"]
            ),
        )

        if (
            "gain_loss" in overall_data.columns
            and overall_data["gain_loss"].notna().any()
        ):
            gains = overall_data["gain_loss"]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Average Gain",
                f"{gains.mean():.1f} yds",
            )

            c2.metric(
                "Longest Gain",
                f"{gains.max():.1f} yds",
            )

            c3.metric(
                "Shortest Gain",
                f"{gains.min():.1f} yds",
            )

        left, right = st.columns(2)

        with left:
            display_distribution(
                overall_data,
                "GAP OR ZONE",
                "Gap or Zone",
            )

        with right:
            display_distribution(
                overall_data,
                "RB PATH",
                "RB Path",
            )

        left, right = st.columns(2)

        with left:
            display_distribution(
                overall_data,
                "OFF STR",
                "Offensive Strength",
            )

        with right:
            display_distribution(
                overall_data,
                "PLAY DIR",
                "Play Direction",
            )

        st.caption(
            "Each distribution uses its recorded values. "
            "At/Away From Strength uses plays where both "
            "strength and direction are L or R."
        )

    # ---------------------------------------------
    # Tabs 2 and 3: Gap / Zone and RB Path
    # ---------------------------------------------
    for tab, field, label in [
        (gap_tab, "GAP OR ZONE", "Gap or Zone"),
        (path_tab, "RB PATH", "RB Path"),
    ]:
        with tab:
            data = df.dropna(
                subset=["formation", field]
            ).copy()

            if data.empty:
                st.info("No recorded data for this selection.")
                continue

            data["concept"] = data["concept"].fillna(
                "(blank)"
            )

            formation_counts = pd.crosstab(
                data["formation"],
                data[field],
            )

            chart_data = (
                formation_counts
                .rename_axis(
                    index="Formation",
                    columns="Category",
                )
                .reset_index()
                .melt(
                    id_vars="Formation",
                    var_name="Category",
                    value_name="Plays",
                )
            )

            chart_data["Percentage"] = (
                chart_data["Plays"]
                / chart_data.groupby("Formation")[
                    "Plays"
                ].transform("sum")
                * 100
            )

            fig = px.bar(
                chart_data,
                x="Formation",
                y="Plays",
                color="Category",
                barmode="stack",
                title=f"{label} by Formation",
                custom_data=["Percentage"],
                color_discrete_sequence=[
                    "#7FDBFF",
                    "#448FC7",
                    "#B8CFDF",
                ],
            )

            fig.update_traces(
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Count: %{y}<br>"
                    "Percentage: %{customdata[0]:.1f}%"
                    "<extra>%{fullData.name}</extra>"
                )
            )

            show_chart(
                fig,
                key=f"simple_tendencies_{field}",
            )

            if selected_formation == "All":
                row_column = "formation"
                row_label = "Formation"
                st.subheader("Formation Breakdown")
            else:
                row_column = "concept"
                row_label = "Play Call"
                st.subheader(
                    f"Play Breakdown — {selected_formation}"
                )

            counts = pd.crosstab(
                data[row_column],
                data[field],
            )

            counts.columns.name = None

            percentages = counts.div(
                counts.sum(axis=1),
                axis=0,
            ) * 100

            summary = pd.DataFrame(index=counts.index)

            for category in counts.columns:
                summary[f"{category} Count"] = counts[category]
                summary[f"{category} %"] = percentages[
                    category
                ].round(1)

            summary["Total Plays"] = counts.sum(axis=1)

            if (
                selected_formation != "All"
                and "gain_loss" in data.columns
            ):
                gain_stats = (
                    data.groupby(row_column)
                    .agg(
                        recorded_gains=("gain_loss", "count"),
                        average_gain=("gain_loss", "mean"),
                        shortest_gain=("gain_loss", "min"),
                        longest_gain=("gain_loss", "max"),
                        total_yards=("gain_loss", "sum"),
                    )
                )

                gain_stats.loc[
                    gain_stats["recorded_gains"] == 0,
                    "total_yards",
                ] = float("nan")

                gain_stats = gain_stats.rename(columns={
                    "recorded_gains": "Recorded Gains",
                    "average_gain": "Average Gain",
                    "shortest_gain": "Shortest Gain",
                    "longest_gain": "Longest Gain",
                    "total_yards": "Total Yards",
                })

                summary = summary.join(gain_stats)

            summary = (
                summary
                .sort_values("Total Plays", ascending=False)
                .rename_axis(row_label)
                .reset_index()
            )

            column_config = percentage_config(
                summary.columns
            )

            column_config.update({
                "Total Plays": st.column_config.NumberColumn(
                    "Total Plays", format="%d"
                ),
                "Recorded Gains": st.column_config.NumberColumn(
                    "Recorded Gains", format="%d"
                ),
                "Average Gain": st.column_config.NumberColumn(
                    "Average Gain", format="%.1f yds"
                ),
                "Shortest Gain": st.column_config.NumberColumn(
                    "Shortest Gain", format="%.1f yds"
                ),
                "Longest Gain": st.column_config.NumberColumn(
                    "Longest Gain", format="%.1f yds"
                ),
                "Total Yards": st.column_config.NumberColumn(
                    "Total Yards", format="%.1f yds"
                ),
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )

            st.caption(
                f"Counts and percentages include plays with a "
                f"recorded {label} value. Gain statistics use "
                "those same plays and exclude missing gains."
            )

    # ---------------------------------------------
    # Tab 4: Strength & Direction
    # ---------------------------------------------
    with strength_tab:
        st.subheader("Strength & Play Direction")

        strength_data = df.dropna(
            subset=["formation"]
        ).copy()

        strength_data["concept"] = (
            strength_data["concept"].fillna("(blank)")
        )

        # Minimum applies only to this table.
        run_counts = (
            strength_data[
                strength_data["PLAY TYPE"]
                .eq("RUN")
                .fillna(False)
            ]
            .groupby("formation")
            .size()
        )

        eligible_formations = run_counts[
            run_counts > 3
        ].index

        strength_data = strength_data[
            strength_data["formation"].isin(
                eligible_formations
            )
        ].copy()

        if selected_formation == "All":
            group_column = "formation"
            group_label = "Formation"
        else:
            group_column = "concept"
            group_label = "Play Call"

        rows = []

        for name, plays in strength_data.groupby(
            group_column,
            sort=True,
        ):
            rows.append({
                group_label: name,
                **strength_percentages(plays),
            })

        if not rows:
            st.info(
                "No formations have at least four runs "
                "for this selection."
            )
        else:
            strength_summary = pd.DataFrame(rows)

            display_columns = [
                group_label,
                "Strength L %",
                "Strength R %",
                "Balanced %",
                "Direction L %",
                "Direction R %",
                "At Strength %",
                "Away From Strength %",
            ]

            st.dataframe(
                strength_summary[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config=percentage_config(
                    display_columns
                ),
            )

        st.caption(
            "Only formations with at least four runs are shown. "
            "Percentages still use all plays in those formations. "
            "At/Away From Strength excludes balanced strength "
            "and plays missing an L/R direction or strength."
        )

    # ---------------------------------------------
    # Tab 5: Pass Zones
    # ---------------------------------------------
    with pass_tab:
        st.subheader("Pass Zones")

        passes = df[
            df["PLAY TYPE"].eq("PASS").fillna(False)
        ].copy()

        valid_zone = passes["PASS ZONE"].isin(
            range(1, 10)
        )

        zone_data = passes[valid_zone].copy()

        zone_data["PASS ZONE"] = (
            zone_data["PASS ZONE"].astype(int)
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Total Pass Plays",
            len(passes),
        )

        c2.metric(
            "Passes With Recorded Zone",
            len(zone_data),
        )

        if zone_data.empty:
            st.info(
                "No pass plays with a recorded zone "
                "from 1 through 9 for this selection."
            )
        else:
            # ---------------------------------------------
            # Nine-zone percentage chart
            #
            # Top:       7  8  9
            # Middle:    4  5  6
            # Bottom:    1  2  3
            # ---------------------------------------------
            zone_counts = (
                zone_data["PASS ZONE"]
                .value_counts()
                .reindex(range(1, 10), fill_value=0)
            )

            zone_percentages = (
                zone_counts / zone_counts.sum() * 100
            )

            zone_layout = [
                [7, 8, 9],
                [4, 5, 6],
                [1, 2, 3],
            ]

            percentage_grid = [
                [
                    float(zone_percentages.loc[zone])
                    for zone in row
                ]
                for row in zone_layout
            ]

            hover_grid = [
                [
                    [
                        zone,
                        int(zone_counts.loc[zone]),
                    ]
                    for zone in row
                ]
                for row in zone_layout
            ]

            title = (
                "Pass Zone Percentages — All Formations"
                if selected_formation == "All"
                else (
                    "Pass Zone Percentages — "
                    f"{selected_formation}"
                )
            )

            fig = px.imshow(
                percentage_grid,
                x=["Left", "Middle", "Right"],
                y=["Top", "Middle", "Bottom"],
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="Blues",
                range_color=[0, 100],
                labels={
                    "x": "Horizontal Location",
                    "y": "Vertical Location",
                    "color": "Passes %",
                },
                title=title,
            )

            fig.update_traces(
                customdata=hover_grid,
                texttemplate=(
                    "Zone %{customdata[0]}<br>%{z:.1f}%"
                ),
                hovertemplate=(
                    "<b>Zone %{customdata[0]}</b><br>"
                    "Passes: %{customdata[1]}<br>"
                    "Percentage: %{z:.1f}%"
                    "<extra></extra>"
                ),
            )

            fig.update_layout(
                height=480,
                xaxis_title=None,
                yaxis_title=None,
            )

            show_chart(
                fig,
                key="formation_pass_zone_heatmap",
            )

            # ---------------------------------------------
            # Formation / play-call breakdown
            # ---------------------------------------------
            if selected_formation == "All":
                row_column = "formation"
                row_label = "Formation"
                st.subheader("Pass Zones by Formation")
            else:
                row_column = "concept"
                row_label = "Play Call"
                st.subheader(
                    f"Pass Zones by Play — {selected_formation}"
                )

            table_data = zone_data.dropna(
                subset=["formation"]
            ).copy()

            table_data["concept"] = (
                table_data["concept"].fillna("(blank)")
            )

            if table_data.empty:
                st.info(
                    "The recorded pass zones have "
                    "no formation labels."
                )
            else:
                counts = pd.crosstab(
                    table_data[row_column],
                    table_data["PASS ZONE"],
                ).reindex(
                    columns=range(1, 10),
                    fill_value=0,
                )

                counts.columns.name = None

                percentages = counts.div(
                    counts.sum(axis=1),
                    axis=0,
                ) * 100

                summary = pd.DataFrame(index=counts.index)

                summary["Zoned Passes"] = counts.sum(axis=1)

                for zone in range(1, 10):
                    summary[f"Zone {zone} Count"] = counts[zone]
                    summary[f"Zone {zone} %"] = percentages[zone]

                if (
                    selected_formation != "All"
                    and "gain_loss" in table_data.columns
                ):
                    gain_stats = (
                        table_data.groupby(row_column)
                        .agg(
                            average_gain=("gain_loss", "mean"),
                            shortest_gain=("gain_loss", "min"),
                            longest_gain=("gain_loss", "max"),
                        )
                        .rename(columns={
                            "average_gain": "Average Gain",
                            "shortest_gain": "Shortest Gain",
                            "longest_gain": "Longest Gain",
                        })
                    )

                    summary = summary.join(gain_stats)

                summary = (
                    summary
                    .sort_values("Zoned Passes", ascending=False)
                    .rename_axis(row_label)
                    .reset_index()
                )

                column_config = percentage_config(
                    summary.columns
                )

                column_config.update({
                    "Zoned Passes": st.column_config.NumberColumn(
                        "Zoned Passes", format="%d"
                    ),
                    "Average Gain": st.column_config.NumberColumn(
                        "Average Gain", format="%.1f yds"
                    ),
                    "Shortest Gain": st.column_config.NumberColumn(
                        "Shortest Gain", format="%.1f yds"
                    ),
                    "Longest Gain": st.column_config.NumberColumn(
                        "Longest Gain", format="%.1f yds"
                    ),
                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config,
                )

            st.caption(
                "Percentages use pass plays with a recorded "
                "zone from 1 through 9. Missing zones are excluded. "
                "Table percentages are calculated within each row. "
                "The four-run minimum does not apply to this tab."
            )


# =========================================================
# SHARED UPLOAD AND NAVIGATION
# =========================================================
def clear_shared_upload():
    st.session_state.pop("shared_upload_bytes", None)
    st.session_state.pop("shared_upload_name", None)
    st.session_state.pop("shared_hudl_uploader", None)


page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Basic Dashboard",
        "Advanced Analytics",
        "Play Breakdown",
        "Formation Tendencies",
    ],
    key="active_page",
)

st.sidebar.divider()
st.sidebar.subheader("Shared Hudl File")

new_upload = st.sidebar.file_uploader(
    "Upload Hudl Excel File",
    type=["xlsx", "xls"],
    key="shared_hudl_uploader",
)

if new_upload is not None:
    st.session_state.shared_upload_bytes = (
        new_upload.getvalue()
    )
    st.session_state.shared_upload_name = new_upload.name

uploaded_file = None

if st.session_state.get("shared_upload_bytes"):
    uploaded_file = BytesIO(
        st.session_state.shared_upload_bytes
    )
    uploaded_file.name = st.session_state.get(
        "shared_upload_name", "hudl_data.xlsx"
    )

    st.sidebar.success(f"Using: {uploaded_file.name}")

    st.sidebar.button(
        "Clear uploaded file",
        on_click=clear_shared_upload,
        use_container_width=True,
    )


# =========================================================
# GLOBAL GROUPING SETTINGS
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Dashboard Settings")

st.sidebar.toggle(
    "Group RPO-tagged plays",
    value=True,
    key="ignore_rpo_tag",
    help=(
        "Remove the (RPO) tag from every play when grouping "
        "data. For example, Power and Power (RPO) will "
        "be counted together."
    ),
)

st.sidebar.toggle(
    "Group plays with leading LT",
    value=True,
    key="group_lt_plays_v2",
    help=(
        "Remove LT when it appears at the beginning of a "
        "play name. For example, LT Troff and Troff "
        "will be counted together."
    ),
)

st.sidebar.toggle(
    "Group Read-tagged plays",
    value=True,
    key="group_read_plays_v2",
    help=(
        "Remove the standalone word Read from play names. "
        "For example, Counter Read and Counter "
        "will be counted together."
    ),
)

st.sidebar.toggle(
    "Group numbered play variations",
    value=True,
    key="group_numbered_plays_v2",
    help=(
        "Remove numbers from play names. For example, "
        "Troff 1 and Troff will be counted together."
    ),
)


# =========================================================
# PAGE ROUTING
# =========================================================
if page == "Basic Dashboard":
    render_basic_dashboard(uploaded_file)

elif page == "Advanced Analytics":
    render_advanced_analytics(uploaded_file)

elif page == "Play Breakdown":
    render_play_breakdown(uploaded_file)

elif page == "Formation Tendencies":
    render_formation_tendencies(uploaded_file)

else:
    render_home()
