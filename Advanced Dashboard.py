import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har-Ber Advanced Analytics", layout="wide")

# -------------------------
# Custom CSS (same style)
# -------------------------
st.markdown("""
<style>
body, .main { background-color: #0d0d0d !important; color: #FFFFFF; }

h1, h2, h3 { color: #7FDBFF; }

.section-header {
    background-color: #0A2342;
    padding: 10px;
    border-radius: 5px;
    color: white;
    font-weight: bold;
    margin-top: 10px;
}

.metric-card {
    background-color: #0A2342;
    padding: 10px 15px;
    border-radius: 8px;
    text-align: center;
    color: white;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.4);
    margin-bottom: 8px;
}

.metric-number {
    font-size: 22px;
    font-weight: bold;
    color: #7FDBFF;
}

.metric-label {
    font-size: 12px;
    color: #AAAAAA;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Logo
# -------------------------
st.sidebar.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
st.sidebar.image("logo_har-ber-high-school.png", width=150)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Sidebar Upload
# -------------------------
st.sidebar.title("Har-Ber Advanced Dashboard")

team_file = st.sidebar.file_uploader(
    "Upload Har-Ber Hudl Data",
    type=["xlsx","xls"]
)

opponent_file = st.sidebar.file_uploader(
    "Upload Opponent Hudl Data",
    type=["xlsx","xls"]
)

# -------------------------
# Load Data
# -------------------------
def load_data(file):
    df = pd.read_excel(file)
    df.columns = df.columns.str.lower().str.strip()
    return df

if team_file:
    df = load_data(team_file)

    COLUMN_MAP = {
        "down": ["down","dn"],
        "distance":["dist","togo","yards to go","ydstogo"],
        "yardline":["yard ln","spot","ball on"],
        "concept":["off play"],
        "play_type":["play type","playtype","type"],
        "play_direction":["play dir"],
        "gain_loss":["gn/ls"]
    }

    rename_dict = {}

    for s,v in COLUMN_MAP.items():
        for col in df.columns:
            if col in v:
                rename_dict[col] = s

    df = df.rename(columns=rename_dict)

    # -------------------------
    # Success Variable
    # -------------------------
    df["success"] = df["gain_loss"] >= 4

    # -------------------------
    # Tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Play Success Prediction",
    "Explosive Play Analysis",
    "Opponent Comparison",
    "Best Play Call",
    "Play Call Advisor",
    "Defensive Tendencies"
])

# ============================================================
# TAB 1 — PLAY SUCCESS PREDICTION
# ============================================================

    with tab1:

        st.markdown(
        '<div class="section-header">Predict Play Success</div>',
        unsafe_allow_html=True
        )

        st.write(
        "Predict probability of success (4+ yards) based on game situation."
        )

        # Model features
        model_df = df.dropna(subset=["down","distance","yardline","play_type"])

        X = model_df[["down","distance","yardline"]]
        y = model_df["success"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        # User Inputs
        c1,c2,c3 = st.columns(3)

        with c1:
            down_input = st.selectbox(
                "Down",
                sorted(df["down"].dropna().unique())
            )

        with c2:
            dist_input = st.slider(
                "Distance to Go",
                1,20,5
            )

        with c3:
            yard_input = st.slider(
                "Yardline",
                -50,50,0
            )

        pred_df = pd.DataFrame({
            "down":[down_input],
            "distance":[dist_input],
            "yardline":[yard_input]
        })

        prob = model.predict_proba(pred_df)[0][1]

        st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{round(prob*100,1)}%</div>
        <div class="metric-label">Predicted Success Probability</div>
        </div>
        """,
        unsafe_allow_html=True
        )

# ============================================================
# TAB 2 — EXPLOSIVE PLAY ANALYSIS
# ============================================================

    with tab2:

        st.markdown(
        '<div class="section-header">Explosive Play Analysis</div>',
        unsafe_allow_html=True
        )

        df["explosive"] = (
            ((df["play_type"]=="Run") & (df["gain_loss"]>=10)) |
            ((df["play_type"]=="Pass") & (df["gain_loss"]>=20))
        )

        explosive_rate = df["explosive"].mean()*100

        st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-number">{round(explosive_rate,1)}%</div>
        <div class="metric-label">Explosive Play Rate</div>
        </div>
        """,
        unsafe_allow_html=True
        )

        # Explosive by concept
        exp_concept = df.groupby("concept")["explosive"].mean().reset_index()

        fig1 = px.bar(
            exp_concept,
            x="explosive",
            y="concept",
            orientation="h",
            title="Explosive Rate by Concept",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )

        st.plotly_chart(fig1, use_container_width=True)

        # Explosive by down
        exp_down = df.groupby("down")["explosive"].mean().reset_index()

        fig2 = px.bar(
            exp_down,
            x="down",
            y="explosive",
            title="Explosive Rate by Down",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )

        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# TAB 3 — OPPONENT COMPARISON
# ============================================================

    with tab3:

        if opponent_file:

            opp = load_data(opponent_file)
            opp = opp.rename(columns=rename_dict)

            st.markdown(
            '<div class="section-header">Concept Usage Comparison</div>',
            unsafe_allow_html=True
            )

            team_concepts = df["concept"].value_counts(normalize=True)
            opp_concepts = opp["concept"].value_counts(normalize=True)

            compare = pd.concat(
                [team_concepts, opp_concepts],
                axis=1
            )

            compare.columns = ["HarBer","Opponent"]
            compare = compare.fillna(0).reset_index()
            compare = compare.rename(columns={"index":"concept"})

            fig = px.bar(
                compare,
                x="concept",
                y=["HarBer","Opponent"],
                barmode="group",
                title="Concept Usage Comparison",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Upload opponent data to enable comparison.")



# ============================================================
# TAB 4 — BEST PLAY CALL
# ============================================================

with tab4:

    st.markdown(
    '<div class="section-header">Best Play Call Recommendation</div>',
    unsafe_allow_html=True
    )

    st.write(
    "Find the most effective play concepts for a given situation based on historical success."
    )

    # User Inputs
    c1, c2, c3 = st.columns(3)

    with c1:
        down_choice = st.selectbox(
            "Down",
            sorted(df["down"].dropna().unique())
        )

    with c2:
        distance_choice = st.slider(
            "Distance to Go",
            1,20,5
        )

    with c3:
        yard_choice = st.slider(
            "Yardline",
            -50,50,0
        )

    # Situation Filter
    situation = df[
        (df["down"] == down_choice) &
        (df["distance"].between(distance_choice-2, distance_choice+2)) &
        (df["yardline"].between(yard_choice-10, yard_choice+10))
    ]

    if situation.empty:

        st.warning("Not enough plays in dataset for this situation.")

    else:

        concept_success = (
            situation
            .groupby("concept")["success"]
            .mean()
            .reset_index()
            .sort_values("success", ascending=False)
            .head(5)
        )

        concept_success["success_pct"] = concept_success["success"] * 100

        # Chart
        fig = px.bar(
            concept_success,
            x="success_pct",
            y="concept",
            orientation="h",
            title="Top 5 Recommended Concepts",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.markdown(
        '<div class="section-header">Concept Success Rates</div>',
        unsafe_allow_html=True
        )

        st.dataframe(concept_success)

  # ============================================================
# TAB 5 — PLAY CALL ADVISOR
# ============================================================

with tab5:

    st.markdown(
    '<div class="section-header">AI Play Call Advisor</div>',
    unsafe_allow_html=True
    )

    st.write(
    "Recommends the best concept based on success rate, expected yards, and explosive probability."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        down_choice = st.selectbox(
            "Down",
            sorted(df["down"].dropna().unique()),
            key="advisor_down"
        )

    with c2:
        dist_choice = st.slider(
            "Distance",
            1,20,5,
            key="advisor_dist"
        )

    with c3:
        yard_choice = st.slider(
            "Yardline",
            -50,50,0,
            key="advisor_yard"
        )

    situation = df[
        (df["down"] == down_choice) &
        (df["distance"].between(dist_choice-2, dist_choice+2)) &
        (df["yardline"].between(yard_choice-10, yard_choice+10))
    ]

    if situation.empty:

        st.warning("Not enough data for this situation.")

    else:

        advisor = (
            situation
            .groupby("concept")
            .agg(
                success_rate=("success","mean"),
                avg_gain=("gain_loss","mean"),
                explosive_rate=("explosive","mean")
            )
            .reset_index()
        )

        advisor["score"] = (
            advisor["success_rate"]*0.5 +
            advisor["avg_gain"]/10*0.3 +
            advisor["explosive_rate"]*0.2
        )

        advisor = advisor.sort_values("score", ascending=False)

        best = advisor.iloc[0]

        # Metric cards
        c1,c2,c3,c4 = st.columns(4)

        c1.markdown(f"""
        <div class="metric-card">
        <div class="metric-number">{best['concept']}</div>
        <div class="metric-label">Recommended Concept</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
        <div class="metric-number">{round(best['avg_gain'],1)}</div>
        <div class="metric-label">Expected Yards</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card">
        <div class="metric-number">{round(best['success_rate']*100,1)}%</div>
        <div class="metric-label">Success Probability</div>
        </div>
        """, unsafe_allow_html=True)

        c4.markdown(f"""
        <div class="metric-card">
        <div class="metric-number">{round(best['explosive_rate']*100,1)}%</div>
        <div class="metric-label">Explosive Probability</div>
        </div>
        """, unsafe_allow_html=True)

        # Chart of top plays
        fig = px.bar(
            advisor.head(8),
            x="score",
            y="concept",
            orientation="h",
            title="Top Recommended Plays",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 6 — DEFENSIVE TENDENCIES
# ============================================================

with tab6:

    st.markdown(
    '<div class="section-header">Defensive Tendencies Heatmap</div>',
    unsafe_allow_html=True
    )

    st.write(
    "Shows the most common concepts offenses run in different down/distance situations."
    )

    # Distance buckets
    df["dist_bucket"] = pd.cut(
        df["distance"],
        bins=[0,3,6,10,20],
        labels=["Short (1-3)","Medium (4-6)","Long (7-10)","Very Long (11+)"]
    )

    tendency = (
        df.groupby(["down","dist_bucket","concept"])
        .size()
        .reset_index(name="plays")
    )

    top = (
        tendency
        .sort_values("plays", ascending=False)
        .groupby(["down","dist_bucket"])
        .first()
        .reset_index()
    )

    pivot = top.pivot(
        index="down",
        columns="dist_bucket",
        values="concept"
    )

    fig = px.imshow(
        pivot.astype(str),
        text_auto=True,
        aspect="auto",
        title="Most Common Play Concept by Situation",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
    "Each cell shows the most common concept run in that situation. "
    "Use this to anticipate opponent tendencies."
    )
