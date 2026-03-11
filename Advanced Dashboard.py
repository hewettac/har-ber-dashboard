import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har‑Ber Football Analytics", layout="wide")

# -------------------------
# Custom CSS (Dark Mode + Cards)
# -------------------------
st.markdown("""
<style>
body, .main { background-color: #0d0d0d !important; color: #FFFFFF; }
.css-1d391kg { background-color: #111111 !important; }
h1, h2, h3 { color: #7FDBFF; }
.section-header { background-color: #0A2342; padding: 10px; border-radius: 5px; color: #FFFFFF; font-weight: bold; margin-top:20px; }
.metric-card { background-color: #1A1A1A; padding: 15px; border-radius: 10px; text-align:center; color:white; box-shadow:0px 4px 12px rgba(0,0,0,0.4); transition: transform 0.2s; }
.metric-card:hover { transform: scale(1.05); }
.metric-number { font-size:24px; font-weight:bold; color:#7FDBFF; }
.metric-label { font-size:12px; color:#AAAAAA; }
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
st.sidebar.title("Har‑Ber Football Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Hudl Excel File", type=["xlsx","xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    # -------------------------
    # Column Renaming
    # -------------------------
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
    # Custom Yard Groups
    # -------------------------
    def custom_yard_group(yardline):
        if pd.isna(yardline):
            return "Unknown"
        if 0 >= yardline >= -9: return "0 - -9"
        if -10 >= yardline >= -19: return "-10 - -19"
        if -20 >= yardline >= -29: return "-20 - -29"
        if -30 >= yardline >= -39: return "-30 - -39"
        if -40 >= yardline >= -50: return "-40 - -50"
        if 50 >= yardline >= 40: return "+50 - +40"
        if 39 >= yardline >= 30: return "+39 - +30"
        if 29 >= yardline >= 20: return "+29 - +20"
        if 19 >= yardline >= 10: return "+19 - +10"
        if 9 >= yardline >= 0: return "+9 - 0"
        return "Other"

    df["yard_group"] = df["yardline"].apply(custom_yard_group)
    yard_order = [
        "0 - -9", "-10 - -19", "-20 - -29", "-30 - -39", "-40 - -50",
        "+50 - +40", "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0"
    ]

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs([
        "Play Success Prediction",
        "Explosive & Success Metrics",
        "Opponent Comparison",
        "Best Play Call",
        "Play Call Advisor",
        "Defensive Tendencies",
        "Opponent Play Predictor",
        "Play Call Win Probability"
    ])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs

    # -------------------------
    # TAB 1: Play Success Prediction
    # -------------------------
    with tab1:
        st.markdown('<div class="section-header">Play Success Prediction</div>', unsafe_allow_html=True)
        down_choice = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="success_down")
        df_down = df[df["down"] == down_choice]
        yard_choice = st.selectbox("Yard Group", [yg for yg in yard_order if yg in df_down["yard_group"].unique()], key="success_yard")
        selected = df_down[df_down["yard_group"] == yard_choice]

        if selected.empty:
            st.warning("No plays for this selection.")
            st.stop()

        avg_gain = round(selected["gain_loss"].mean(),1)
        max_gain = selected["gain_loss"].max()
        min_gain = selected["gain_loss"].min()
        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-number">{avg_gain}</div><div class="metric-label">Average Gain</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-number">{max_gain}</div><div class="metric-label">Max Gain</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-number">{min_gain}</div><div class="metric-label">Min Gain</div></div>', unsafe_allow_html=True)

        gain_summary = selected.groupby("gain_loss").size().reset_index(name="plays").sort_values("gain_loss")
        gain_fig = px.bar(
            gain_summary,
            x="gain_loss", y="plays",
            labels={"gain_loss":"Yards Gained","plays":"Number of Plays"},
            title="Gain / Loss Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#7FDBFF"]
        )
        st.plotly_chart(gain_fig, use_container_width=True)

    # -------------------------
    # TAB 2: Explosive & Success Metrics
    # -------------------------
    with tab2:
        st.markdown('<div class="section-header">Explosive Play & Success Metrics</div>', unsafe_allow_html=True)
        st.markdown("""
        **Definitions:**  
        - **Explosive Plays:** Runs ≥ 10 yards, Passes ≥ 20 yards  
        - **Success:** Plays gaining ≥ 4 yards
        """)

        # Compute explosive and success flags
        df['explosive'] = df.apply(lambda row: row['gain_loss'] >= 10 if row['play_type']=='Run' else row['gain_loss'] >= 20, axis=1)
        df['success'] = df['gain_loss'] >= 4

        # Metrics
        total_plays = len(df)
        explosive_runs_pct = df[df['play_type']=='Run']['explosive'].mean()*100
        explosive_pass_pct = df[df['play_type']=='Pass']['explosive'].mean()*100
        success_pct = df['success'].mean()*100

        # Metric Cards
        m1,m2,m3,m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-number">{total_plays}</div><div class="metric-label">Total Plays</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-number">{explosive_runs_pct:.1f}%</div><div class="metric-label">Explosive Runs</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-number">{explosive_pass_pct:.1f}%</div><div class="metric-label">Explosive Passes</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-number">{success_pct:.1f}%</div><div class="metric-label">Overall Success %</div></div>', unsafe_allow_html=True)

        # Aggregate for heatmaps
        run_df = df[df['play_type']=='Run'].groupby(['down','yard_group'])['explosive'].mean().reset_index()
        run_df['explosive'] *= 100
        run_df['explosive'] = pd.to_numeric(run_df['explosive'], errors='coerce').fillna(0)

        pass_df = df[df['play_type']=='Pass'].groupby(['down','yard_group'])['explosive'].mean().reset_index()
        pass_df['explosive'] *= 100
        pass_df['explosive'] = pd.to_numeric(pass_df['explosive'], errors='coerce').fillna(0)

        success_df = df.groupby(['down','yard_group'])['success'].mean().reset_index()
        success_df['success'] *= 100
        success_df['success'] = pd.to_numeric(success_df['success'], errors='coerce').fillna(0)

        # Ensure down is categorical for heatmap
        for df_heat in [run_df, pass_df, success_df]:
            df_heat['down'] = pd.Categorical(df_heat['down'], categories=sorted(df['down'].dropna().unique()))

        # Heatmap function
        def plot_heatmap(df_heat, val_col, title):
            fig = px.density_heatmap(
                df_heat,
                x='yard_group',
                y='down',
                z=val_col,
                text=df_heat[val_col].round(1).astype(str)+'%',
                color_continuous_scale='Blues',
                labels={val_col:title,'yard_group':'Yard Group','down':'Down'},
                template='plotly_dark',
                title=title
            )
            fig.update_traces(texttemplate="%{text}", textfont_size=14)
            return fig

        st.markdown("### Explosive Plays")
        c1,c2 = st.columns(2)
        c1.plotly_chart(plot_heatmap(run_df,'explosive','Run Explosive Plays %'), use_container_width=True)
        c2.plotly_chart(plot_heatmap(pass_df,'explosive','Pass Explosive Plays %'), use_container_width=True)

        st.markdown("### Success Rate")
        st.plotly_chart(plot_heatmap(success_df,'success','Success Rate %'), use_container_width=True)

    # -------------------------
    # TAB 5: Play Call Advisor
    # -------------------------
    with tab5:
        st.markdown('<div class="section-header">Play Call Advisor</div>', unsafe_allow_html=True)
        down_input = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="advisor_down")
        dist_input = st.slider("Distance", 1, 20, 5, key="advisor_distance")
        yard_input = st.slider("Yardline", -50, 50, 0, key="advisor_yardline")
        advisor_df = df[(df["down"]==down_input)&(df["distance"]==dist_input)&(df["yardline"]==yard_input)]
        top_concepts = advisor_df["concept"].value_counts().head(5).reset_index()
        top_concepts.columns = ["concept","count"]
        fig = px.bar(top_concepts, x="count", y="concept", orientation="h", color_discrete_sequence=["#7FDBFF"], template="plotly_dark", title="Top Concepts for This Situation")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 7: Opponent Play Predictor
    # -------------------------
    with tab7:
        st.markdown('<div class="section-header">Opponent Play Predictor</div>', unsafe_allow_html=True)
        model_df = df.dropna(subset=["down","distance","yardline","concept","play_type"])
        X = model_df[["down","distance","yardline"]]
        y_concept = model_df["concept"]
        y_type = model_df["play_type"]

        concept_model = RandomForestClassifier(n_estimators=200, random_state=42)
        type_model = RandomForestClassifier(n_estimators=200, random_state=42)
        concept_model.fit(X, y_concept)
        type_model.fit(X, y_type)

        c1,c2,c3 = st.columns(3)
        down_input = c1.selectbox("Down", sorted(df["down"].dropna().unique()), key="predictor_down")
        dist_input = c2.slider("Distance", 1,20,5, key="predictor_distance")
        yard_input = c3.slider("Yardline", -50,50,0, key="predictor_yardline")

        pred_df = pd.DataFrame({"down":[down_input], "distance":[dist_input], "yardline":[yard_input]})
        concept_probs = concept_model.predict_proba(pred_df)[0]
        concept_names = concept_model.classes_
        type_probs = type_model.predict_proba(pred_df)[0]
        type_names = type_model.classes_

        type_results = dict(zip(type_names, type_probs))
        run_prob = type_results.get("Run",0)*100
        pass_prob = type_results.get("Pass",0)*100
        concept_df = pd.DataFrame({"concept":concept_names,"prob":concept_probs}).sort_values("prob", ascending=False)
        top3 = concept_df.head(3)

        c1,c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-number">{round(run_prob,1)}%</div><div class="metric-label">Run Probability</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-number">{round(pass_prob,1)}%</div><div class="metric-label">Pass Probability</div></div>', unsafe_allow_html=True)

        fig = px.bar(top3, x="prob", y="concept", orientation="h", color_discrete_sequence=["#7FDBFF"], template="plotly_dark", title="Top 3 Predicted Plays")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # TAB 8: Play Call Win Probability
    # -------------------------
    with tab8:
        st.markdown('<div class="section-header">Play Call Win Probability</div>', unsafe_allow_html=True)
        st.info("""
        **How to read this tab:**
        - **Expected Gain:** Average yards gained historically for this play/concept.
        - **Success %:** Percentage of plays gaining 4+ yards (or more than distance to go).
        - **Explosive %:** Percentage of plays gaining 20+ yards.
        - **Best Concept:** The play concept with highest expected gain × success probability.
        """)

        down_input = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="winprob_down")
        dist_input = st.slider("Distance to Go", 1, 20, 5, key="winprob_distance")
        yard_input = st.slider("Yardline", -50, 50, 0, key="winprob_yardline")
        play_type_input = st.multiselect(
            "Play Type / Concept to Evaluate",
            df["concept"].unique(),
            default=df["concept"].unique()[:5],
            key="winprob_concepts"
        )

        hist_df = df[
            (df["down"] == down_input) &
            (df["distance"] == dist_input) &
            (df["yardline"] == yard_input) &
            (df["concept"].isin(play_type_input))
        ]

        if hist_df.empty:
            st.warning("No historical plays found for this combination. Adjust inputs.")
            st.stop()

        hist_df["success"] = hist_df["gain_loss"] >= max(4, dist_input)
        hist_df["explosive"] = hist_df["gain_loss"] >= 20

        summary = hist_df.groupby("concept").agg(
            expected_gain=("gain_loss","mean"),
            success_pct=("success","mean"),
            explosive_pct=("explosive","mean")
        ).reset_index()

        summary["rank_score"] = summary["expected_gain"] * summary["success_pct"]
        best_play = summary.sort_values("rank_score", ascending=False).iloc[0]

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-number">{round(best_play.expected_gain,1)}</div><div class="metric-label">Expected Gain</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-number">{round(best_play.success_pct*100,1)}%</div><div class="metric-label">Success %</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-number">{round(best_play.explosive_pct*100,1)}%</div><div class="metric-label">Explosive %</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-number">{best_play.concept}</div><div class="metric-label">Best Concept</div></div>', unsafe_allow_html=True)

        fig_summary = px.bar(
            summary.sort_values("expected_gain"),
            x="expected_gain",
            y="concept",
            orientation="h",
            color="success_pct",
            color_continuous_scale="Blues",
            text=summary["success_pct"].apply(lambda x: f"{x*100:.1f}%"),
            labels={"expected_gain":"Expected Gain (yards)","concept":"Play Concept","success_pct":"Success %"},
            title="Play Comparison: Expected Gain vs Success %",
            template="plotly_dark"
        )
        fig_summary.update_layout(yaxis=dict(dtick=1))
        st.plotly_chart(fig_summary, use_container_width=True)

        summary_display = summary.copy()
        summary_display["expected_gain"] = summary_display["expected_gain"].round(1)
        summary_display["success_pct"] = (summary_display["success_pct"]*100).round(1).astype(str)+"%"
        summary_display["explosive_pct"] = (summary_display["explosive_pct"]*100).round(1).astype(str)+"%"
        summary_display = summary_display.sort_values("rank_score", ascending=False)
        st.markdown('<div class="section-header">Detailed Play Stats</div>', unsafe_allow_html=True)
        st.dataframe(summary_display[["concept","expected_gain","success_pct","explosive_pct"]], use_container_width=True)
