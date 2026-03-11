import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har‑Ber Football Analytics", layout="wide")

# -------------------------
# Custom CSS (Dark Mode)
# -------------------------
st.markdown("""
<style>
body, .main { background-color: #0d0d0d !important; color: #FFFFFF; }
.css-1d391kg { background-color: #111111 !important; }
h1, h2, h3 { color: #7FDBFF; }
.section-header { background-color: #0A2342; padding: 10px; border-radius: 5px; color: #FFFFFF; font-weight: bold; }
.metric-card { background-color: #0A2342; padding: 20px; border-radius: 10px; text-align:center; color:white; box-shadow:0px 4px 12px rgba(0,0,0,0.4); transition: transform 0.2s; }
.metric-card:hover { transform: scale(1.05); }
.metric-number { font-size:30px; font-weight:bold; color:#7FDBFF; }
.metric-label { font-size:14px; color:#AAAAAA; }
.css-1lcbmhc.e1fqkh3o3 { background-color:#1A1A1A !important; color:#FFFFFF !important; }
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
        else:
            return "Other"
    df["yard_group"] = df["yardline"].apply(custom_yard_group)

    # -------------------------
    # Custom Yard Order
    # -------------------------
    yard_order = [
        "0 - -9", "-10 - -19", "-20 - -29", "-30 - -39", "-40 - -49",
        "+50 - +40", "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0"
    ]

    # -------------------------
    # Sidebar Filters
    # -------------------------
    st.sidebar.header("Filters")
    down_choices = sorted(df["down"].dropna().unique())
    down_selected = st.sidebar.selectbox("Down", down_choices)

    # Filter yard groups based on selected down
    df_down = df[df["down"] == down_selected]
    yard_choices = [yg for yg in yard_order if yg in df_down["yard_group"].unique()]
    yard_choice = st.sidebar.selectbox("Yard Group", yard_choices)

    # Selected plays for Tab 1
    selected = df_down[df_down["yard_group"] == yard_choice]
    if selected.empty:
        st.warning("No plays for this selection.")
        st.stop()

    # -------------------------
    # Tabs
    # -------------------------
    tab1, tab2 = st.tabs(["Filtered by Down/Yardline", "Entire Dataset"])

    # -------------------------
    # Tab 1
    # -------------------------
    with tab1:
        # Metrics
        avg_gain = round(selected["gain_loss"].mean(),1)
        max_gain = selected["gain_loss"].max()
        min_gain = selected["gain_loss"].min()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-number">{avg_gain}</div><div class="metric-label">Average Gain</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-number">{max_gain}</div><div class="metric-label">Max Gain</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-number">{min_gain}</div><div class="metric-label">Min Gain</div></div>', unsafe_allow_html=True)

        # Gain/Loss Distribution
        gain_summary = selected.groupby("gain_loss").size().reset_index(name="plays").sort_values("gain_loss")
        gain_fig = px.bar(gain_summary, x="gain_loss", y="plays", labels={"gain_loss":"Yards Gained","plays":"Number of Plays"}, title="Gain / Loss Distribution", template="plotly_dark", color_discrete_sequence=["#7FDBFF"])

        # Top 8 Concepts
        top_concepts = selected.groupby(["concept","play_direction"]).size().reset_index(name="count").sort_values("count",ascending=False).head(8)
        concept_fig = px.bar(top_concepts, x="count", y="concept", color="play_direction", orientation="h", title="Top 6 Concepts by Play Direction", template="plotly_dark", color_discrete_sequence=["#7FDBFF","#0A2342","#AAAAAA"])

        # Run/Pass Pie
        play_type_summary = selected["play_type"].value_counts().reset_index()
        play_type_summary.columns = ["play_type","count"]
        run_pass_fig = px.pie(play_type_summary, names="play_type", values="count", title="Run vs Pass %", color="play_type", color_discrete_map={"Run":"#0A2342","Pass":"#7FDBFF"}, template="plotly_dark")

        # Concept Pie (Top 6)
        concept_summary = selected["concept"].value_counts().head(6).reset_index()
        concept_summary.columns = ["concept","count"]
        concept_pie_fig = px.pie(concept_summary, names="concept", values="count", title="Top 6 Concepts", color_discrete_sequence=px.colors.sequential.Blues, template="plotly_dark")

        # Layout Charts
        st.markdown('<div class="section-header">Gain / Loss & Top Concepts</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig, use_container_width=True)
        r1c2.plotly_chart(concept_fig, use_container_width=True)

        st.markdown('<div class="section-header">Run/Pass & Concept Distribution</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig, use_container_width=True)
        r2c2.plotly_chart(concept_pie_fig, use_container_width=True)

        st.markdown('<div class="section-header">Raw Play Data</div>', unsafe_allow_html=True)
        st.dataframe(selected, use_container_width=True)

    # -------------------------
    # Tab 2
    # -------------------------
    with tab2:
        avg_gain_all = round(df["gain_loss"].mean(),1)
        max_gain_all = df["gain_loss"].max()
        min_gain_all = df["gain_loss"].min()
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-number">{avg_gain_all}</div><div class="metric-label">Average Gain</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-number">{max_gain_all}</div><div class="metric-label">Max Gain</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-number">{min_gain_all}</div><div class="metric-label">Min Gain</div></div>', unsafe_allow_html=True)

        gain_summary_all = df.groupby("gain_loss").size().reset_index(name="plays").sort_values("gain_loss")
        gain_fig_all = px.bar(gain_summary_all, x="gain_loss", y="plays", labels={"gain_loss":"Yards Gained","plays":"Number of Plays"}, title="Gain / Loss Distribution", template="plotly_dark", color_discrete_sequence=["#7FDBFF"])

        top_concepts_all = df.groupby(["concept","play_direction"]).size().reset_index(name="count").sort_values("count",ascending=False).head(8)
        concept_fig_all = px.bar(top_concepts_all, x="count", y="concept", color="play_direction", orientation="h", title="Top 6 Concepts by Play Direction", template="plotly_dark", color_discrete_sequence=["#7FDBFF","#0A2342","#AAAAAA"])

        play_type_summary_all = df["play_type"].value_counts().reset_index()
        play_type_summary_all.columns = ["play_type","count"]
        run_pass_fig_all = px.pie(play_type_summary_all, names="play_type", values="count", title="Run vs Pass %", color="play_type", color_discrete_map={"Run":"#0A2342","Pass":"#7FDBFF"}, template="plotly_dark")

        concept_summary_all = df["concept"].value_counts().head(6).reset_index()
        concept_summary_all.columns = ["concept","count"]
        concept_pie_fig_all = px.pie(concept_summary_all, names="concept", values="count", title="Top 6 Concepts", color_discrete_sequence=px.colors.sequential.Blues, template="plotly_dark")

        st.markdown('<div class="section-header">Gain / Loss & Top Concepts</div>', unsafe_allow_html=True)
        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig_all, use_container_width=True)
        r1c2.plotly_chart(concept_fig_all, use_container_width=True)

        st.markdown('<div class="section-header">Run/Pass & Concept Distribution</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig_all, use_container_width=True)
        r2c2.plotly_chart(concept_pie_fig_all, use_container_width=True)