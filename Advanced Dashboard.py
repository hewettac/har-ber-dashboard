import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har‑Ber Football Analytics", layout="wide")

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
body, .main { background-color: #0d0d0d !important; color: #FFFFFF; }
h1,h2,h3 { color: #7FDBFF; }
.section-header { background-color: #0A2342; padding: 10px; border-radius: 5px; color: #FFFFFF; font-weight: bold; margin-top:20px; }
.metric-card { background-color: #1A1A1A; padding: 15px; border-radius: 10px; text-align:center; color:white; box-shadow:0px 4px 12px rgba(0,0,0,0.4); transition: transform 0.2s; }
.metric-card:hover { transform: scale(1.05); }
.metric-number { font-size:24px; font-weight:bold; color:#7FDBFF; }
.metric-label { font-size:12px; color:#AAAAAA; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar Logo & Upload
# -------------------------
st.sidebar.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
st.sidebar.image("logo_har-ber-high-school.png", width=150)
st.sidebar.markdown('</div>', unsafe_allow_html=True)
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
    # Yard Groups
    # -------------------------
    def custom_yard_group(y):
        if pd.isna(y): return "Unknown"
        if 0 >= y >= -9: return "0 - -9"
        if -10 >= y >= -19: return "-10 - -19"
        if -20 >= y >= -29: return "-20 - -29"
        if -30 >= y >= -39: return "-30 - -39"
        if -40 >= y >= -50: return "-40 - -50"
        if 50 >= y >= 40: return "+50 - +40"
        if 39 >= y >= 30: return "+39 - +30"
        if 29 >= y >= 20: return "+29 - +20"
        if 19 >= y >= 10: return "+19 - +10"
        if 9 >= y >= 0: return "+9 - 0"
        return "Other"
    df["yard_group"] = df["yardline"].apply(custom_yard_group)

    yard_order = ["0 - -9", "-10 - -19", "-20 - -29", "-30 - -39", "-40 - -50",
                  "+50 - +40", "+39 - +30", "+29 - +20", "+19 - +10", "+9 - 0"]

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
        gain_fig = px.bar(gain_summary, x="gain_loss", y="plays",
                          labels={"gain_loss":"Yards Gained","plays":"Number of Plays"},
                          title="Gain / Loss Distribution",
                          template="plotly_dark",
                          color_discrete_sequence=["#7FDBFF"])
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
        df['explosive'] = df.apply(lambda row: row['gain_loss'] >= 10 if row['play_type']=='Run' else row['gain_loss'] >= 20, axis=1)
        df['success'] = df['gain_loss'] >= 4

        total_plays = len(df)
        explosive_runs_pct = df[df['play_type']=='Run']['explosive'].mean()*100
        explosive_pass_pct = df[df['play_type']=='Pass']['explosive'].mean()*100
        success_pct = df['success'].mean()*100

        m1,m2,m3,m4 = st.columns(4)
        for metric,val,label in zip([m1,m2,m3,m4],
                                    [total_plays,explosive_runs_pct,explosive_pass_pct,success_pct],
                                    ["Total Plays","Explosive Runs","Explosive Passes","Overall Success %"]):
            metric.markdown(f'<div class="metric-card"><div class="metric-number">{round(val,1) if isinstance(val,float) else val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

        run_df = df[df['play_type']=='Run'].groupby(['down','yard_group'])['explosive'].mean().reset_index()
        run_df['explosive'] *= 100
        pass_df = df[df['play_type']=='Pass'].groupby(['down','yard_group'])['explosive'].mean().reset_index()
        pass_df['explosive'] *= 100
        success_df = df.groupby(['down','yard_group'])['success'].mean().reset_index()
        success_df['success'] *= 100

        def plot_heatmap(df_heat,val_col,title):
            fig = px.density_heatmap(df_heat,x='yard_group',y='down',z=val_col,
                                     text=df_heat[val_col].round(1).astype(str)+'%',
                                     color_continuous_scale='Blues',
                                     labels={val_col:title,'yard_group':'Yard Group','down':'Down'},
                                     template='plotly_dark',title=title)
            fig.update_traces(texttemplate="%{text}", textfont_size=14)
            fig.update_layout(yaxis={'categoryorder':'array','categoryarray':[1,2,3,4]})
            return fig

        st.markdown("### Explosive Plays")
        c1,c2 = st.columns(2)
        c1.plotly_chart(plot_heatmap(run_df,'explosive','Run Explosive Plays %'),use_container_width=True)
        c2.plotly_chart(plot_heatmap(pass_df,'explosive','Pass Explosive Plays %'),use_container_width=True)

        st.markdown("### Success Rate")
        st.plotly_chart(plot_heatmap(success_df,'success','Success Rate %'),use_container_width=True)

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
        if not top_concepts.empty:
            fig = px.bar(top_concepts, x="count", y="concept", orientation="h",
                         color_discrete_sequence=["#7FDBFF"], template="plotly_dark",
                         title="Top Concepts for This Situation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No plays found for this scenario.")
