import streamlit as st
import pandas as pd
import numpy as np
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
# Sidebar Logo & Upload
# -------------------------
st.sidebar.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
# Ensure this file exists in your directory
try:
    st.sidebar.image("logo_har-ber-high-school.png", width=150)
except:
    pass 
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.title("Har‑Ber Football Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Hudl Excel File", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    # -------------------------
    # Column Renaming
    # -------------------------
    COLUMN_MAP = {
        "down": ["down", "dn"],
        "distance": ["dist", "togo", "yards to go", "ydstogo"],
        "yardline": ["yard ln", "spot", "ball on"],
        "concept": ["off play", "concept"],
        "play_type": ["play type", "playtype", "type"],
        "play_direction": ["play dir"],
        "gain_loss": ["gn/ls", "gain_loss", "gain loss"],
        "formation": ["off form"]
    }
    rename_dict = {}
    for s, v in COLUMN_MAP.items():
        for col in df.columns:
            if col in v:
                rename_dict[col] = s
    df = df.rename(columns=rename_dict)

    if 'gain_loss' in df.columns:
        df['gain_loss'] = pd.to_numeric(df['gain_loss'], errors='coerce').fillna(0)
    else:
        st.error("No 'gain_loss' column found.")
        st.stop()

    # -------------------------
    # Custom Yard Groups
    # -------------------------
    def custom_yard_group(yardline):
        if pd.isna(yardline): return "Unknown"
        y = float(yardline)
        if 0 <= y <= 9: return "0 - 9"
        elif 10 <= y <= 19: return "10 - 19"
        elif 20 <= y <= 29: return "20 - 29"
        elif 30 <= y <= 39: return "30 - 39"
        elif 40 <= y <= 50: return "40 - 50"
        elif -9 <= y <= 0: return "-9 - 0"
        elif -19 <= y <= -10: return "-19 - -10"
        elif -29 <= y <= -20: return "-29 - -20"
        elif -39 <= y <= -30: return "-39 - -30"
        elif -50 <= y <= -40: return "-50 - -40"
        else: return "Other"

    df["yard_group"] = df["yardline"].apply(custom_yard_group) if 'yardline' in df.columns else "Unknown"
    yard_order = ["-50 - -40", "-39 - -30", "-29 - -20", "-19 - -10", "-9 - 0", "0 - 9", "10 - 19", "20 - 29", "30 - 39", "40 - 50"]

    # -------------------------
    # Shared Logic
    # -------------------------
    df['explosive'] = df.apply(lambda row: row['gain_loss'] >= 10 if row.get('play_type','') == 'Run' else row['gain_loss'] >= 20, axis=1)
    df['success'] = df['gain_loss'] >= 4
    down_order = sorted([x for x in df['down'].dropna().unique() if isinstance(x, (int, float))])

    # -------------------------
    # Tabs Layout
    # -------------------------
    tabs = st.tabs(["Explosive & Success", "Gain/Loss", "Concept/Yardline", "Formation", "Concept Breakdown"])
    tab1, tab2, tab3, tab4, tab5 = tabs

    # TAB 1: Explosive & Success
    with tab1:
        st.markdown('<div class="section-header">Explosive & Success Heatmaps</div>', unsafe_allow_html=True)
        
        def plot_heatmap_hover(df_heat, val_col, title):
            if df_heat.empty: return None
            summary = df_heat.groupby(['down','yard_group']).agg(
                num_plays=('gain_loss','size'),
                avg_gain=('gain_loss','mean'),
                rate=(val_col,'mean')
            ).reset_index()
            
            z_piv = summary.pivot(index='down', columns='yard_group', values='rate').fillna(0) * 100
            plays_piv = summary.pivot(index='down', columns='yard_group', values='num_plays').fillna(0)
            gain_piv = summary.pivot(index='down', columns='yard_group', values='avg_gain').fillna(0)
            
            c_data = np.stack([plays_piv.values, gain_piv.values], axis=-1)
            
            fig = px.imshow(z_piv, text_auto=".1f", color_continuous_scale='Blues', template='plotly_dark', title=title)
            fig.update_traces(
                hovertemplate="<b>Down:</b> %{y}<br><b>Zone:</b> %{x}<br><b>Rate:</b> %{z:.1f}%<br><b>Plays:</b> %{customdata[0]}<br><b>Avg:</b> %{customdata[1]:.1f}y<extra></extra>",
                customdata=c_data
            )
            return fig

        st.plotly_chart(plot_heatmap_hover(df[df['play_type']=='Run'], 'explosive', 'Run Explosive %'), use_container_width=True)
        st.plotly_chart(plot_heatmap_hover(df, 'success', 'Overall Success %'), use_container_width=True)

    # TAB 2: Gain/Loss
    with tab2:
        st.markdown('<div class="section-header">Gain/Loss Breakdown</div>', unsafe_allow_html=True)
        sum2 = df.groupby(['down','yard_group']).agg(avg_gain=('gain_loss','mean'), plays=('gain_loss','count')).reset_index()
        z_piv = sum2.pivot(index='down', columns='yard_group', values='avg_gain').fillna(0)
        p_piv = sum2.pivot(index='down', columns='yard_group', values='plays').fillna(0)
        
        fig2 = px.imshow(z_piv, text_auto=".1f", color_continuous_scale='Blues', template='plotly_dark')
        fig2.update_traces(
            hovertemplate="<b>Down:</b> %{y}<br><b>Zone:</b> %{x}<br><b>Avg Gain:</b> %{z:.1f}y<br><b>Plays:</b> %{customdata[0]}<extra></extra>",
            customdata=np.stack([p_piv.values], axis=-1)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # TAB 3: Concept by Yardline
    with tab3:
        st.markdown('<div class="section-header">Concept Effectiveness</div>', unsafe_allow_html=True)
        if 'concept' in df.columns:
            sum3 = df.groupby(['concept', 'yard_group']).agg(plays=('gain_loss','count'), avg=('gain_loss','mean')).reset_index()
            sum3 = sum3[sum3['plays'] >= 2]
            z_piv = sum3.pivot(index='concept', columns='yard_group', values='avg').fillna(0)
            p_piv = sum3.pivot(index='concept', columns='yard_group', values='plays').fillna(0)
            
            fig3 = px.imshow(z_piv, text_auto=".1f", color_continuous_scale='Blues', template='plotly_dark')
            fig3.update_traces(
                hovertemplate="<b>Concept:</b> %{y}<br><b>Zone:</b> %{x}<br><b>Avg:</b> %{z:.1f}y<br><b>Plays:</b> %{customdata[0]}<extra></extra>",
                customdata=np.stack([p_piv.values], axis=-1)
            )
            st.plotly_chart(fig3, use_container_width=True)

    # TAB 4: Formation Breakdown
    with tab4:
        st.markdown('<div class="section-header">Formation Effectiveness</div>', unsafe_allow_html=True)
        if 'formation' in df.columns:
            sum4 = df.groupby(['formation', 'down']).agg(avg=('gain_loss','mean'), plays=('gain_loss','count')).reset_index()
            z_piv = sum4.pivot(index='formation', columns='down', values='avg').fillna(0)
            p_piv = sum4.pivot(index='formation', columns='down', values='plays').fillna(0)
            
            fig4 = px.imshow(z_piv, text_auto=".1f", color_continuous_scale='Blues', template='plotly_dark')
            fig4.update_traces(
                hovertemplate="<b>Form:</b> %{y}<br><b>Down:</b> %{x}<br><b>Avg:</b> %{z:.1f}y<br><b>Plays:</b> %{customdata[0]}<extra></extra>",
                customdata=np.stack([p_piv.values], axis=-1)
            )
            st.plotly_chart(fig4, use_container_width=True)

    # TAB 5: Concept Breakdown (Bar Charts)
    with tab5:
        st.markdown('<div class="section-header">Concept Success vs Explosive</div>', unsafe_allow_html=True)
        if 'concept' in df.columns:
            sum5 = df.groupby('concept').agg(plays=('gain_loss','size'), avg=('gain_loss','mean'), s_pct=('success','mean'), e_pct=('explosive','mean')).reset_index()
            sum5 = sum5[sum5['plays'] >= 3].sort_values('s_pct')
            
            # Use np.stack for the bar chart customdata to avoid the error
            c_data_bar = np.stack([sum5['plays'], sum5['avg']], axis=-1)
            
            fig5 = px.bar(sum5, x='s_pct', y='concept', orientation='h', color='s_pct', color_continuous_scale='Blues', template='plotly_dark')
            fig5.update_traces(
                hovertemplate="<b>Concept:</b> %{y}<br><b>Success Rate:</b> %{x:.1%}<br><b>Plays:</b> %{customdata[0]}<br><b>Avg:</b> %{customdata[1]:.1f}y<extra></extra>",
                customdata=c_data_bar
            )
            st.plotly_chart(fig5, use_container_width=True)
