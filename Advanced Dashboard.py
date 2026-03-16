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
# Sidebar Logo
# -------------------------
st.sidebar.markdown('<div style="text-align:center; margin-bottom:20px;">', unsafe_allow_html=True)
try:
    st.sidebar.image("logo_har-ber-high-school.png", width=150)
except:
    pass
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Sidebar Upload
# -------------------------
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
        st.error("No 'gain_loss' column found in uploaded file.")
        st.stop()

    # -------------------------
    # Custom Yard Groups
    # -------------------------
    def custom_yard_group(yardline):
        if pd.isna(yardline): return "Unknown"
        yardline = float(yardline)
        if 0 <= yardline <= 9: return "0 - 9"
        elif 10 <= yardline <= 19: return "10 - 19"
        elif 20 <= yardline <= 29: return "20 - 29"
        elif 30 <= yardline <= 39: return "30 - 39"
        elif 40 <= yardline <= 50: return "40 - 50"
        elif -9 <= yardline <= 0: return "-9 - 0"
        elif -19 <= yardline <= -10: return "-19 - -10"
        elif -29 <= yardline <= -20: return "-29 - -20"
        elif -39 <= yardline <= -30: return "-39 - -30"
        elif -50 <= yardline <= -40: return "-50 - -40"
        else: return "Other"

    if 'yardline' in df.columns:
        df["yard_group"] = df["yardline"].apply(custom_yard_group)
    else:
        df["yard_group"] = "Unknown"

    yard_order = ["-50 - -40", "-39 - -30", "-29 - -20", "-19 - -10", "-9 - 0", "0 - 9", "10 - 19", "20 - 29", "30 - 39", "40 - 50"]

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(["Explosive & Success Metrics", "Gain/Loss Breakdown", "Concept by Yardline", "Formation Breakdown", "Concept Breakdown"])
    tab1, tab2, tab3, tab4, tab5 = tabs

    # -------------------------
    # TAB 1: Explosive & Success Metrics
    # -------------------------
    with tab1:
        st.markdown('<div class="section-header">Explosive Play & Success Metrics</div>', unsafe_allow_html=True)
        st.markdown("**Definitions:** \n- **Explosive Plays:** Runs ≥ 10 yards, Passes ≥ 20 yards  \n- **Success:** Plays gaining ≥ 4 yards")
    
        df['explosive'] = df.apply(lambda row: row['gain_loss'] >= 10 if row.get('play_type','') == 'Run' else row['gain_loss'] >= 20, axis=1)
        df['success'] = df['gain_loss'] >= 4
    
        total_plays = len(df)
        explosive_runs_pct = df[df['play_type'] == 'Run']['explosive'].mean() * 100 if 'play_type' in df.columns else 0
        explosive_pass_pct = df[df['play_type'] == 'Pass']['explosive'].mean() * 100 if 'play_type' in df.columns else 0
        success_pct = df['success'].mean() * 100
    
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-number">{total_plays}</div><div class="metric-label">Total Plays</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-number">{explosive_runs_pct:.1f}%</div><div class="metric-label">Explosive Runs</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-number">{explosive_pass_pct:.1f}%</div><div class="metric-label">Explosive Passes</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-number">{success_pct:.1f}%</div><div class="metric-label">Overall Success %</div></div>', unsafe_allow_html=True)
    
        down_order = sorted(df['down'].dropna().unique()) if 'down' in df.columns else []
    
        def plot_heatmap_hover(df_heat, val_col, title):
            summary = df_heat.groupby(['down','yard_group']).agg(num_plays=('gain_loss','size'), avg_gain=('gain_loss','mean'), rate=(val_col,'mean')).reset_index()
            z_values = summary.pivot(index='down', columns='yard_group', values='rate').fillna(0) * 100
            piv_plays = summary.pivot(index='down', columns='yard_group', values='num_plays').fillna(0)
            piv_gain = summary.pivot(index='down', columns='yard_group', values='avg_gain').fillna(0)
            customdata_array = np.stack([piv_plays.values, piv_gain.values], axis=-1)
    
            fig = px.imshow(z_values, text_auto=".1f", aspect="auto", labels={'x':'Yard Group','y':'Down','color':title}, color_continuous_scale='Blues', template='plotly_dark', title=title)
            fig.update_traces(hovertemplate="<b>Down:</b> %{y}<br><b>Yard Group:</b> %{x}<br><b>Rate:</b> %{z:.1f}%<br><b>Plays:</b> %{customdata[0]:.0f}<br><b>Avg Gain:</b> %{customdata[1]:.1f}y<extra></extra>", customdata=customdata_array)
            if down_order:
                fig.update_layout(yaxis={'categoryorder':'array','categoryarray':down_order})
            return fig
    
        run_df = df[df['play_type']=='Run'].copy() if 'play_type' in df.columns else pd.DataFrame()
        pass_df = df[df['play_type']=='Pass'].copy() if 'play_type' in df.columns else pd.DataFrame()
        success_df = df.copy()
    
        st.markdown("### Explosive Plays")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_heatmap_hover(run_df, 'explosive', 'Run Explosive %'), use_container_width=True)
        with c2: st.plotly_chart(plot_heatmap_hover(pass_df, 'explosive', 'Pass Explosive %'), use_container_width=True)
        st.markdown("### Success Rate")
        st.plotly_chart(plot_heatmap_hover(success_df, 'success', 'Success Rate %'), use_container_width=True)

    # -------------------------
    # TAB 2: Gain/Loss Breakdown
    # -------------------------
    with tab2:
        st.markdown('<div class="section-header">Gain/Loss Breakdown</div>', unsafe_allow_html=True)
        if 'down' in df.columns:
            summary = df.groupby(['down','yard_group']).agg(avg_gain=('gain_loss', 'mean'), num_plays=('gain_loss', 'count')).reset_index()
            pivot_gain = summary.pivot(index='down', columns='yard_group', values='avg_gain').fillna(0)
            pivot_plays = summary.pivot(index='down', columns='yard_group', values='num_plays').fillna(0)
            customdata_array = np.stack([pivot_plays.values], axis=-1)
            
            fig_heat = px.imshow(pivot_gain, text_auto=".1f", color_continuous_scale='Blues', labels={'x':'Yard Group','y':'Down','color':'Avg Gain'}, template='plotly_dark', title="Average Gain / Loss by Down & Yard Group")
            fig_heat.update_traces(hovertemplate="<b>Down:</b> %{y}<br><b>Yard Group:</b> %{x}<br><b>Avg Gain:</b> %{z:.1f} yards<br><b>Plays:</b> %{customdata[0]:.0f}<extra></extra>", customdata=customdata_array)
            fig_heat.update_layout(yaxis={'categoryorder':'array','categoryarray':down_order})
            st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------
    # TAB 3: Concept by Yardline
    # -------------------------
    with tab3:
        st.markdown('<div class="section-header">Concept Effectiveness by Field Zone</div>', unsafe_allow_html=True)
        if 'concept' in df.columns:
            c1, c2 = st.columns(2)
            with c1: min_plays = st.number_input("Min Plays to Show Concept", min_value=1, value=2, step=1)
            with c2: metric_to_show = st.selectbox("Select Metric", ["Success Rate %", "Average Gain", "Explosive Play %"])

            metric_map = {"Success Rate %": "success", "Average Gain": "gain_loss", "Explosive Play %": "explosive"}
            concept_summary = df.groupby(['concept', 'yard_group']).agg(plays=('gain_loss', 'count'), avg_gain=('gain_loss', 'mean'), success_rate=('success', 'mean'), explosive_rate=('explosive', 'mean')).reset_index()
            concept_summary = concept_summary[concept_summary['plays'] >= min_plays]

            if concept_summary.empty:
                st.warning("No concepts met the threshold.")
            else:
                if metric_to_show == "Success Rate %": concept_summary['display_val'] = (concept_summary['success_rate'] * 100).round(1)
                elif metric_to_show == "Explosive Play %": concept_summary['display_val'] = (concept_summary['explosive_rate'] * 100).round(1)
                else: concept_summary['display_val'] = concept_summary['avg_gain'].round(1)

                pivot_concept = concept_summary.pivot(index='concept', columns='yard_group', values='display_val').fillna(0)
                pivot_plays = concept_summary.pivot(index='concept', columns='yard_group', values='plays').fillna(0)
                existing_yard_order = [y for y in yard_order if y in pivot_concept.columns]
                pivot_concept = pivot_concept[existing_yard_order]
                pivot_plays = pivot_plays[existing_yard_order]

                fig_concept = px.imshow(pivot_concept, text_auto=True, aspect="auto", color_continuous_scale='Blues', template='plotly_dark', labels={'x': 'Field Zone', 'y': 'Play Concept', 'color': metric_to_show}, title=f"{metric_to_show} by Concept and Yardline")
                fig_concept.update_traces(hovertemplate="<b>Concept:</b> %{y}<br><b>Yard Group:</b> %{x}<br><b>Value:</b> %{z}<br><b>Plays:</b> %{customdata[0]:.0f}<extra></extra>", customdata=np.stack([pivot_plays.values], axis=-1))
                st.plotly_chart(fig_concept, use_container_width=True)

    # -------------------------
    # TAB 4: Formation Breakdown
    # -------------------------
    with tab4:
        st.markdown('<div class="section-header">Formation Effectiveness by Down</div>', unsafe_allow_html=True)
        if 'formation' in df.columns and 'down' in df.columns:
            form_summary = df.groupby(['formation', 'down']).agg(avg_gain=('gain_loss', 'mean'), num_plays=('gain_loss', 'count')).reset_index()
            form_summary = form_summary[form_summary['num_plays'] >= 2]
            pivot_form = form_summary.pivot(index='formation', columns='down', values='avg_gain').fillna(0)
            pivot_form_plays = form_summary.pivot(index='formation', columns='down', values='num_plays').fillna(0)
            
            if down_order:
                existing_downs = [d for d in down_order if d in pivot_form.columns]
                pivot_form, pivot_form_plays = pivot_form[existing_downs], pivot_form_plays[existing_downs]

            fig_form = px.imshow(pivot_form, text_auto=True, aspect="auto", color_continuous_scale='Blues', template='plotly_dark', labels={'x': 'Down', 'y': 'Formation', 'color': 'Avg Gain'}, title="Average Gain by Formation and Down")
            fig_form.update_traces(hovertemplate="<b>Formation:</b> %{y}<br><b>Down:</b> %{x}<br><b>Avg Gain:</b> %{z:.1f}y<br><b>Plays:</b> %{customdata[0]:.0f}<extra></extra>", customdata=np.stack([pivot_form_plays.values], axis=-1))
            st.plotly_chart(fig_form, use_container_width=True)

    # -------------------------
    # TAB 5: Concept Breakdown
    # -------------------------
    with tab5:
        st.markdown('<div class="section-header">Concept Breakdown</div>', unsafe_allow_html=True)
        if 'concept' in df.columns:
            summary = df.groupby('concept').agg(plays=('gain_loss','size'), avg_gain=('gain_loss','mean'), success_pct=('success','mean'), explosive_pct=('explosive','mean')).reset_index()
            summary = summary[summary['plays'] >= 3]
            
            st.dataframe(summary.sort_values('success_pct', ascending=False), use_container_width=True)

            summary_success = summary.sort_values('success_pct')
            fig = px.bar(summary_success, x='success_pct', y='concept', orientation='h', template='plotly_dark', title="Concept Success Rate", labels={'success_pct':'Success %'}, color='success_pct', color_continuous_scale='Blues')
            fig.update_traces(hovertemplate="<b>Concept:</b> %{y}<br><b>Success %:</b> %{x:.1%}<br><b>Plays:</b> %{customdata[0]}<br><b>Avg Gain:</b> %{customdata[1]:.1f}y<extra></extra>", customdata=np.stack([summary_success['plays'], summary_success['avg_gain']], axis=-1))
            st.plotly_chart(fig, use_container_width=True)

            summary_explosive = summary.sort_values('explosive_pct')
            fig2 = px.bar(summary_explosive, x='explosive_pct', y='concept', orientation='h', template='plotly_dark', title="Concept Explosive Play %", labels={'explosive_pct':'Explosive %'}, color='explosive_pct', color_continuous_scale='Blues')
            fig2.update_traces(hovertemplate="<b>Concept:</b> %{y}<br><b>Explosive %:</b> %{x:.1%}<br><b>Plays:</b> %{customdata[0]}<br><b>Avg Gain:</b> %{customdata[1]:.1f}y<extra></extra>", customdata=np.stack([summary_explosive['plays'], summary_explosive['avg_gain']], axis=-1))
            st.plotly_chart(fig2, use_container_width=True)


