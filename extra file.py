    # -------------------------
    # TAB 5: Play Call Advisor
    # -------------------------
    with tab5:
        st.markdown('<div class="section-header">Play Call Advisor</div>', unsafe_allow_html=True)
    
        down_input = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="advisor_down")
        dist_input = st.slider("Distance", 1, 20, 5, key="advisor_distance")
        yard_input = st.slider("Yardline", -50, 50, 0, key="advisor_yardline")
    
        advisor_df = df[
            (df["down"] == down_input) &
            (df["distance"] == dist_input) &
            (df["yardline"] == yard_input)
        ]
    
        if advisor_df.empty:
            st.warning("No plays available for this selection.")
        else:
            top_concepts = advisor_df["concept"].value_counts().head(5).reset_index()
            top_concepts.columns = ["concept", "count"]
    
            fig = px.bar(
                top_concepts,
                x="count",
                y="concept",
                orientation="h",
                color_discrete_sequence=["#7FDBFF"],
                template="plotly_dark",
                title="Top Concepts for This Situation"
            )
            st.plotly_chart(fig, use_container_width=True)
    
            st.markdown('<div class="section-header">Play Data</div>', unsafe_allow_html=True)
            st.dataframe(
                advisor_df[["concept","play_type","play_direction","gain_loss"]].sort_values("gain_loss", ascending=False),
                use_container_width=True
            )
    
    # -------------------------
    # TAB 6: Defensive Tendencies
    # -------------------------
    with tab6:
        st.markdown('<div class="section-header">Defensive Tendencies</div>', unsafe_allow_html=True)
    
        down_input = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="defense_down")
        yard_input = st.slider("Yardline", -50, 50, 0, key="defense_yardline")
    
        defense_df = df[(df["down"] == down_input) & (df["yardline"] == yard_input)]
    
        if defense_df.empty:
            st.warning("No plays for this selection.")
        else:
            defense_summary = defense_df.groupby(["play_type","concept"]).size().reset_index(name="count")
            pivot_df = defense_summary.pivot(index="play_type", columns="concept", values="count").fillna(0)
    
            fig = px.imshow(
                pivot_df,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x":"Play Concept","y":"Play Type","color":"Count"},
                title="Defensive Tendencies Heatmap",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
            st.markdown('<div class="section-header">Raw Play Counts</div>', unsafe_allow_html=True)
            st.dataframe(defense_summary.sort_values("count", ascending=False), use_container_width=True)
    
    # -------------------------
    # TAB 7: Opponent Play Predictor
    # -------------------------
    with tab7:
        st.markdown('<div class="section-header">Opponent Play Predictor</div>', unsafe_allow_html=True)
    
        if opponent_file:
            try:
                opp_df = pd.read_excel(opponent_file)
                opp_df.columns = opp_df.columns.str.lower().str.strip()
                st.write("Columns in uploaded file:", opp_df.columns.tolist())
    
                # Standardize columns
                COLUMN_MAP = {
                    "down": ["down", "dn"],
                    "distance": ["dist", "togo", "yards to go", "ydstogo"],
                    "yardline": ["yard ln", "spot", "ball on"],
                    "concept": ["off play"],
                    "play_type": ["play type", "playtype", "type"]
                }
                rename_dict = {}
                for standard, variants in COLUMN_MAP.items():
                    for col in opp_df.columns:
                        if col in variants:
                            rename_dict[col] = standard
                opp_df = opp_df.rename(columns=rename_dict)
    
                model_df = opp_df.dropna(subset=["down", "distance", "yardline", "concept", "play_type"])
                if model_df.empty:
                    st.warning("No usable data after cleaning. Check column names.")
                else:
                    X = model_df[["down", "distance", "yardline"]]
                    y_concept = model_df["concept"]
                    y_type = model_df["play_type"]
    
                    concept_model = RandomForestClassifier(n_estimators=200, random_state=42)
                    type_model = RandomForestClassifier(n_estimators=200, random_state=42)
                    concept_model.fit(X, y_concept)
                    type_model.fit(X, y_type)
    
                    st.success("Opponent model trained successfully!")
    
                    # User Inputs for Prediction
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        down_input = st.selectbox("Down", sorted(model_df["down"].dropna().unique()), key="predictor_down")
                    with c2:
                        dist_input = st.slider("Distance", 1, 20, 5, key="predictor_distance")
                    with c3:
                        yard_input = st.slider("Yardline", -50, 50, 0, key="predictor_yardline")
    
                    pred_df = pd.DataFrame({"down": [down_input], "distance": [dist_input], "yardline": [yard_input]})
    
                    concept_probs = concept_model.predict_proba(pred_df)[0]
                    concept_names = concept_model.classes_
                    type_probs = type_model.predict_proba(pred_df)[0]
                    type_names = type_model.classes_
    
                    type_results = dict(zip(type_names, type_probs))
                    run_prob = type_results.get("Run", 0) * 100
                    pass_prob = type_results.get("Pass", 0) * 100
    
                    concept_df = pd.DataFrame({"concept": concept_names, "prob": concept_probs * 100}).sort_values("prob", ascending=False)
                    top3 = concept_df.head(3)
    
                    c1, c2 = st.columns(2)
                    c1.markdown(f'<div class="metric-card"><div class="metric-number">{round(run_prob,1)}%</div><div class="metric-label">Run Probability</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="metric-card"><div class="metric-number">{round(pass_prob,1)}%</div><div class="metric-label">Pass Probability</div></div>', unsafe_allow_html=True)
    
                    fig = px.bar(
                        top3,
                        x="prob",
                        y="concept",
                        orientation="h",
                        color_discrete_sequence=["#7FDBFF"],
                        template="plotly_dark",
                        title="Top 3 Predicted Plays"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
            except Exception as e:
                st.error(f"Error processing opponent file: {e}")
        else:
            st.info("Upload an opponent Excel file in the sidebar to train the play predictor.")
    
    # -------------------------
    # TAB 8: Play Call Win Probability
    # -------------------------
    with tab8:
        st.markdown('<div class="section-header">Play Call Win Probability</div>', unsafe_allow_html=True)
    
        st.info("""
        **How to read this tab:**
        - **Expected Gain:** Average yards gained historically for this play/concept.
        - **Success %:** Percentage of plays gaining ≥ 4 yards or distance to go.
        - **Explosive %:** Percentage of plays gaining 20+ yards.
        - **Best Concept:** Concept with highest Expected Gain × Success %.
        """)
    
        down_input = st.selectbox("Down", sorted(df["down"].dropna().unique()), key="winprob_down")
        dist_input = st.slider("Distance to Go", 1, 20, 5, key="winprob_distance")
        yard_input = st.slider("Yardline", -50, 50, 0, key="winprob_yardline")
        play_type_input = st.multiselect(
            "Play Concepts to Evaluate",
            df["concept"].unique(),
            default=df["concept"].unique()[:5],
            key="winprob_concepts"
        )
    
        hist_df = df[
            (df["down"] == down_input) &
            (df["distance"] == dist_input) &
            (df["yardline"] == yard_input) &
            (df["concept"].isin(play_type_input))
        ].copy()
    
        if hist_df.empty:
            st.warning("No historical plays found for this combination. Adjust inputs.")
        else:
            hist_df["success"] = hist_df["gain_loss"] >= max(4, dist_input)
            hist_df["explosive"] = hist_df["gain_loss"] >= 20
    
            summary = hist_df.groupby("concept").agg(
                expected_gain=("gain_loss","mean"),
                success_pct=("success","mean"),
                explosive_pct=("explosive","mean")
            ).reset_index()
    
            summary["rank_score"] = summary["expected_gain"] * summary["success_pct"]
            best_play = summary.sort_values("rank_score", ascending=False).iloc[0]
    
            c1, c2, c3, c4 = st.columns(4)
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
    
            st.markdown('<div class="section-header">Detailed Play Stats</div>', unsafe_allow_html=True)
            summary_display = summary.copy()
            summary_display["expected_gain"] = summary_display["expected_gain"].round(1)
            summary_display["success_pct"] = (summary_display["success_pct"]*100).round(1).astype(str) + "%"
            summary_display["explosive_pct"] = (summary_display["explosive_pct"]*100).round(1).astype(str) + "%"
            summary_display = summary_display.sort_values("rank_score", ascending=False)
            st.dataframe(
                summary_display[["concept","expected_gain","success_pct","explosive_pct"]],
                use_container_width=True
            )


    # -------------------------
    # TAB 4: Play Success Predictor
    # -------------------------
    with tab4:
    
        st.markdown('<div class="section-header">Play Success Predictor</div>', unsafe_allow_html=True)
    
        if {'down','distance','yardline','play_type','gain_loss', 'formation'}.issubset(df.columns):
    
            model_df = df.copy()
    
            # Define success
            model_df['success'] = model_df['gain_loss'] >= 4
    
            # Keep only needed columns
            model_df = model_df[['down','distance','yardline','play_type','success']]
    
            # Encode play type
            model_df = pd.get_dummies(model_df, columns=['play_type'])
    
            features = model_df.drop(columns=['success'])
            target = model_df['success']
    
            # Fill missing values
            features = features.fillna(0)
    
            # Train model
            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(features, target)
    
            st.markdown("### Enter Game Situation")
    
            c1, c2, c3 = st.columns(3)
    
            down = c1.selectbox("Down", sorted(df['down'].dropna().unique()))
            distance = c2.slider("Distance", 1, 20, 5)
            yardline = c3.slider("Yardline", -50, 50, 0)
    
            play_type = st.selectbox("Play Type", df['play_type'].dropna().unique())
    
            # Build prediction row
            input_dict = {
                'down': down,
                'distance': distance,
                'yardline': yardline
            }
    
            # Add play type columns
            for col in features.columns:
                if col.startswith("play_type_"):
                    input_dict[col] = 1 if col == f"play_type_{play_type}" else 0
    
            input_df = pd.DataFrame([input_dict])
    
            # Ensure same column order
            input_df = input_df.reindex(columns=features.columns, fill_value=0)
    
            # Predict
            prob = model.predict_proba(input_df)[0][1]
    
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-number">{prob*100:.1f}%</div>
            <div class="metric-label">Predicted Success Probability</div>
            </div>
            """, unsafe_allow_html=True)
    
        else:
            st.warning("Not enough columns for machine learning model.")





-----------------------------------------------------------


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

st.sidebar.image("logo_har-ber-high-school.png", width=150)

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



    # Ensure numeric gain_loss

    if 'gain_loss' in df.columns:

        df['gain_loss'] = pd.to_numeric(df['gain_loss'], errors='coerce').fillna(0)

    else:

        st.error("No 'gain_loss' column found in uploaded file.")

        st.stop()



    # -------------------------
    # Custom Yard Groups
    # -------------------------

    def custom_yard_group(yardline):

        if pd.isna(yardline):

            return "Unknown"

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



    yard_order = [

        "-50 - -40", "-39 - -30", "-29 - -20", "-19 - -10", "-9 - 0",

        "0 - 9", "10 - 19", "20 - 29", "30 - 39", "40 - 50"

    ]



    # -------------------------
    # Tabs
    # -------------------------

    tabs = st.tabs([

        "Explosive & Success Metrics",

        "Gain/Loss Breakdown",

        "Concept by Yardline",

        "Formation Breakdown",

        "Concept Breakdown"

    ])

    tab1, tab2, tab3, tab4, tab5 = tabs





    # -------------------------
    # TAB 1: Explosive & Success Metrics
    # -------------------------

    with tab1:

        st.markdown('<div class="section-header">Explosive Play & Success Metrics</div>', unsafe_allow_html=True)

        st.markdown("""

        **Definitions:**  

        - **Explosive Plays:** Runs ≥ 10 yards, Passes ≥ 20 yards  

        - **Success:** Plays gaining ≥ 4 yards

        """)

    

        # Add explosive & success flags

        df['explosive'] = df.apply(

            lambda row: row['gain_loss'] >= 10 if row.get('play_type','') == 'Run' else row['gain_loss'] >= 20, axis=1

        )

        df['success'] = df['gain_loss'] >= 4

    

        # Metrics cards

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

    

        # -------------------------
        # Heatmap function with customdata hover
        # -------------------------

        def plot_heatmap_hover(df_heat, val_col, title):

            # Aggregate plays, avg_gain, and rate

            summary = df_heat.groupby(['down','yard_group']).agg(

                num_plays=('gain_loss','size'),

                avg_gain=('gain_loss','mean'),

                rate=(val_col,'mean')  # fraction 0-1

            ).reset_index()

    

            # Pivot for z-values and customdata

            z_values = summary.pivot(index='down', columns='yard_group', values='rate').fillna(0) * 100  # % for cell text

            customdata_df = summary.pivot(index='down', columns='yard_group', values=['num_plays','avg_gain']).fillna(0)

            customdata_array = np.stack([customdata_df['num_plays'].values, customdata_df['avg_gain'].values], axis=-1)

    

            # Create heatmap

            fig = px.imshow(

                z_values,

                text_auto=True,

                aspect="auto",

                labels={'x':'Yard Group','y':'Down','color':title},

                color_continuous_scale='Blues',

                template='plotly_dark',

                title=title

            )





            fig.update_traces(

                hovertemplate="<b>Down:</b> %{y}<br>"

                              "<b>Yard Group:</b> %{x}<br>"

                              f"<b>{title}:</b> %{{z:.1f}}%<br>"

                              "<b>Number of Plays:</b> %{customdata[0]:.0f}<br>"

                              "<b>Average Gain:</b> %{customdata[1]:.1f} yards",

                customdata=customdata_array

            )

    

            if down_order:

                fig.update_layout(yaxis={'categoryorder':'array','categoryarray':down_order})

    

            return fig

    

        # -------------------------
        # Prepare filtered DataFrames
        # -------------------------

        run_df = df[df['play_type']=='Run'].copy() if 'play_type' in df.columns else pd.DataFrame()

        pass_df = df[df['play_type']=='Pass'].copy() if 'play_type' in df.columns else pd.DataFrame()

        success_df = df.copy()

    

        # Convert booleans to numeric for % calculation

        run_df['explosive'] = run_df['explosive'].astype(float)

        pass_df['explosive'] = pass_df['explosive'].astype(float)

        success_df['success'] = success_df['success'].astype(float)

    

        # -------------------------
        # Display heatmaps
        # -------------------------

        st.markdown("### Explosive Plays")

        c1, c2 = st.columns(2)

        with c1:

            st.plotly_chart(plot_heatmap_hover(run_df, 'explosive', 'Run Explosive Plays %'), use_container_width=True)

        with c2:

            st.plotly_chart(plot_heatmap_hover(pass_df, 'explosive', 'Pass Explosive Plays %'), use_container_width=True)

    

        st.markdown("### Success Rate")

        st.plotly_chart(plot_heatmap_hover(success_df, 'success', 'Success Rate %'), use_container_width=True)



    # -------------------------
    # TAB 2: Gain/Loss Breakdown
    # -------------------------

    with tab2:

        st.markdown('<div class="section-header">Gain/Loss Breakdown</div>', unsafe_allow_html=True)

    

        if 'down' in df.columns:

            # Aggregate average gain AND play count

            summary = df.groupby(['down','yard_group']).agg(

                avg_gain=('gain_loss', 'mean'),

                num_plays=('gain_loss', 'count')

            ).reset_index()

            

            summary['avg_gain'] = summary['avg_gain'].round(1)

            

            # Pivot for z-values (avg gain) and customdata (number of plays)

            pivot_gain = summary.pivot(index='down', columns='yard_group', values='avg_gain').fillna(0)

            pivot_plays = summary.pivot(index='down', columns='yard_group', values='num_plays').fillna(0)

            

            # Stack the custom data for Plotly

            customdata_array = np.stack([pivot_plays.values], axis=-1)

            

            fig_heat = px.imshow(

                pivot_gain, 

                text_auto=True, 

                color_continuous_scale='Blues',

                labels={'x':'Yard Group','y':'Down','color':'Avg Gain'},

                template='plotly_dark',

                title="Average Gain / Loss by Down & Yard Group"

            )

            

            # Add custom hover template

            fig_heat.update_traces(

                hovertemplate="<b>Down:</b> %{y}<br>"

                              "<b>Yard Group:</b> %{x}<br>"

                              "<b>Avg Gain:</b> %{z} yards<br>"

                              "<b>Number of Plays:</b> %{customdata[0]:.0f}<extra></extra>",

                customdata=customdata_array

            )

            

            fig_heat.update_layout(yaxis={'categoryorder':'array','categoryarray':down_order})

            st.plotly_chart(fig_heat, use_container_width=True)

    

    # -------------------------
    # TAB 3: Concept by Yardline
    # -------------------------

    with tab3:

        st.markdown('<div class="section-header">Concept Effectiveness by Field Zone</div>', unsafe_allow_html=True)

        

        if 'concept' in df.columns:

            c1, c2 = st.columns(2)

            with c1:

                min_plays = st.number_input("Min Plays to Show Concept", min_value=1, value=2, step=1)

            with c2:

                metric_to_show = st.selectbox("Select Metric", ["Success Rate %", "Average Gain", "Explosive Play %"])



            metric_map = {"Success Rate %": "success", "Average Gain": "gain_loss", "Explosive Play %": "explosive"}

            target_col = metric_map[metric_to_show]



            concept_summary = df.groupby(['concept', 'yard_group']).agg(

                plays=('gain_loss', 'count'),

                avg_gain=('gain_loss', 'mean'),

                success_rate=('success', 'mean'),

                explosive_rate=('explosive', 'mean')

            ).reset_index()



            concept_summary = concept_summary[concept_summary['plays'] >= min_plays]



            if concept_summary.empty:

                st.warning("No concepts met the minimum play threshold for these yard groups.")

            else:

                if metric_to_show == "Success Rate %":

                    concept_summary['display_val'] = (concept_summary['success_rate'] * 100).round(1)

                elif metric_to_show == "Explosive Play %":

                    concept_summary['display_val'] = (concept_summary['explosive_rate'] * 100).round(1)

                else:

                    concept_summary['display_val'] = concept_summary['avg_gain'].round(1)



                # Pivot for Heatmap (z-values) and Custom Data (number of plays)

                pivot_concept = concept_summary.pivot(index='concept', columns='yard_group', values='display_val').fillna(0)

                pivot_plays = concept_summary.pivot(index='concept', columns='yard_group', values='plays').fillna(0)

                

                # Reorder columns based on field position

                existing_yard_order = [y for y in yard_order if y in pivot_concept.columns]

                pivot_concept = pivot_concept[existing_yard_order]

                pivot_plays = pivot_plays[existing_yard_order]



                # Stack custom data

                customdata_array = np.stack([pivot_plays.values], axis=-1)



                fig_concept = px.imshow(

                    pivot_concept,

                    text_auto=True,

                    aspect="auto",

                    color_continuous_scale='Blues',

                    template='plotly_dark',

                    labels={'x': 'Field Zone', 'y': 'Play Concept', 'color': metric_to_show},

                    title=f"{metric_to_show} by Concept and Yardline"

                )

                

                # Apply custom hover data

                fig_concept.update_traces(

                    hovertemplate="<b>Concept:</b> %{y}<br>"

                                  "<b>Yard Group:</b> %{x}<br>"

                                  f"<b>{metric_to_show}:</b> %{{z}}<br>"

                                  "<b>Number of Plays:</b> %{customdata[0]:.0f}<extra></extra>",

                    customdata=customdata_array

                )

                

                st.plotly_chart(fig_concept, use_container_width=True)



    # -------------------------
    # TAB 4: Formation Breakdown
    # -------------------------

    with tab4:

        st.markdown('<div class="section-header">Formation Effectiveness by Down</div>', unsafe_allow_html=True)

        

        if 'formation' in df.columns and 'down' in df.columns:

            # Aggregate average gain and play count

            form_summary = df.groupby(['formation', 'down']).agg(

                avg_gain=('gain_loss', 'mean'),

                num_plays=('gain_loss', 'count')

            ).reset_index()

            

            # Filter out noise (optional, but good for formations)

            form_summary = form_summary[form_summary['num_plays'] >= 2]

            form_summary['avg_gain'] = form_summary['avg_gain'].round(1)

            

            # Pivot data

            pivot_form = form_summary.pivot(index='formation', columns='down', values='avg_gain').fillna(0)

            pivot_form_plays = form_summary.pivot(index='formation', columns='down', values='num_plays').fillna(0)

            

            # Ensure down order if it exists

            if down_order:
                existing_downs = [d for d in down_order if d in pivot_form.columns]
                pivot_form = pivot_form[existing_downs]
                pivot_form_plays = pivot_form_plays[existing_downs]

            customdata_form = np.stack([pivot_form_plays.values], axis=-1)



            fig_form = px.imshow(
                pivot_form,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                template='plotly_dark',
                labels={'x': 'Down', 'y': 'Formation', 'color': 'Avg Gain'},
                title="Average Gain by Formation and Down"

            )

        
            fig_form.update_traces(

                hovertemplate="<b>Formation:</b> %{y}<br>"
                              "<b>Down:</b> %{x}<br>"
                              "<b>Avg Gain:</b> %{z} yards<br>"
                              "<b>Number of Plays:</b> %{customdata[0]:.0f}<extra></extra>",

                customdata=customdata_form

            )

            st.plotly_chart(fig_form, use_container_width=True)

        else:

            st.warning("The 'formation' or 'down' columns were not found in your file.")

    # -------------------------
    # TAB 5: Concept Breakdown
    # -------------------------

    with tab5:

        st.markdown('<div class="section-header">Concept Breakdown</div>', unsafe_allow_html=True)

        if 'concept' in df.columns:

            concept_df = df.copy()

    

            concept_df['success'] = concept_df['gain_loss'] >= 4
            concept_df['explosive'] = concept_df.apply(
                lambda row: row['gain_loss'] >= 10 if row.get('play_type','') == 'Run'
                else row['gain_loss'] >= 20, axis=1

            )

            summary = concept_df.groupby('concept').agg(

                plays=('gain_loss','size'),
                avg_gain=('gain_loss','mean'),
                success_pct=('success','mean'),
                explosive_pct=('explosive','mean')
            ).reset_index()
            summary = summary[summary['plays'] >= 3]

            display = summary.copy()
            display['avg_gain'] = display['avg_gain'].round(1)
            display['success_pct'] = (display['success_pct']*100).round(1)
            display['explosive_pct'] = (display['explosive_pct']*100).round(1)


            st.dataframe(display.sort_values('success_pct', ascending=False),

                         use_container_width=True)

            # --- Success Rate Bar Chart ---

            summary_success = summary.sort_values('success_pct')

            fig = px.bar(

                summary_success,
                x='success_pct',
                y='concept',
                orientation='h',
                template='plotly_dark',
                title="Concept Success Rate",
                labels={'success_pct':'Success %','concept':'Concept'},
                color='success_pct',
                color_continuous_scale='Blues',
                customdata=['plays', 'avg_gain'] # Pass columns for hover
            )

            

            fig.update_traces(
                hovertemplate="<b>Concept:</b> %{y}<br>"
                              "<b>Success Rate:</b> %{x:.1%}<br>"
                              "<b>Number of Plays:</b> %{customdata[0]}<br>"
                              "<b>Average Gain:</b> %{customdata[1]:.1f} yards<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)


            # --- Explosive % Bar Chart ---

            summary_explosive = summary.sort_values('explosive_pct')

            fig2 = px.bar(

                summary_explosive,
                x='explosive_pct',
                y='concept',
                orientation='h',
                template='plotly_dark',
                title="Concept Explosive Play %",
                labels={'explosive_pct':'Explosive %','concept':'Concept'},
                color='explosive_pct',
                color_continuous_scale='Blues',
                customdata=['plays', 'avg_gain'] # Pass columns for hover
            )
            
            fig2.update_traces(

                hovertemplate="<b>Concept:</b> %{y}<br>"
                              "<b>Explosive Rate:</b> %{x:.1%}<br>"
                              "<b>Number of Plays:</b> %{customdata[0]}<br>"
                              "<b>Average Gain:</b> %{customdata[1]:.1f} yards<extra></extra>"

            )
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.warning("No 'concept' column found in dataset.")
