import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har‑Ber Football Analytics", layout="wide")

st.markdown("""
<style>
/* -------------------------
   Metric Cards (Compact)
------------------------- */
.metric-card {
    background-color: #0A2342;       /* card color */
    padding: 10px 15px;               /* smaller padding top/bottom & left/right */
    border-radius: 8px;               /* slightly smaller corners */
    text-align: center;
    color: white;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.4); /* slightly lighter shadow */
    transition: transform 0.15s;      /* faster, subtle hover */
    margin-bottom: 8px;               /* spacing below cards */
}

.metric-card:hover {
    transform: scale(1.03);           /* small hover effect */
}

.metric-number {
    font-size: 22px;                  /* smaller number size */
    font-weight: 700;
    color: #7FDBFF;                   /* highlight color */
    margin-bottom: 4px;               /* space between number and label */
}

.metric-label {
    font-size: 12px;                  /* smaller label */
    color: #AAAAAA;
    font-weight: 500;
}

/* optional: make metrics in columns evenly spaced */
.metric-column {
    display: flex;
    flex-direction: column;
    gap: 8px;  /* space between stacked cards if needed */
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
    # Setup tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Entire Dataset",
        "Filtered by Down/Yardline",
        "Success Heatmaps",
        "Concept Effectiveness",
        "Play Prediction"
    ])

      # -------------------------
    # Tab 1 - whole dataset
    # -------------------------
    with tab1:
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
        concept_fig_all = px.bar(top_concepts_all, x="count", y="concept", color="play_direction", orientation="h", title="Most Frequent Concepts by Play Direction", template="plotly_dark", color_discrete_sequence=["#7FDBFF","#0A2342","#AAAAAA"])

        play_type_summary_all = df["play_type"].value_counts().reset_index()
        play_type_summary_all.columns = ["play_type","count"]
        run_pass_fig_all = px.pie(play_type_summary_all, names="play_type", values="count", title="Run vs Pass %", color="play_type", color_discrete_map={"Run":"#0A2342","Pass":"#7FDBFF"}, template="plotly_dark")

        concept_summary_all = df["concept"].value_counts().head(6).reset_index()
        concept_summary_all.columns = ["concept","count"]
        concept_pie_fig_all = px.pie(concept_summary_all, names="concept", values="count", title="Most Frequent Concepts", color_discrete_sequence=px.colors.sequential.Blues, template="plotly_dark")

        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig_all, use_container_width=True)
        r1c2.plotly_chart(concept_fig_all, use_container_width=True)

        st.markdown('<div class="section-header">Run/Pass & Concept Distribution</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig_all, use_container_width=True)
        r2c2.plotly_chart(concept_pie_fig_all, use_container_width=True)

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
   
           # Custom hover
           fig.update_traces(
               hovertemplate="<b>Down:</b> %{y}<br>"
                             "<b>Yard Group:</b> %{x}<br>"
                             f"<b>{title}:</b> %{z:.1f}%<br>"
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
    # --------------
    # Tab 3 - heatmap
    # --------------

    with tab3:

        st.markdown("### Play Success Heatmap")

        with tab3:
            st.markdown("### Play Success Heatmap")

            with st.expander("How to read this chart"):
                st.write("""
                This heatmap shows the **success rate of plays by down and field position**.

                - Darker blue = higher success rate
                - Lighter blue = lower success rate
                - Success is defined as gaining **4 or more yards on a play**

                Coaches can use this to identify **field zones where the offense is most efficient**.
                """)

        import plotly.express as px

        # Make sure 'success' column exists
        df["success"] = df["gain_loss"] >= 4

        # Aggregate by down and yard_group
        heatmap_df = df.groupby(["down", "yard_group"]).agg(
            success_rate=("success", "mean"),  # fraction of successful plays
            num_plays=("success", "count")  # total plays in that bin
        ).reset_index()

        # Create heatmap with hover showing both metrics
        heatmap_fig = px.imshow(
            heatmap_df.pivot(index="down", columns="yard_group", values="success_rate"),
            text_auto=True,
            aspect="auto",
            labels=dict(x="Yard Group", y="Down", color="Success Rate"),
            color_continuous_scale="Blues",
        )

        # Add custom hover
        heatmap_fig.update_traces(
            hovertemplate="<b>Down:</b> %{y}<br>"
                          "<b>Yard Group:</b> %{x}<br>"
                          "<b>Success Rate:</b> %{z:.0%}<br>"
                          "<b>Number of Plays:</b> %{customdata}",
            customdata=heatmap_df.pivot(index="down", columns="yard_group", values="num_plays").values
        )

        # Show in Streamlit
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # --------------
    # Tab 4 - Concept Effectiveness
    # --------------
        with tab4:
            st.markdown("### Concept Effectiveness")

            with st.expander("How to read this chart"):
                st.write("""
                This chart evaluates offensive concepts using three metrics:

                **X-axis:** Success Rate (% of plays gaining 4+ yards)

                **Y-axis:** Average yards gained per play

                **Bubble Size:** Number of times the concept was run

                The best concepts arre large and towards the right:
                • High success rate
                • High average gain
                • Large sample size
                """)

            concept_stats = (
                df.groupby("concept")
                .agg(
                    avg_gain=("gain_loss", "mean"),
                    success_rate=("gain_loss", lambda x: (x >= 4).mean()),
                    plays=("gain_loss", "count")
                )
                .reset_index()
            )

            concept_stats = concept_stats.sort_values("avg_gain", ascending=False)

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

            st.dataframe(concept_stats)
    # --------------
    # Tab 5
    # --------------

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    with tab5:

        st.markdown("### Run / Pass Prediction")

        model_df = df.dropna(subset=["down", "distance", "yardline", "play_type"])

        le = LabelEncoder()
        model_df["play_type_encoded"] = le.fit_transform(model_df["play_type"])

        X = model_df[["down", "distance", "yardline"]]
        y = model_df["play_type_encoded"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        st.markdown("### Predict Play Type")

        pred_down = st.selectbox("Down", sorted(df["down"].unique()))
        pred_dist = st.number_input("Distance", 1, 20, 10)
        pred_yard = st.slider("Yardline", -50, 50, 0)

        prediction = model.predict([[pred_down, pred_dist, pred_yard]])
        predicted_play = le.inverse_transform(prediction)[0]

        st.metric("Predicted Play Type", predicted_play)





