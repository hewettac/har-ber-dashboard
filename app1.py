import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Har‑Ber Basic Analytics", layout="wide")

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
st.sidebar.title("Har‑Ber Basic Dashboard")
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
        "Overall Snapshot",
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
    # Tab 2
    # -------------------------
    with tab2:
        # Metrics
        avg_gain = round(selected["gain_loss"].mean(), 1)
        max_gain = selected["gain_loss"].max()
        min_gain = selected["gain_loss"].min()
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="metric-card"><div class="metric-number">{avg_gain}</div><div class="metric-label">Average Gain</div></div>',
            unsafe_allow_html=True)
        c2.markdown(
            f'<div class="metric-card"><div class="metric-number">{max_gain}</div><div class="metric-label">Max Gain</div></div>',
            unsafe_allow_html=True)
        c3.markdown(
            f'<div class="metric-card"><div class="metric-number">{min_gain}</div><div class="metric-label">Min Gain</div></div>',
            unsafe_allow_html=True)

        # Gain/Loss Distribution
        gain_summary = selected.groupby("gain_loss").size().reset_index(name="plays").sort_values("gain_loss")
        gain_fig = px.bar(gain_summary, x="gain_loss", y="plays",
                          labels={"gain_loss": "Yards Gained", "plays": "Number of Plays"},
                          title="Gain / Loss Distribution", template="plotly_dark", color_discrete_sequence=["#7FDBFF"])

        # Most ran concepts
        top_concepts = selected.groupby(["concept", "play_direction"]).size().reset_index(name="count").sort_values(
            "count", ascending=False).head(8)
        concept_fig = px.bar(top_concepts, x="count", y="concept", color="play_direction", orientation="h",
                             title="Most Frequent Concepts by Play Direction", template="plotly_dark",
                             color_discrete_sequence=["#7FDBFF", "#0A2342", "#AAAAAA"])

        # Run/Pass Pie
        play_type_summary = selected["play_type"].value_counts().reset_index()
        play_type_summary.columns = ["play_type", "count"]
        run_pass_fig = px.pie(play_type_summary, names="play_type", values="count", title="Run vs Pass %",
                              color="play_type", color_discrete_map={"Run": "#0A2342", "Pass": "#7FDBFF"},
                              template="plotly_dark")

        # Concept Pie (Top 6)
        concept_summary = selected["concept"].value_counts().head(6).reset_index()
        concept_summary.columns = ["concept", "count"]
        concept_pie_fig = px.pie(concept_summary, names="concept", values="count", title="Most Frequent Concepts",
                                 color_discrete_sequence=px.colors.sequential.Blues, template="plotly_dark")

        # Layout Charts
        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(gain_fig, use_container_width=True)
        r1c2.plotly_chart(concept_fig, use_container_width=True)

        st.markdown('<div class="section-header">Run/Pass & Concept Distribution</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(run_pass_fig, use_container_width=True)
        r2c2.plotly_chart(concept_pie_fig, use_container_width=True)

        st.markdown('<div class="section-header">Raw Play Data</div>', unsafe_allow_html=True)
        st.dataframe(selected, use_container_width=True)

    # --------------
    # Tab 3 - heatmap
    # --------------

    with tab3:
        with tab3:
            st.markdown("### Play Success Heatmap")

            with st.expander("How to read this chart"):
                st.write("""
                This heatmap shows the **success rate of plays by down and field position**.

                - Success: Plays gaining ≥ 4 yards
                - Darker blue = higher success rate
                - Lighter blue = lower success rate
                - Success is defined as gaining **4 or more yards on a play**
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

                The best concepts are large and towards the right:
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
                template="plotly_dark"
            )

            st.plotly_chart(bubble, use_container_width=True)

            st.dataframe(concept_stats)
    # --------------
    # Tab 5
    # --------------

    # --------------
    # Tab 5 (ELITE Play Prediction System)
    # --------------

   
    # -------------------------
    # Feature Engineering
    # -------------------------
    def add_features(df):
        df = df.copy()
   
        df["distance_bucket"] = pd.cut(
            df["distance"],
            bins=[0, 3, 7, 20],
            labels=[0, 1, 2]
        ).astype(float)
   
        df["field_zone"] = pd.cut(
            df["yardline"],
            bins=[-50, -20, 20, 50],
            labels=[0, 1, 2]  # backed up, midfield, redzone-ish
        ).astype(float)
   
        return df
   
   # -------------------------
   # Load Base Dataset (cached)
   # -------------------------
    @st.cache_data
    def load_base_data():
        return pd.read_csv("AllPlaysTrainData.csv")
   
   # -------------------------
   # Train Model (cached)
   # -------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
   
    @st.cache_resource
    def train_model(base_df, weekly_df):
   
        base_df = add_features(base_df)
        weekly_df = add_features(weekly_df)
   
        base_df = base_df.dropna(subset=["down", "distance", "yardline", "play_type"])
        weekly_df = weekly_df.dropna(subset=["down", "distance", "yardline", "play_type"])
   
        base_df["weight"] = 1
        weekly_df["weight"] = 6
   
        combined = pd.concat([base_df, weekly_df])
   
        le = LabelEncoder()
        combined["play_type_encoded"] = le.fit_transform(combined["play_type"])
   
        features = ["down", "distance", "yardline", "distance_bucket", "field_zone"]
   
        X = combined[features]
        y = combined["play_type_encoded"]
        weights = combined["weight"]
   
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
   
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )
   
        model.fit(X_train, y_train, sample_weight=w_train)
   
       # 🔥 Overall Accuracy
        y_pred = model.predict(X_test)
        overall_accuracy = accuracy_score(y_test, y_pred)
   
       # Save test set for situational accuracy later
        return model, le, overall_accuracy, X_test, y_test
   
   # -------------------------
   # TAB UI
   # -------------------------
        with tab5:
   
            st.markdown("## 🧠 Elite Play Prediction Engine")
   
       # Load datasets
            try:
                base_df = load_base_data()
            except:
                st.error("Missing AllPlaysTrainData.csv")
                st.stop()
   
            weekly_df = df.copy()
   
       # Train model (cached = FAST)
            model, le = train_model(base_df, weekly_df)
   
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Play", predicted_play)
            
            with col2:
                st.metric("Confidence", f"{top_confidence:.1f}%")
            
            with col3:
                st.metric("Model Accuracy", f"{overall_accuracy*100:.1f}%")
   
          # -------------------------
          # Prediction
          # -------------------------
            probs = model.predict_proba(input_df[features])[0]
      
            top_indices = np.argsort(probs)[::-1][:3]
      
            st.markdown("### 🎯 Top Play Predictions")
      
            for i in top_indices:
              play = le.inverse_transform([i])[0]
              confidence = probs[i] * 100
      
              st.metric(play, f"{confidence:.1f}%")

            # -------------------------
            # Prediction
            # -------------------------
            pred_class = np.argmax(probs)
            predicted_play = le.inverse_transform([pred_class])[0]
            top_confidence = probs[pred_class] * 100

                       # -------------------------
            # Situation-Specific Accuracy
            # -------------------------
            # Find similar plays in test set
            mask = (
                (X_test["down"] == pred_down) &
                (abs(X_test["distance"] - pred_dist) <= 2) &
                (abs(X_test["yardline"] - pred_yard) <= 10)
            )
            
            similar_X = X_test[mask]
            similar_y = y_test[mask]
            
            if len(similar_X) > 20:
                similar_preds = model.predict(similar_X)
                situation_accuracy = accuracy_score(similar_y, similar_preds)
            else:
                situation_accuracy = None
      
          # -------------------------
          # Situation Insight
          # -------------------------
            st.markdown("### 📊 Situation Insight")
      
            if pred_down == 3 and pred_dist >= 7:
               st.info("Likely PASS situation (3rd & long tendency)")
            elif pred_down == 1 and pred_dist <= 3:
               st.info("High RUN probability (short yardage)")
