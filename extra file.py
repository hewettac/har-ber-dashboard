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
