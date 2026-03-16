    # Tab 3 - heatmap
    # --------------

    with tab3:

        st.markdown("### Play Success Heatmap")

        with tab3:
      with tab3:
            st.markdown("### Play Success Heatmap")

            with st.expander("How to read this chart"):
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
