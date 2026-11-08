
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Har-Ber Defensive Analytics",
    page_icon="🏈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #07182d;
        color: #f5fbff;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061426, #0a2342);
    }
    h1, h2, h3 {
        color: #7FDBFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Har-Ber Defensive Analytics")

uploaded_file = st.sidebar.file_uploader(
    "Upload Hudl Excel File",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("Upload a Hudl Excel file from the sidebar to get started.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    from io import BytesIO

    data = pd.read_excel(BytesIO(file_bytes), dtype="string")
    data.columns = (
        data.columns.str.strip()
        .str.upper()
        .str.replace(r"[\s_]+", " ", regex=True)
    )
    return data


try:
    df = load_data(uploaded_file.getvalue())
except Exception as error:
    st.error(f"Could not read the Excel file: {error}")
    st.stop()

# Accept both the Hudl header and the full column name.
if "DEF FRONT" not in df.columns and "DEFENSIVE FRONT" in df.columns:
    df = df.rename(columns={"DEFENSIVE FRONT": "DEF FRONT"})

st.sidebar.header("Filters")

for field, label in [("DN", "Down"), ("PLAY TYPE", "Play Type")]:
    if field in df.columns:
        values = df[field].str.strip().str.upper().replace("", pd.NA)
        options = [None] + sorted(values.dropna().unique().tolist())

        selected = st.sidebar.selectbox(
            label,
            options,
            format_func=lambda value: "All" if value is None else value,
        )

        if selected is not None:
            df = df.loc[values.eq(selected).fillna(False)]

st.caption(f"{len(df):,} plays in the current selection")


def render_breakdown(field, label):
    st.subheader(f"{label} Breakdown")

    if field not in df.columns:
        st.info(f"This file does not contain a {field} column.")
        return

    values = (
        df[field].str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "?": pd.NA})
    )
    charted = values.dropna()

    c1, c2, c3 = st.columns(3)
    c1.metric("Charted Plays", f"{len(charted):,}")
    c2.metric("Uncharted Plays", f"{values.isna().sum():,}")

    if field == "BLITZ":
        blitzes = int(charted.ne("NO").sum())
        rate = f"{blitzes / len(charted):.1%}" if len(charted) else "N/A"
        c3.metric("Blitz Rate", rate)
    else:
        c3.metric(f"Unique {label}s", charted.nunique())

    if charted.empty:
        st.info("No charted plays for this selection.")
        return

    summary = (
        charted.value_counts()
        .rename_axis(label)
        .reset_index(name="Plays")
    )
    summary["Share %"] = summary["Plays"] / len(charted) * 100

    chart_data = summary.sort_values("Plays")
    fig = px.bar(
        chart_data,
        x="Plays",
        y=label,
        orientation="h",
        custom_data=["Share %"],
        color_discrete_sequence=["#7FDBFF"],
        template="plotly_dark",
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Plays: %{x}<br>"
            "Share: %{customdata[0]:.1f}%"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        height=max(350, 26 * len(summary) + 70),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title=None,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        key=f"defense_chart_{field}",
    )

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share %": st.column_config.NumberColumn(
                "Share %", format="%.1f%%"
            ),
        },
    )

    if field == "BLITZ":
        st.caption(
            "NO means no blitz. Other charted labels are counted "
            "as blitz calls. Blank entries and ? are excluded "
            "from the rate and percentages."
        )
    else:
        st.caption(
            "Percentages are based on charted plays. "
            "Blank entries and ? are excluded."
        )


front_tab, blitz_tab, coverage_tab = st.tabs(
    ["Defensive Front", "Blitz", "Coverage"]
)

with front_tab:
    render_breakdown("DEF FRONT", "Defensive Front")

with blitz_tab:
    render_breakdown("BLITZ", "Blitz")

with coverage_tab:
    render_breakdown("COVERAGE", "Coverage")
