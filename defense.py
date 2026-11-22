from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Har-Ber Defensive Analytics",
    page_icon="🏈",
    layout="wide",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --st-text-color: #ffffff;
    }

    .stApp {
        background-color: #07182d;
        color: #ffffff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061426, #0a2342);
    }

    .stApp p,
    .stApp label,
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] small,
    .stApp [data-testid="stFileUploaderFile"] small {
        color: #ffffff !important;
    }

    h1, h2, h3 {
        color: #7FDBFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    data = pd.read_excel(
        BytesIO(file_bytes),
        dtype="string",
    )

    data.columns = (
        data.columns.str.strip()
        .str.upper()
        .str.replace(r"[\s_]+", " ", regex=True)
        .str.replace(r"\s*/\s*", " / ", regex=True)
    )

    aliases = {
        "DEFENSIVE FRONT": "DEF FRONT",
        "DEFENSIVE FRONTS": "DEF FRONT",
        "DOWN": "DN",
        "FORMATION": "OFF FORM",
        "OFF FORMATION": "OFF FORM",
        "DEFENSIVE LINE ALIGNMENT": "DL ALIGNMENT",
        "STUNT/GAME": "STUNT / GAME",
    }

    for original, standard in aliases.items():
        if original in data.columns and standard not in data.columns:
            data = data.rename(columns={original: standard})

    return data


def clean_breakdown_values(values):
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"": pd.NA, "?": pd.NA})
    )


# ---------------------------------------------------------
# Custom grouping
# ---------------------------------------------------------
def apply_custom_groups(field, values, full_values):
    state_key = f"defense_groups_{field}"
    groups = st.session_state.setdefault(state_key, {})
    next_id_key = f"defense_next_group_id_{field}"

    original_labels = set(
        clean_breakdown_values(full_values)
        .dropna()
        .unique()
        .tolist()
    )

    used_labels = {
        member
        for group in groups.values()
        for member in group["members"]
    }

    available_labels = original_labels - used_labels

    # Keep explicit no-blitz entries separate from blitz calls.
    if field == "BLITZ":
        available_labels.discard("NO")

    grouped = values.copy()

    with st.expander("Custom Grouping"):
        if field in ["DEF FRONT", "DL ALIGNMENT"]:
            st.caption(
                "Select labels to combine, then give the group a name. "
                "Turn a group off to show its original labels."
            )

        notice_key = f"defense_group_notice_{field}"
        notice = st.session_state.pop(notice_key, None)

        if notice:
            st.success(notice)

        with st.form(
            f"defense_group_form_{field}",
            clear_on_submit=True,
        ):
            members = st.multiselect(
                "Labels to combine",
                sorted(available_labels),
            )

            name = " ".join(
                st.text_input("Group name").split()
            )

            submitted = st.form_submit_button("Add group")

        if submitted:
            name_key = name.upper()

            if len(members) < 2:
                st.warning("Select at least two labels.")

            elif not name:
                st.warning("Enter a group name.")

            elif (
                name_key in original_labels
                or name_key in {"NO", "?"}
            ):
                st.warning(
                    "Choose a name different from the original labels."
                )

            elif any(
                group["name"].upper() == name_key
                for group in groups.values()
            ):
                st.warning(
                    "A group with that name already exists."
                )

            else:
                group_id = st.session_state.get(
                    next_id_key, 1
                )

                groups[group_id] = {
                    "name": name,
                    "members": members,
                }

                st.session_state[next_id_key] = group_id + 1
                st.session_state[notice_key] = f"Added {name}."
                st.rerun()

        for group_id, group in list(groups.items()):
            toggle_col, details_col, remove_col = st.columns(
                [1, 4, 1]
            )

            enabled = toggle_col.toggle(
                "On",
                value=True,
                key=f"defense_group_enabled_{field}_{group_id}",
            )

            details_col.write(group["name"])
            details_col.caption(", ".join(group["members"]))

            if remove_col.button(
                "Remove",
                key=f"defense_group_remove_{field}_{group_id}",
            ):
                del groups[group_id]
                st.rerun()

            if enabled:
                grouped.loc[
                    values.isin(group["members"])
                ] = group["name"]

    return grouped


# ---------------------------------------------------------
# Breakdown chart and table
# ---------------------------------------------------------
def render_breakdown(
    data,
    full_data,
    field,
    label,
    group_fronts_by_number=False,
):
    st.subheader(f"{label} Breakdown")

    if field not in data.columns:
        st.info(
            f"This file does not contain a {field} column."
        )
        return

    values = clean_breakdown_values(data[field])
    original_charted = values.dropna()

    grouped = apply_custom_groups(
        field,
        values,
        full_data[field],
    )

    # Apply first-number grouping to fronts AND DL alignment.
    # Automatic grouping takes priority over manual groups.
    if (
        field in ["DEF FRONT", "DL ALIGNMENT"]
        and group_fronts_by_number
    ):
        number_names = {
            "0": "Zero",
            "1": "One",
            "2": "Two",
            "3": "Odd",
            "4": "Even",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }

        first_number = values.str.extract(
            r"^(\d+)",
            expand=False,
        )

        number_group = first_number.map(number_names)

        grouped = grouped.mask(
            number_group.notna(),
            number_group,
        )

        st.caption(
            "Automatic grouping is on: "
            "3 → Odd, 4 → Even, 5 → Five, etc. "
            "Labels without a recognized starting number "
            "keep their original or custom group."
        )

    charted = grouped.dropna()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Charted Plays",
        f"{len(original_charted):,}",
    )

    c2.metric(
        "Uncharted Plays",
        f"{values.isna().sum():,}",
    )

    if field == "BLITZ":
        blitzes = int(
            original_charted.ne("NO").sum()
        )

        rate = (
            f"{blitzes / len(original_charted):.1%}"
            if not original_charted.empty
            else "N/A"
        )

        c3.metric("Blitz Rate", rate)

    else:
        c3.metric(
            "Categories Shown",
            charted.nunique(),
        )

    if charted.empty:
        st.info("No charted plays for this selection.")
        return

    summary = (
        charted.value_counts()
        .rename_axis(label)
        .reset_index(name="Plays")
    )

    summary["Share %"] = (
        summary["Plays"] / len(original_charted) * 100
    )

    fig = px.bar(
        summary.sort_values("Plays"),
        x="Plays",
        y=label,
        orientation="h",
        custom_data=["Share %"],
        color_discrete_sequence=["#7FDBFF"],
        template="plotly_dark",
    )

    fig.update_yaxes(type="category")

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
        font_color="#ffffff",
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
            "Plays": st.column_config.NumberColumn(
                "Plays",
                format="%d",
            ),
            "Share %": st.column_config.NumberColumn(
                "Share %",
                format="%.1f%%",
            ),
        },
    )

    if field == "BLITZ":
        st.caption(
            "NO means no blitz. Other charted labels count "
            "as blitz calls. Blank entries and ? are excluded "
            "from the rate and percentages."
        )

    else:
        st.caption(
            "Percentages are based on charted plays. "
            "Blank entries and ? are excluded."
        )


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
st.title("Har-Ber Defensive Analytics")

uploaded_file = st.sidebar.file_uploader(
    "Upload Hudl Excel File",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info(
        "Upload a Hudl Excel file from the sidebar to get started."
    )
    st.stop()

try:
    full_df = load_data(uploaded_file.getvalue())

except Exception as error:
    st.error(f"Could not read the Excel file: {error}")
    st.stop()

df = full_df.copy()


# ---------------------------------------------------------
# Sidebar settings and filters
# ---------------------------------------------------------
st.sidebar.header("Settings")

group_fronts_by_number = st.sidebar.toggle(
    "Group fronts and DL alignment by first number",
    value=False,
    help=(
        "Applies to Defensive Front and DL Alignment. "
        "3 → Odd, 4 → Even, 5 → Five, 6 → Six, etc."
    ),
)

st.sidebar.header("Filters")

for field, label in [
    ("DN", "Down"),
    ("PLAY TYPE", "Play Type"),
    ("OFF FORM", "Formation"),
]:
    if field not in df.columns:
        continue

    values = clean_breakdown_values(df[field])

    options = [None] + sorted(
        values.dropna().unique().tolist()
    )

    filter_key = f"defense_filter_{field}"

    # Reset unavailable selections after another filter changes
    # or a different file is uploaded.
    if (
        filter_key in st.session_state
        and st.session_state[filter_key] not in options
    ):
        st.session_state.pop(filter_key, None)

    selected = st.sidebar.selectbox(
        label,
        options,
        format_func=lambda value: (
            "All" if value is None else value
        ),
        key=filter_key,
    )

    if selected is not None:
        df = df.loc[
            values.eq(selected).fillna(False)
        ].copy()

st.caption(
    f"{len(df):,} plays in the current selection"
)


# ---------------------------------------------------------
# Six defensive tabs
# ---------------------------------------------------------
tab_fields = [
    ("DEF FRONT", "Defensive Front"),
    ("DL ALIGNMENT", "DL Alignment"),
    ("STUNT / GAME", "Stunt / Game"),
    ("BLITZ", "Blitz"),
    ("PRESSURE", "Pressure"),
    ("COVERAGE", "Coverage"),
]

tabs = st.tabs([
    label for field, label in tab_fields
])

for tab, (field, label) in zip(tabs, tab_fields):
    with tab:
        render_breakdown(
            df,
            full_df,
            field,
            label,
            group_fronts_by_number=group_fronts_by_number,
        )
