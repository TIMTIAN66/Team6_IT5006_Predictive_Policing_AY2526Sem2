import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
from typing import List, Tuple

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(
    page_title="Chicago Crime Interactive Dashboard",
    layout="wide",
)

DATASETS = {
    "All (2015-2025)": None,
    "Train (2015-2024)": "https://it5006group6.s3.ap-southeast-2.amazonaws.com/crime_train_2015_2024.csv",
    "Test/Val (2025)": "https://it5006group6.s3.ap-southeast-2.amazonaws.com/crime_test_val_2025.csv",
}

USE_COLS = [
    "Date",
    "Time",
    "Primary Type",
    "Description",
    "Location Description",
    "Arrest",
    "Domestic",
    "District",
    "Community Area",
    "Year",
    "Latitude",
    "Longitude",
    "Month",
    "DayOfWeek",
    "Hour",
]

DTYPE = {
    "Primary Type": "category",
    "Description": "category",
    "Location Description": "category",
    "DayOfWeek": "category",
    "Arrest": "bool",
    "Domestic": "bool",
    "Year": "Int16",
    "Month": "Int8",
    "Hour": "Int8",
    "District": "Int16",
    "Community Area": "Int16",
    "Latitude": "float32",
    "Longitude": "float32",
}

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

MONTH_ORDER = list(range(1, 13))
HOUR_ORDER = list(range(24))

sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=USE_COLS, dtype=DTYPE, low_memory=False)

    # Data is already cleaned and feature-engineered in the provided CSVs.
    return df


@st.cache_data(show_spinner=False)
def load_dataset(choice: str) -> pd.DataFrame:
    if choice == "All (2015-2025)":
        train = load_csv(DATASETS["Train (2015-2024)"])
        test = load_csv(DATASETS["Test/Val (2025)"])
        return pd.concat([train, test], ignore_index=True)
    return load_csv(DATASETS[choice])


def numeric_xy(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(series.index, errors="coerce")
    y = pd.to_numeric(series.values, errors="coerce")
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def format_number(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if abs(value - int(value)) < 1e-6:
        return f"{int(value):,}"
    return f"{value:,.2f}"


def find_outliers(y: np.ndarray, z_threshold: float = 2.0, max_points: int = 3) -> List[int]:
    if len(y) < 3:
        return []
    mean = float(np.mean(y))
    std = float(np.std(y))
    if std == 0:
        return []
    zscores = (y - mean) / std
    idxs = np.where(np.abs(zscores) >= z_threshold)[0]
    if len(idxs) > max_points:
        order = np.argsort(np.abs(zscores[idxs]))[::-1]
        idxs = idxs[order[:max_points]]
    return idxs.tolist()


def plot_line(
    x,
    y,
    title: str,
    x_label: str,
    y_label: str,
    xticks=None,
    annotate_peak: bool = True,
    annotate_outliers: bool = True,
    outlier_z: float = 2.0,
):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines+markers", line={"width": 3}, marker={"size": 6})
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        if xticks is not None:
            fig.update_xaxes(tickmode="array", tickvals=list(xticks))

        if len(y) > 0 and annotate_peak:
            peak_idx = int(np.argmax(y))
            fig.add_trace(
                go.Scatter(
                    x=[x[peak_idx]],
                    y=[y[peak_idx]],
                    mode="markers+text",
                    text=[f"Peak: {format_number(y[peak_idx])}"],
                    textposition="top center",
                    marker=dict(color="crimson", size=10, symbol="star"),
                    name="Peak",
                    showlegend=False,
                )
            )

        if annotate_outliers:
            outlier_idxs = find_outliers(np.asarray(y), z_threshold=outlier_z)
            if outlier_idxs:
                fig.add_trace(
                    go.Scatter(
                        x=[x[i] for i in outlier_idxs],
                        y=[y[i] for i in outlier_idxs],
                        mode="markers+text",
                        text=[f"Outlier: {format_number(y[i])}" for i in outlier_idxs],
                        textposition="top center",
                        marker=dict(color="#E67E22", size=9, symbol="diamond"),
                        name="Outlier",
                        showlegend=False,
                    )
                )

        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xticks is not None:
            ax.set_xticks(list(xticks))

        if len(y) > 0 and annotate_peak:
            peak_idx = int(np.argmax(y))
            ax.scatter([x[peak_idx]], [y[peak_idx]], color="crimson", zorder=5)
            ax.annotate(
                f"Peak: {format_number(y[peak_idx])}",
                xy=(x[peak_idx], y[peak_idx]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color="crimson",
            )

        if annotate_outliers:
            outlier_idxs = find_outliers(np.asarray(y), z_threshold=outlier_z)
            for i in outlier_idxs:
                ax.scatter([x[i]], [y[i]], color="#E67E22", zorder=5)
                ax.annotate(
                    f"Outlier: {format_number(y[i])}",
                    xy=(x[i], y[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    color="#E67E22",
                )

        plt.tight_layout()
        st.pyplot(fig)


def plot_multi_line(
    series_dict: dict,
    title: str,
    x_label: str,
    y_label: str,
    xticks=None,
    annotate_peak: bool = True,
):
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for name, (x, y) in series_dict.items():
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=str(name)))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=10, r=10, t=40, b=10),
            height=360,
        )
        if xticks is not None:
            fig.update_xaxes(tickmode="array", tickvals=list(xticks))

        if annotate_peak:
            for name, (x, y) in series_dict.items():
                if len(y) == 0:
                    continue
                peak_idx = int(np.argmax(y))
                fig.add_trace(
                    go.Scatter(
                        x=[x[peak_idx]],
                        y=[y[peak_idx]],
                        mode="markers+text",
                        text=[f"{name} Peak: {format_number(y[peak_idx])}"],
                        textposition="top center",
                        marker=dict(size=9, symbol="star"),
                        showlegend=False,
                    )
                )

        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, (x, y) in series_dict.items():
            ax.plot(x, y, marker="o", label=str(name))
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xticks is not None:
            ax.set_xticks(list(xticks))
        ax.legend()

        if annotate_peak:
            for name, (x, y) in series_dict.items():
                if len(y) == 0:
                    continue
                peak_idx = int(np.argmax(y))
                ax.scatter([x[peak_idx]], [y[peak_idx]], zorder=5)
                ax.annotate(
                    f"{name} Peak: {format_number(y[peak_idx])}",
                    xy=(x[peak_idx], y[peak_idx]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

        plt.tight_layout()
        st.pyplot(fig)


def plot_bar(x, y, title: str, x_label: str, y_label: str, orientation: str = "v"):
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=x if orientation == "v" else y,
            y=y if orientation == "v" else x,
            orientation=orientation,
        )
        if orientation == "v":
            fig.update_xaxes(categoryorder="array", categoryarray=list(x))
        else:
            fig.update_yaxes(categoryorder="array", categoryarray=list(x))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=10, r=10, t=40, b=10),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        if orientation == "v":
            ax.bar(x, y)
        else:
            ax.barh(x, y)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        st.pyplot(fig)


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    colorscale: str,
    xticks=None,
    yticks=None,
):
    if PLOTLY_AVAILABLE:
        fig = px.imshow(
            df,
            aspect="auto",
            color_continuous_scale=colorscale,
            labels={"x": x_label, "y": y_label, "color": "Count"},
        )
        if xticks is not None:
            fig.update_xaxes(tickmode="array", tickvals=list(xticks))
        if yticks is not None:
            fig.update_yaxes(tickmode="array", tickvals=list(yticks))
        fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df, cmap=colorscale, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if xticks is not None:
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_xticklabels(list(df.columns))
        if yticks is not None:
            ax.set_yticks(np.arange(len(df.index)))
            ax.set_yticklabels(list(df.index))
        plt.tight_layout()
        st.pyplot(fig)


def apply_filters(
    df: pd.DataFrame,
    year_range: Tuple[int, int],
    districts: List[int],
    crime_types: List[str],
    arrest_filter: str,
) -> pd.DataFrame:
    filtered = df
    if "Year" in filtered.columns:
        filtered = filtered[
            (filtered["Year"] >= year_range[0]) & (filtered["Year"] <= year_range[1])
        ]
    if districts:
        filtered = filtered[filtered["District"].isin(districts)]
    if crime_types:
        filtered = filtered[filtered["Primary Type"].isin(crime_types)]
    if arrest_filter == "Arrested Only":
        filtered = filtered[filtered["Arrest"] == True]
    elif arrest_filter == "Not Arrested Only":
        filtered = filtered[filtered["Arrest"] == False]
    return filtered


st.title("Chicago Crime Interactive Dashboard (2015-2025)")
st.caption(
    "Built from the EDA workflow in your PDF. Use the filters to explore temporal patterns, spatial distributions, and crime correlations."
)
if not PLOTLY_AVAILABLE:
    st.info(
        "Plotly is not installed in this environment. Charts will use a static fallback. "
        "Install plotly to unlock hover/zoom interactions."
    )

with st.sidebar:
    st.header("Filters")
    dataset_choice = st.radio("Dataset", list(DATASETS.keys()), index=0)
    df = load_dataset(dataset_choice)

    year_min = int(np.nanmin(df["Year"]))
    year_max = int(np.nanmax(df["Year"]))
    year_range = st.slider(
        "Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
    )

    district_options = (
        pd.Series(df["District"].dropna().unique())
        .sort_values()
        .astype(int)
        .tolist()
    )
    district_choice = st.multiselect("District", district_options, key="district_choice")

    crime_type_options = (
        pd.Series(df["Primary Type"].dropna().unique()).sort_values().tolist()
    )
    crime_type_choice = st.multiselect(
        "Primary Type", crime_type_options, key="crime_type_choice"
    )

    arrest_filter = st.selectbox(
        "Arrest Filter",
        ["All", "Arrested Only", "Not Arrested Only"],
        index=0,
    )

    top_n = st.slider("Top N (Ranked Lists)", min_value=5, max_value=30, value=15, step=1)

    compare_mode = st.checkbox("Comparison Mode", value=False)
    compare_dim = None
    compare_values = []
    compare_view = "Yearly"
    if compare_mode:
        compare_dim = st.radio("Compare By", ["District", "Primary Type"], horizontal=True)
        compare_view = st.radio(
            "Compare View", ["Yearly", "Monthly", "Hourly"], horizontal=True
        )
        if compare_dim == "District":
            compare_values = st.multiselect(
                "Select up to 2 districts", district_options, key="compare_districts"
            )
        else:
            compare_values = st.multiselect(
                "Select up to 2 crime types", crime_type_options, key="compare_types"
            )
        if len(compare_values) > 2:
            st.warning("Please select no more than 2 items for comparison.")
            compare_values = compare_values[:2]

    with st.expander("Quick Focus (Linked Filters)"):
        focus_district = st.selectbox(
            "Focus District",
            ["All"] + district_options,
            index=0,
            key="focus_district",
        )
        focus_crime = st.selectbox(
            "Focus Crime Type",
            ["All"] + crime_type_options,
            index=0,
            key="focus_crime",
        )
        if st.button("Apply Focus Filters"):
            st.session_state["district_choice"] = [] if focus_district == "All" else [focus_district]
            st.session_state["crime_type_choice"] = [] if focus_crime == "All" else [focus_crime]
            st.experimental_rerun()

    map_sample_size = st.slider(
        "Map Sample Size",
        min_value=1000,
        max_value=100000,
        value=50000,
        step=1000,
    )

filtered_df = apply_filters(
    df,
    year_range=year_range,
    districts=district_choice,
    crime_types=crime_type_choice,
    arrest_filter=arrest_filter,
)

if filtered_df.empty:
    st.warning("No records match the current filters. Please adjust and try again.")
    st.stop()

yearly_counts = filtered_df.groupby("Year").size().sort_index()
monthly_counts = filtered_df.groupby("Month").size().reindex(MONTH_ORDER, fill_value=0)
hourly_counts = filtered_df.groupby("Hour").size().reindex(HOUR_ORDER, fill_value=0)
day_counts = filtered_df.groupby("DayOfWeek").size().reindex(DAY_ORDER, fill_value=0)

latest_year = int(yearly_counts.index.max())
latest_year_count = int(yearly_counts.loc[latest_year]) if latest_year in yearly_counts else 0
prev_year_count = (
    int(yearly_counts.loc[latest_year - 1])
    if (latest_year - 1) in yearly_counts
    else None
)
if prev_year_count:
    yoy_delta = latest_year_count - prev_year_count
    yoy_delta_pct = (yoy_delta / prev_year_count) * 100
else:
    yoy_delta = None
    yoy_delta_pct = None

top_crime_type = filtered_df["Primary Type"].value_counts().idxmax()
peak_hour = int(hourly_counts.idxmax()) if hourly_counts.sum() > 0 else None

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Incidents", f"{len(filtered_df):,}")
with col2:
    st.metric("Unique Crime Types", f"{filtered_df['Primary Type'].nunique()}")
with col3:
    arrest_rate = filtered_df["Arrest"].mean() * 100
    st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
with col4:
    st.metric("Districts", f"{filtered_df['District'].nunique()}")

col5, col6, col7, col8 = st.columns(4)
with col5:
    if yoy_delta is not None:
        st.metric(
            f"Latest Year ({latest_year})",
            f"{latest_year_count:,}",
            delta=f"{yoy_delta:+,} ({yoy_delta_pct:+.1f}%)",
        )
    else:
        st.metric(f"Latest Year ({latest_year})", f"{latest_year_count:,}")
with col6:
    st.metric("Top Crime Type", str(top_crime_type))
with col7:
    st.metric("Peak Hour", f"{peak_hour:02d}:00" if peak_hour is not None else "N/A")
with col8:
    st.metric("Year Range", f"{year_range[0]} - {year_range[1]}")

@st.cache_data(show_spinner=False)
def to_csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")

with st.expander("Download Filtered Data"):
    download_mode = st.selectbox("Download Size", ["Full", "Sample (100k)"], index=1)
    if download_mode == "Sample (100k)" and len(filtered_df) > 100000:
        download_df = filtered_df.sample(n=100000, random_state=42)
    else:
        download_df = filtered_df
    st.download_button(
        "Download CSV",
        data=to_csv_bytes(download_df),
        file_name="crime_filtered.csv",
        mime="text/csv",
    )


(tab_temporal, tab_spatial, tab_correlation) = st.tabs(
    ["Temporal Pattern Analysis", "Spatial Distribution Study", "Crime Correlation Analysis"]
)


with tab_temporal:
    st.subheader("Long-Term and Seasonal Trends")

    col_a, col_b = st.columns(2)
    with col_a:
        x, y = numeric_xy(yearly_counts)
        plot_line(x, y, "Crime Incidents by Year", "Year", "Total Incidents", xticks=x)

    with col_b:
        x, y = numeric_xy(monthly_counts)
        plot_line(x, y, "Crime Incidents by Month", "Month", "Total Incidents", xticks=MONTH_ORDER)

    col_c, col_d = st.columns(2)
    with col_c:
        x, y = numeric_xy(hourly_counts)
        plot_line(x, y, "Crime Incidents by Hour of Day", "Hour", "Total Incidents", xticks=HOUR_ORDER)

    with col_d:
        plot_bar(
            day_counts.index.tolist(),
            day_counts.values.tolist(),
            "Crime Incidents by Day of Week",
            "Day of Week",
            "Total Incidents",
        )

    st.subheader("Heatmaps")
    heat_col1, heat_col2 = st.columns(2)

    with heat_col1:
        pivot_year_hour = (
            filtered_df.pivot_table(index="Year", columns="Hour", aggfunc="size", fill_value=0)
            .reindex(columns=HOUR_ORDER, fill_value=0)
            .sort_index()
        )
        plot_heatmap(
            pivot_year_hour,
            "Crime Incidents by Year and Hour",
            "Hour of Day",
            "Year",
            "Viridis",
            xticks=HOUR_ORDER,
        )

    with heat_col2:
        pivot_year_month = (
            filtered_df.pivot_table(index="Year", columns="Month", aggfunc="size", fill_value=0)
            .reindex(columns=MONTH_ORDER, fill_value=0)
            .sort_index()
        )
        plot_heatmap(
            pivot_year_month,
            "Crime Trends by Year and Month",
            "Month",
            "Year",
            "Magma",
        )

    pivot_day_hour = (
        filtered_df.pivot_table(
            index="DayOfWeek", columns="Hour", aggfunc="size", fill_value=0
        )
        .reindex(index=DAY_ORDER, columns=HOUR_ORDER, fill_value=0)
    )
    plot_heatmap(
        pivot_day_hour,
        "Crime Incidents by Day of Week and Hour",
        "Hour of Day",
        "Day of Week",
        "Cividis",
        xticks=HOUR_ORDER,
    )

    if compare_mode and compare_values:
        st.subheader("Comparison View")
        compare_series = {}
        for value in compare_values:
            subset = filtered_df[filtered_df[compare_dim] == value]
            if compare_view == "Monthly":
                counts = subset.groupby("Month").size().reindex(MONTH_ORDER, fill_value=0)
                x, y = numeric_xy(counts)
                compare_series[value] = (x, y)
                xticks = MONTH_ORDER
                x_label = "Month"
            elif compare_view == "Hourly":
                counts = subset.groupby("Hour").size().reindex(HOUR_ORDER, fill_value=0)
                x, y = numeric_xy(counts)
                compare_series[value] = (x, y)
                xticks = HOUR_ORDER
                x_label = "Hour"
            else:
                counts = subset.groupby("Year").size().reindex(yearly_counts.index, fill_value=0)
                x, y = numeric_xy(counts)
                compare_series[value] = (x, y)
                xticks = yearly_counts.index
                x_label = "Year"

        plot_multi_line(
            compare_series,
            f"Comparison by {compare_dim} ({compare_view})",
            x_label,
            "Total Incidents",
            xticks=xticks,
        )


with tab_spatial:
    st.subheader("District and Community Patterns")

    district_counts = filtered_df["District"].value_counts().sort_index()
    community_counts = filtered_df["Community Area"].value_counts().sort_index()
    top_locations = filtered_df["Location Description"].value_counts().head(top_n)

    col_a, col_b = st.columns(2)
    with col_a:
        plot_bar(
            district_counts.index.tolist(),
            district_counts.values.tolist(),
            "Crime Distribution by Police District",
            "District",
            "Incidents",
        )

    with col_b:
        community_top = community_counts.head(top_n)
        plot_bar(
            community_top.index.tolist(),
            community_top.values.tolist(),
            f"Top {top_n} Community Areas by Crime Count",
            "Community Area",
            "Incidents",
        )

    plot_bar(
        top_locations.index.tolist(),
        top_locations.values.tolist(),
        f"Top {top_n} Most Common Crime Locations",
        "Incidents",
        "Location Description",
        orientation="h",
    )

    st.subheader("Spatial Distribution Map")
    map_mode = st.radio(
        "Map Mode",
        ["District Scatter", "Heatmap"],
        horizontal=True,
        index=0,
    )
    map_df = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if not map_df.empty:
        sample_n = min(map_sample_size, len(map_df))
        map_sample = map_df.sample(n=sample_n, random_state=42)
        view_state = pdk.ViewState(
            latitude=float(map_sample["Latitude"].mean()),
            longitude=float(map_sample["Longitude"].mean()),
            zoom=9,
            pitch=0,
        )

        if map_mode == "Heatmap":
            heat_radius = st.slider("Heatmap Radius", min_value=10, max_value=80, value=35, step=5)
            layer = pdk.Layer(
                "HeatmapLayer",
                data=map_sample,
                get_position="[Longitude, Latitude]",
                get_weight=1,
                radius=heat_radius,
            )
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
            st.caption("Heatmap shows crime density. Sampling keeps performance responsive.")
        else:
            districts = sorted(pd.Series(map_sample["District"].dropna().unique()).astype(int))
            palette = sns.color_palette("tab20", n_colors=max(1, len(districts)))
            color_map = {
                dist: [int(c * 255) for c in palette[i]] for i, dist in enumerate(districts)
            }

            def color_for_district(value):
                if pd.isna(value):
                    return [160, 160, 160]
                return color_map.get(int(value), [160, 160, 160])

            map_sample = map_sample.copy()
            map_sample["color"] = map_sample["District"].apply(color_for_district)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_sample,
                get_position="[Longitude, Latitude]",
                get_fill_color="color",
                get_radius=40,
                pickable=True,
                opacity=0.6,
            )

            tooltip = {
                "html": "<b>District:</b> {District}<br/>"
                "<b>Type:</b> {Primary Type}<br/>"
                "<b>Location:</b> {Location Description}",
            }

            st.pydeck_chart(
                pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
            )
            st.caption(
                "Map shows a random sample to keep performance responsive. Colors represent districts."
            )
    else:
        st.info("No latitude/longitude data available for the current filters.")


with tab_correlation:
    st.subheader("Crime vs Arrest Dynamics")

    arrests_per_year = (
        filtered_df[filtered_df["Arrest"] == True].groupby("Year").size().sort_index()
    )
    crimes_per_year = filtered_df.groupby("Year").size().sort_index()

    col_a, col_b = st.columns(2)
    with col_a:
        x, y = numeric_xy(arrests_per_year)
        plot_line(x, y, "Total Arrests per Year", "Year", "Arrests", xticks=x)

    with col_b:
        x1, y1 = numeric_xy(crimes_per_year)
        x2, y2 = numeric_xy(arrests_per_year)
        plot_multi_line(
            {"Crimes": (x1, y1), "Arrests": (x2, y2)},
            "Crime vs Arrest Trends",
            "Year",
            "Incidents",
            xticks=sorted(set(x1).union(set(x2))),
        )

    st.subheader("Arrest Rates by Crime Type")
    crime_total = filtered_df["Primary Type"].value_counts()
    crime_arrests = filtered_df[filtered_df["Arrest"] == True]["Primary Type"].value_counts()
    arrest_rate = (crime_arrests / crime_total).dropna().sort_values(ascending=False)

    top_rate = arrest_rate.head(top_n)
    plot_bar(
        top_rate.index.tolist(),
        (top_rate.values * 100).tolist(),
        f"Top {top_n} Crime Types by Arrest Rate",
        "Arrest Rate (%)",
        "Crime Type",
        orientation="h",
    )

    st.subheader("Arrest Proportions by Hour and District")
    col_c, col_d = st.columns(2)

    with col_c:
        arrest_hour = pd.crosstab(filtered_df["Hour"], filtered_df["Arrest"], normalize="index")
        arrest_hour = arrest_hour.reindex(index=HOUR_ORDER, fill_value=0)
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            arrest_hour = arrest_hour.reindex(columns=[False, True], fill_value=0)
            fig.add_bar(x=arrest_hour.index, y=arrest_hour[False], name="No Arrest")
            fig.add_bar(x=arrest_hour.index, y=arrest_hour[True], name="Arrest")
            fig.update_layout(
                barmode="stack",
                title="Arrest Proportion by Hour",
                xaxis_title="Hour",
                yaxis_title="Proportion",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            arrest_hour.plot(kind="bar", stacked=True, color=["#AED6F1", "#2E86C1"], ax=ax)
            ax.set_title("Arrest Proportion by Hour")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Proportion")
            ax.legend(["No Arrest", "Arrest"], loc="upper right")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

    with col_d:
        arrest_district = pd.crosstab(
            filtered_df["District"], filtered_df["Arrest"], normalize="index"
        )
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            arrest_district = arrest_district.reindex(columns=[False, True], fill_value=0)
            fig.add_bar(x=arrest_district.index.astype(str), y=arrest_district[False], name="No Arrest")
            fig.add_bar(x=arrest_district.index.astype(str), y=arrest_district[True], name="Arrest")
            fig.update_layout(
                barmode="stack",
                title="Arrest Proportion by District",
                xaxis_title="District",
                yaxis_title="Proportion",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            arrest_district.plot(kind="bar", stacked=True, color=["#AED6F1", "#2E86C1"], ax=ax)
            ax.set_title("Arrest Proportion by District")
            ax.set_xlabel("District")
            ax.set_ylabel("Proportion")
            ax.legend(["No Arrest", "Arrest"], loc="upper right")
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

    st.subheader("Crime Type vs District Correlation")
    top_crimes = filtered_df["Primary Type"].value_counts().head(top_n).index
    top_districts = filtered_df["District"].value_counts().head(top_n).index
    matrix_df = filtered_df[
        (filtered_df["Primary Type"].isin(top_crimes))
        & (filtered_df["District"].isin(top_districts))
    ]
    crime_district_matrix = pd.crosstab(
        matrix_df["Primary Type"], matrix_df["District"]
    )
    crime_district_matrix_norm = crime_district_matrix.div(
        crime_district_matrix.sum(axis=0), axis=1
    )

    plot_heatmap(
        crime_district_matrix_norm,
        "Crime Composition by District (Normalized)",
        "District",
        "Crime Type",
        "Blues",
    )
