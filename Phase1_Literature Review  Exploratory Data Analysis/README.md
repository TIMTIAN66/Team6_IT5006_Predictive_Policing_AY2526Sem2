# Chicago Crime Dashboard (dashboard.py)



**Sidebar Filters**
- `Dataset`: Choose the dataset scope.
- `Year Range`: Filter the data by year. If the dataset contains only one year (e.g., Test/Val 2025), the slider is hidden and the year-based charts in the Temporal tab are hidden.
- `District`: Filter by one or more police districts.
- `Primary Type`: Filter by one or more crime types.
- `Arrest Filter`: Filter by arrest status (`All`, `Arrested Only`, `Not Arrested Only`).
- `Top N (Ranked Lists)`: Controls how many items appear in ranked charts (community areas, locations, arrest rate, and correlation matrix).
- `Comparison Mode`: Compare up to two districts or two crime types with Yearly/Monthly/Hourly views.
- `Quick Focus (Linked Filters)`: Apply one district and one crime type in a single action.
- `Map Sample Size`: Controls how many points are sampled for the map (bigger = more detail but slower).

**Download**
- `Download Filtered Data`: Export the filtered data as CSV. Use `Sample (100k)` to export a smaller random sample.

**Summary Metrics**
- `Total Incidents`: Number of records after filters.
- `Unique Crime Types`: Distinct crime categories in the filtered data.
- `Arrest Rate`: Percent of incidents with arrests.
- `Districts`: Number of districts represented.
- `Latest Year`: Most recent year in the filtered data and YoY change if applicable.
- `Top Crime Type`: Most common crime category.
- `Peak Hour`: Hour with the most incidents.
- `Year Range`: Active year filter range.

**Tab: Temporal Pattern Analysis**
- `Crime Incidents by Year`: Long-term trend across years (hidden for single-year datasets).
- `Crime Incidents by Month`: Seasonal pattern across months.
- `Crime Incidents by Hour of Day`: Daily cycle.
- `Crime Incidents by Day of Week`: Weekly distribution.
- `Crime Incidents by Year and Hour`: Heatmap of yearly vs hourly intensity (hidden for single-year datasets).
- `Crime Trends by Year and Month`: Heatmap of yearly vs monthly intensity (hidden for single-year datasets).
- `Crime Incidents by Day of Week and Hour`: Heatmap of weekly vs hourly intensity.
- `Comparison View`: Side-by-side trend lines for selected districts or crime types.

**Tab: Spatial Distribution Study**
- `Crime Distribution by Police District`: Counts by district.
- `Top Community Areas by Crime Count`: Top-N community areas with the most incidents.
- `Top Crime Locations`: Top-N most common location descriptions.
- `Spatial Distribution Map (District Scatter)`: Points colored by district.
- `Spatial Distribution Map (Heatmap)`: Density-based heatmap of incidents.

**Tab: Crime Correlation Analysis**
- `Total Arrests per Year`: Annual arrest counts.
- `Crime vs Arrest Trends`: Compare total crimes vs arrests over time.
- `Top Crime Types by Arrest Rate`: Highest arrest-rate categories.
- `Arrest Proportion by Hour`: Stacked bar of arrest vs non-arrest by hour.
- `Arrest Proportion by District`: Stacked bar of arrest vs non-arrest by district.
- `Crime Composition by District (Normalized)`: Heatmap showing crime-type mix across districts.
