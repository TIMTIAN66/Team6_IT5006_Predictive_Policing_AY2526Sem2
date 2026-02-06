# Chicago Crime Dashboard - 右侧筛选说明

下面解释右侧（Sidebar）每个筛选控件的作用：

- `Dataset`
  - 选择要分析的数据集范围。
  - `All (2015-2025)`：训练集 + 测试/验证集的合并。
  - `Train (2015-2024)`：仅 2015-2024 年。
  - `Test/Val (2025)`：仅 2025 年。

- `Year Range`
  - 过滤年份范围，仅保留指定年份区间内的记录。
  - 会影响所有图表、指标和地图。

- `District`
  - 选择一个或多个警区（District）。
  - 若不选，则表示不过滤。

- `Primary Type`
  - 选择一个或多个犯罪类型。
  - 若不选，则表示不过滤。

- `Arrest Filter`
  - `All`：不过滤是否逮捕。
  - `Arrested Only`：仅保留有逮捕记录的案件。
  - `Not Arrested Only`：仅保留无逮捕记录的案件。

- `Top N (Ranked Lists)`
  - 仅控制“排名类图表”的显示数量（不会改变数据本身）。
  - 影响：
    - Top N Community Areas
    - Top N Locations
    - Top N Crime Types by Arrest Rate
    - Crime Type vs District Correlation（取前 N 的犯罪类型和前 N 的 District）

- `Comparison Mode`
  - 开启后可进行对比视图（两条线）。
  - `Compare By`：选择按 District 或 Primary Type 对比。
  - `Compare View`：选择 Yearly / Monthly / Hourly 视角。
  - `Select up to 2...`：最多选 2 个对象，生成对比曲线。

- `Quick Focus (Linked Filters)`
  - 快速将某个 District 或某个 Crime Type 写入全局筛选。
  - 点击 `Apply Focus Filters` 后，会直接覆盖上面的筛选。

- `Map Sample Size`
  - 地图采样数量，控制地图上的点数/热力点数量。
  - 数值越大越精细，但渲染更慢。

- `Map Mode`
  - 在 Spatial Distribution Map 中显示模式：
  - `District Scatter`：按 District 上色的散点图。
  - `Heatmap`：犯罪密度热力图。

- `Heatmap Radius`（仅在 Heatmap 模式出现）
  - 控制热力图影响半径。
  - 越大越平滑，越小越细粒度。

- `Download Filtered Data`
  - `Full`：导出当前过滤后的完整数据。
  - `Sample (100k)`：若数据量过大，导出 10 万条随机样本。

