# 论文附表与附图

| 材料 | 内容 |
| --- | --- |
| [表 S1. 瓦片选年](tile-years.csv) | 676 个瓦片的 ID、四个候选年份的 Sentinel-2 观测条目数、两期所选 AEF 年份 |
| [表 S2. 分组成员](group-membership.csv) | 90 个组的瓦片成员。将瓦片 ID 与表 S1 连接后可得到每组两期用到的实际年份 |
| [表 S3. 评价样本数量](reference-point-counts.csv) | 按国家和四类地表覆盖合并统计，包含坐标去重后的点位数及判读记录数 |
| [图 S1. 两期 AEF 年份](selected-aef-years.pdf) · [PNG 预览](selected-aef-years.png) | 在同一瓦片范围画出早期和晚期的选年结果。PDF 为矢量图 |

瓦片编号是固定的字符串，读取 CSV 时需保留前导零。成员表以 `group_id, tile_id` 为联合键。部分正式组共享瓦片，全国或全域覆盖统计按瓦片 ID 去重。

**图 S1.** 对每块瓦片，在各时期内比较与瓦片相交且场景云量低于 60% 的 Sentinel-2 条目数，取数量较多年份的 AEF。早期 676 块选中 2019，晚期 58 块选中 2024，618 块选中 2025。

表和图通过 [build_manuscript_tables.py](../../tools/build_manuscript_tables.py) 从 Sentinel-2 计数导出表、格网和冻结的成员清单生成。脚本按瓦片 ID 检查期次匹配、晚期库存的年份和完整的分组覆盖。 [选择规则](../../docs/manuscript-supplement.md)

评价样本数量由 [summarize_reference_points.py](../summarize_reference_points.py) 从 11 国共 22 个评价层离线汇总。地理点位用保留 7 位小数的 WGS84 经纬度去重，地类数量以判读记录计数。表中的总数统计整套参考样本库，图件覆盖到的样本数在各自的精度评价中另行报告。
