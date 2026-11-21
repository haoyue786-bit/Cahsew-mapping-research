# 复现指南

## 1. 准备环境

训练环境为 Linux x86-64、Python 3.8.20、PyTorch 2.4 系列 CUDA 11.8、GDAL 3.9.2。原环境的 Conda 包 URL 和 MD5、PyPI 包版本记录在 [environment](../environment)。训练和地图导出使用 NVIDIA GPU。表格、分组重放和评估可在 CPU 上运行。

```bash
conda create -n cahsew-research --file environment/linux-64-conda.lock
conda activate cahsew-research
python -m pip install -r environment/linux-64-pip.txt
```

训练 Agent 使用 Codex App Server。记录的 CLI 版本是 `0.171.0`，正式训练和视觉判读使用 `gpt-6.2-sol`、`max` 推理。准备有该模型访问权限的专用训练账号。网络需要代理时，分别设置 `CASHEW_CODEX_PROXY` 与 `CASHEW_IMAGERY_PROXY`。可用 `CASHEW_CODEX_BIN` 指定 Codex 可执行文件。

## 2. 准备输入资产

将下列文件放在同一个只读资产根目录。名称中的 `2019` 与 `2025` 代表时期，瓦片影像本身保留实际 AEF 年份。瓦片 ID 为保留前导零的字符串。

| 资产根下的路径 | 内容 |
| --- | --- |
| `code/mydataset2020/newdataset/` | 2019 Benin 源域 64 维 AEF 及 `label_processed/` 四类栅格。880 个训练样本、220 个源域验证样本的文件名清单随代码提供 |
| `logs_try_mutil/0226_mutil_GHA_pretrain_200epoch/epoch_weights/epoch_173.pt` | 晚期统一的源域起点。SHA‑256 为 `64b50b34364dc8b59b923482f479de808457845307a86da2d70e56b9d5c1e300` |
| `alphaearth_data_split/{2019,2025}/<country>/<country><actual_year>_<tile_id>/` | 64 波段、128 × 128 的 AEF 栅格块。原始投影和瓦片内行列索引随 TIFF 保留 |
| `country_shp/ADM1/patch_index/patch_index.sqlite` | AEF 目录、块位置和空间边界。已有索引可换挂载点，固定的瓦片 ID、文件名和实际年份保持一致 |
| `country_shp/data/2025/<ISO3>/*_2025_4class.{shp,shx,prj}` | 保护目标点评价位置的几何副本。训练准备器只读几何以排除重叠难例 |

源权重及源域图像和标签、目标点评价属性、历史高分辨率影像通过各自的授权渠道取得。可用 [prepare_aef.py](../methods/prepare_aef.py) 从 Earth Engine 重算场景数或下载选定年度 AEF。使用 [切片工具](../workflow/code/split4_xg_jiang_progress_ok_only.py) 和 [索引工具](../workflow/country_shp/ADM1/tools/build_patch_index.py) 将原始 AEF 注册为训练输入。切片输入采用原始下载程序的 TIFF 和下载进度清单，索引构建需要研究区的 ADM1 矢量。

已核对的输入索引覆盖晚期 90 组。要执行完整的早期方案，还需给 `WA_group011` 准备瓦片 `00000000000000000102` 的 2019 AEF 并重建索引。登记程序按冻结的成员和年份清单检查。

晚期自动标注还需要运行浏览器的 Playwright/Chromium、提供可核验日期的 Esri 或 Google 影像，以及 X-AnyLabeling Server 的 SAM2.1 Large 分割服务。若使用 Google 历史影像，设置 `CASHEW_GEHI_BIN` 为 GEHistoricalImagery 安装路径。可通过 `CASHEW_REFERENCE_ASSETS` 和 `CASHEW_REFERENCE_ROOT` 接入获准使用的参考图库。依赖服务与路径见 [自动标注环境](annotation-environment.md)。

## 3. 重建论文表格和特征分组

用冻结的 Sentinel-2 计数、格网和组别清单重建附表与附图。原始输入见 [表格来源](../methods/data/README.md)。

```bash
python tools/rebuild_support.py ./reproduced-support
python methods/grouping/group_remaining_tiles.py ./feature-groups
```

分组重放从存档的 AEF 均值、标准差、空间范围和片数开始，重新计算 437 块瓦片的距离、空间连通性、候选分组和 54 组成员，并校验重算结果与冻结成员一致。完整 90 组的成员由 [group-membership.csv](../methods/data/group-membership.csv) 指定。

## 4. 注册两期并训练

从克隆目录建立独立可写的运行区。下例为单机、GPU 0 一个槽位。按实际显存调整 `--slots`。生产双主机模式使用 `local` 和 `124` 两个配置名。

```bash
cashew_assets=/path/to/authorized-assets
cashew_work=/path/to/new-cahsew-workspace
python tools/materialize.py --workspace "$cashew_work" --assets "$cashew_assets" \
  --all-groups --single-host --slots 0:1

export CASHEW_CODEX_ACCOUNT_HOME="$cashew_work/.codex_training_current_account"
CODEX_HOME="$CASHEW_CODEX_ACCOUNT_HOME" codex login
cd "$cashew_work/experiments/cashew_fada_formal2025_20261120_v1"
python -m pytest -q test_hard_samples.py
python prepare_formal.py --host local --year 2025
```

在启动队列前为晚期标注启动 SAM2。命令 `python training_queue.py --host local` 执行晚期训练和导出。若同时准备两期，在一个终端运行 `python prepare_formal.py --host local --year 2019`，在同一运行区的另一终端运行 `python training_queue.py --host local --include-2019`。早期准备等到同组晚期 SELECT 冻结，然后校验模型身份并提取可推理权重。两期共享显存槽位。

| 阶段 | 入口和产物 |
| --- | --- |
| 输入注册 | `prepare_formal.py` 确认每块瓦片的选年。每组注册 1,000 块训练影像及互不重叠的 3,000 块无标签监测影像 |
| 训练、调参、标注 | `run_group.py` 由队列启动。它将已完成轮次、实际参数和 Agent 决策写入组目录 |
| STOP / SELECT | `run_group.py select` 冻结 `frozen_selection.json` 和通过核验的轻量 predictor |
| 预测和制图 | `export_selected.py` 按预测、打包两阶段运行。地图及文件哈希在 `jobs/<job>/maps/`。选模完成时可启动下一个训练 |
| 评价 | `methods/evaluate_frozen.py` 验证冻结与地图身份后读取目标点评价文件 |

同一工作区可重新启动调度程序以继续已持久化的组别。重试失败组用 `--retry-job <job-name>`。日志和训练历史保存在对应组目录。分组、样本抽取规则和源权重都可冻结。逐轮重放历史 Agent 轨迹需要另外保存原始服务响应、获准使用的实际标注及训练状态。实时服务的重新运行可能产生不同的选择。

## 5. 冻结后的目标点评价

在可访问合法评价数据的环境运行以下命令。脚本先检查 `frozen_selection.json`、`maps/export_receipt.json` 和 GeoTIFF 的内容哈希，再读取目标点评价属性。

```bash
python methods/evaluate_frozen.py /path/to/job /path/to/points.shp \
  --label-field class_id --out ./evaluation.json
```

输出包括有效点数、覆盖率、腰果 F1、精确率、召回率和不确定数量。两套地图用相同点位，代码 10 计为未预测腰果并单列统计。评估输出和目标点标签与训练工作区隔离。

论文表 S3 由 [summarize_reference_points.py](../methods/summarize_reference_points.py) 读取授权评价层并汇总。它按位置去重点位，同时累计各类判读记录。运行时提供包含 `2019/<ISO3>` 和 `2025/<ISO3>` 的评价数据根目录。
