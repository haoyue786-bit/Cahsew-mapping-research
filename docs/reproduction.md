# Reproduction Guide

## 1. Software environment

The training environment uses Linux x86-64, Python 3.8.20, PyTorch 2.4 with CUDA 11.8, and GDAL 3.9.2. The environment files record Conda package URLs and MD5 checksums, along with PyPI package versions. Training and map export use NVIDIA GPUs. Table generation, grouping replay, and evaluation can run on a CPU. [Environment files](../environment)

```bash
conda create -n cahsew-research --file environment/linux-64-conda.lock
conda activate cahsew-research
python -m pip install -r environment/linux-64-pip.txt
```

The training agent uses Codex App Server. The recorded Codex CLI version was `0.171.0`. Formal training and visual interpretation used `gpt-6.2-sol` with `max` reasoning effort. Use a dedicated training account with access to that model. If a proxy is required, configure `CASHEW_CODEX_PROXY` for Codex and `CASHEW_IMAGERY_PROXY` for imagery. Set `CASHEW_CODEX_BIN` to select the Codex executable.

## 2. Input assets

Place these files under a single read-only asset root. The names `2019` and `2025` identify periods. Each tile image records its actual AEF year. Treat tile IDs as strings to preserve leading zeros.

| Path relative to the asset root | Contents |
| --- | --- |
| `code/mydataset2020/newdataset/` | 2019 Benin source-domain 64-dimensional AEF inputs and four-class label rasters in `label_processed/`. The repository includes filename lists for 880 training samples and 220 source-domain validation samples. |
| `logs_try_mutil/0226_mutil_GHA_pretrain_200epoch/epoch_weights/epoch_173.pt` | Common source checkpoint for late-period training. The SHA-256 digest is `64b50b34364dc8b59b923482f479de808457845307a86da2d70e56b9d5c1e300`. |
| `alphaearth_data_split/{2019,2025}/<country>/<country><actual_year>_<tile_id>/` | 64-band AEF patches of 128 × 128 pixels. The TIFFs preserve the original projection and the row and column indexes within each tile. |
| `country_shp/ADM1/patch_index/patch_index.sqlite` | AEF directories, patch locations, and spatial bounds. An existing index can be used at another mount point with the same tile IDs, filenames, and actual imagery years. |
| `country_shp/data/2025/<ISO3>/*_2025_4class.{shp,shx,prj}` | Geometry files for protecting target evaluation locations. Training preparation reads only geometry to exclude overlapping hard examples. |

Obtain source weights, labeled source-domain imagery, target reference attributes, and historical high-resolution imagery through authorized channels. Use the AEF preparation script to recompute scene counts in Earth Engine or download AEF for the selected years. Use the tiling and indexing scripts to register AEF inputs for training. Tiling reads the TIFFs and progress records from the original downloader. Indexing requires ADM1 vector boundaries for the study area. [AEF preparation](../methods/prepare_aef.py) · [Tiling script](../workflow/code/split4_xg_jiang_progress_ok_only.py) · [Indexing script](../workflow/country_shp/ADM1/tools/build_patch_index.py)

The verified input index covers all 90 late-period groups. The full early-period run requires the 2019 AEF tile `00000000000000000102` for `WA_group011` and a refreshed index. Registration checks the frozen group membership and year inventories.

Late-period automated annotation also requires Playwright with Chromium, Esri or Google imagery with verifiable acquisition dates, and a SAM 2.1 Large segmentation service through X-AnyLabeling Server. For historical Google imagery, set `CASHEW_GEHI_BIN` to the GEHistoricalImagery installation path. Use `CASHEW_REFERENCE_ASSETS` and `CASHEW_REFERENCE_ROOT` to connect an authorized reference image library. [Annotation environment](annotation-environment.md)

## 3. Rebuild the supplementary tables and groups

Rebuild the supplementary tables and figures from the frozen Sentinel-2 scene counts, tile grid, and group roster. [Inputs and expected outputs](../methods/data/README.md)

```bash
python tools/rebuild_support.py ./reproduced-support
python methods/grouping/group_remaining_tiles.py ./feature-groups
```

Grouping replay starts from the archived AEF means, standard deviations, spatial bounds, and patch counts. It recomputes distances, spatial connectivity, and candidate partitions for 437 tiles, then verifies that the resulting membership of 54 groups matches the frozen records. See the membership table for all 90 groups. [Full group membership](../methods/data/group-membership.csv)

## 4. Register the tiles and run training

Create a separate writable workspace from the cloned repository. The example uses one slot on GPU 0 on a single host. Adjust `--slots` to the available GPU memory. The production two-host configuration uses the host aliases `local` and `124`.

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

Start SAM 2 before launching the queue for late-period annotation. Run `python training_queue.py --host local` to train and export the late-period maps. To process both periods together, run `python prepare_formal.py --host local --year 2019` in one terminal and `python training_queue.py --host local --include-2019` in another terminal in the same workspace. Early-period preparation waits for the same group's late-period SELECT to freeze. It then verifies the model identity and extracts the weights for inference. The two periods share GPU slots.

| Stage | Entry point and outputs |
| --- | --- |
| Input registration | `prepare_formal.py` verifies the selected year per tile and registers 1,000 training patches per group and a separate, nonoverlapping panel of 3,000 unlabeled monitoring patches. |
| Training, tuning, and annotation | The queue launches `run_group.py`. It records completed epochs, applied parameters, and agent decisions in the group directory. |
| STOP and SELECT | `run_group.py select` freezes `frozen_selection.json` and a verified lightweight predictor. |
| Inference and map production | `export_selected.py` runs inference and packaging in two stages. Maps and file hashes are in `jobs/<job>/maps/`. Training the next group can start once model selection is complete. |
| Evaluation | `methods/evaluate_frozen.py` verifies the frozen selection and map identities before reading target reference points. |

Restart the scheduler in the same workspace to resume groups whose state was saved. Retry a failed group with `--retry-job <job-name>`. Each group directory contains its logs and training history. Group membership, sampling rules, and source weights can be frozen. Replaying an original agent trajectory epoch by epoch also requires its original service responses, authorized annotations, and training state. Decisions can differ when rerunning the live service.

## 5. Evaluate frozen maps

Run the command below from the repository root in an environment with authorized access to the reference data. The evaluator verifies `frozen_selection.json`, `maps/export_receipt.json`, and the GeoTIFF content hashes before reading target reference attributes.

```bash
python methods/evaluate_frozen.py /path/to/job /path/to/points.shp \
  --label-field class_id --out ./evaluation.json
```

The output reports the number of valid points, coverage, cashew F1, precision, recall, and the count of uncertain predictions. The evaluator uses the same points for both map versions. It treats code 10 as a negative prediction for cashew and reports uncertain predictions separately. Store target reference labels and evaluation outputs outside the training workspace.

The reference point summary script produces manuscript Table S3 from authorized evaluation layers. It deduplicates locations by coordinates and counts interpretation records by class. Supply the evaluation data root containing `2019/<ISO3>` and `2025/<ISO3>`. [Reference point summary script](../methods/summarize_reference_points.py)
