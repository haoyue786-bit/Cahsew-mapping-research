# -*- coding: utf-8 -*-
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from osgeo import gdal
from tqdm import tqdm


ALLOWED_PROGRESS_STATUS = {"OK", "EXIST", "COPY"}


def check_folder(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def pad_tile(tile: np.ndarray, size: int, pad_value: float):
    if tile.ndim == 3:
        padded = np.full((tile.shape[0], size, size), pad_value, dtype=tile.dtype)
        padded[:, :tile.shape[1], :tile.shape[2]] = tile
    else:
        padded = np.full((size, size), pad_value, dtype=tile.dtype)
        padded[:tile.shape[0], :tile.shape[1]] = tile
    return padded


def clean_to_zero(arr: np.ndarray, nodata_value, vmin=-10000, vmax=100000):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if nodata_value is not None:
        arr = np.where(np.isclose(arr, nodata_value), 0.0, arr)
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < vmin] = 0.0
    arr[arr > vmax] = 0.0
    return arr


def _gdal_dtype(arr: np.ndarray):
    if arr.dtype == np.uint8:
        return gdal.GDT_Byte
    if arr.dtype == np.uint16:
        return gdal.GDT_UInt16
    if arr.dtype == np.int16:
        return gdal.GDT_Int16
    if arr.dtype == np.uint32:
        return gdal.GDT_UInt32
    if arr.dtype == np.int32:
        return gdal.GDT_Int32
    if arr.dtype == np.float32:
        return gdal.GDT_Float32
    if arr.dtype == np.float64:
        return gdal.GDT_Float64
    return gdal.GDT_Float32


def write_tile(filename: str, proj: str, geotrans: tuple, im_data: np.ndarray):
    if im_data.ndim == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        filename,
        im_width,
        im_height,
        im_bands,
        _gdal_dtype(im_data),
        options=["BIGTIFF=YES", "COMPRESS=LZW"],
    )
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    dataset = None


def tile_geotransform(gt: tuple, col0: int, row0: int):
    return (
        gt[0] + col0 * gt[1] + row0 * gt[2],
        gt[1],
        gt[2],
        gt[3] + col0 * gt[4] + row0 * gt[5],
        gt[4],
        gt[5],
    )


def iter_expected_tile_names(base: str, steps_x: int, steps_y: int):
    for y_step in range(steps_y):
        for x_step in range(steps_x):
            yield f"{base}_{x_step}_{y_step}.tif"


def find_progress_txt(inputpath: str, progress_txt_path: str = None):
    if progress_txt_path:
        if not os.path.isfile(progress_txt_path):
            raise FileNotFoundError(f"progress txt not found: {progress_txt_path}")
        return progress_txt_path

    candidates = sorted(
        os.path.join(inputpath, fn)
        for fn in os.listdir(inputpath)
        if fn.endswith("_progress.txt")
    )
    if not candidates:
        raise FileNotFoundError(f"no *_progress.txt found under: {inputpath}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple progress txt found under {inputpath}, please pass progress_txt_path explicitly: {candidates}"
        )
    return candidates[0]


def load_allowed_tile_ids(progress_txt_path: str):
    last_status_by_tile = {}
    with open(progress_txt_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            status = parts[0].strip().upper()
            tile_id = parts[2].strip()
            if not tile_id or tile_id == "-":
                continue
            last_status_by_tile[tile_id] = status

    allowed_tile_ids = {
        tile_id
        for tile_id, status in last_status_by_tile.items()
        if status in ALLOWED_PROGRESS_STATUS
    }
    return allowed_tile_ids, last_status_by_tile


def extract_source_tile_id(file_path: str):
    base = os.path.splitext(os.path.basename(file_path))[0]
    parts = base.split("_")
    if len(parts) < 2:
        raise ValueError(f"unexpected tif name: {file_path}")
    return parts[-1]


def split_one_tif(
    file_path: str,
    outpath: str,
    size: int,
    vmin: float,
    vmax: float,
    skip_existing_tiles: bool = True,
):
    base = os.path.splitext(os.path.basename(file_path))[0]
    dataset = gdal.Open(file_path)
    if dataset is None:
        return f"Error: failed to open {file_path}"

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    gt = dataset.GetGeoTransform()
    proj = dataset.GetProjection()

    band1 = dataset.GetRasterBand(1)
    nodata = band1.GetNoDataValue()

    file_folder = os.path.join(outpath, base)
    check_folder(file_folder)

    rows = im_height
    cols = im_width
    steps_x = (cols + size - 1) // size
    steps_y = (rows + size - 1) // size
    total_tiles = steps_x * steps_y

    existing_tile_names = set()
    if os.path.isdir(file_folder):
        existing_tile_names = {
            fn for fn in os.listdir(file_folder) if fn.lower().endswith(".tif")
        }

    expected_tile_names = set(iter_expected_tile_names(base, steps_x, steps_y))
    existing_expected_count = len(expected_tile_names & existing_tile_names)

    if skip_existing_tiles and expected_tile_names.issubset(existing_tile_names):
        dataset = None
        return f"Skip: {base} (already split {existing_expected_count}/{total_tiles})"

    data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)
    stride = size

    written_tiles = 0
    reused_tiles = 0
    with tqdm(total=total_tiles, desc=f"Splitting {base}", unit="tile") as pbar:
        for y_step in range(steps_y):
            for x_step in range(steps_x):
                col0 = x_step * stride
                row0 = y_step * stride
                col1 = min(col0 + size, cols)
                row1 = min(row0 + size, rows)

                if dataset.RasterCount != 1:
                    cur = data[:, row0:row1, col0:col1]
                else:
                    cur = data[row0:row1, col0:col1]

                cur = clean_to_zero(cur, nodata, vmin=vmin, vmax=vmax)

                if cur.ndim == 3:
                    needs_pad = cur.shape[1] < size or cur.shape[2] < size
                else:
                    needs_pad = cur.shape[0] < size or cur.shape[1] < size
                if needs_pad:
                    cur = pad_tile(cur, size, 0.0)

                out_name = f"{base}_{x_step}_{y_step}.tif"
                out_tile = os.path.join(file_folder, out_name)
                if skip_existing_tiles and out_name in existing_tile_names:
                    reused_tiles += 1
                    pbar.update(1)
                    continue

                write_tile(out_tile, proj, tile_geotransform(gt, col0, row0), cur)
                written_tiles += 1

                if np.all(cur == 0):
                    print(f"[WARNING] All-zero tile detected: {out_tile}")

                pbar.update(1)

    dataset = None
    return f"Done: {base} (written={written_tiles}, reused={reused_tiles}, total={total_tiles})"


def split(
    inputpath: str,
    outpath: str,
    size: int,
    progress_txt_path: str = None,
    vmin: float = -10000,
    vmax: float = 100000,
    workers: int = 4,
    skip_existing_tiles: bool = True,
):
    check_folder(outpath)

    progress_txt_path = find_progress_txt(inputpath, progress_txt_path)
    allowed_tile_ids, last_status_by_tile = load_allowed_tile_ids(progress_txt_path)

    tif_list = []
    skipped_by_progress = 0
    for fn in sorted(os.listdir(inputpath)):
        if not fn.lower().endswith(".tif"):
            continue
        fp = os.path.join(inputpath, fn)
        tile_id = extract_source_tile_id(fp)
        if tile_id not in allowed_tile_ids:
            skipped_by_progress += 1
            last_status = last_status_by_tile.get(tile_id, "MISSING")
            print(f"Skip by progress: {fn} (status={last_status})")
            continue
        tif_list.append(fp)

    print(f"Progress txt: {progress_txt_path}")
    print(f"Allowed status: {sorted(ALLOWED_PROGRESS_STATUS)}")
    print(
        f"Source tif: total={len([fn for fn in os.listdir(inputpath) if fn.lower().endswith('.tif')])}, "
        f"eligible={len(tif_list)}, skipped_by_progress={skipped_by_progress}"
    )

    if not tif_list:
        print("no eligible tif file found")
        return

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(split_one_tif, fp, outpath, size, vmin, vmax, skip_existing_tiles)
            for fp in tif_list
        ]
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="Nigeria")
    parser.add_argument("--year", default="2025")
    parser.add_argument("--inputpath", default=None)
    parser.add_argument("--outpath", default=None)
    parser.add_argument("--progress-txt-path", default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--vmin", type=float, default=-10000)
    parser.add_argument("--vmax", type=float, default=100000)
    args = parser.parse_args()

    inputpath = args.inputpath or f"@WORKSPACE@/alphaearth_data/2024_2025_by_s2_count_cloudlt60/{args.country}"
    outpath = args.outpath or f"@ASSETS@/alphaearth_data_split/{args.year}/{args.country}"

    split(
        inputpath,
        outpath,
        args.size,
        progress_txt_path=args.progress_txt_path,
        vmin=args.vmin,
        vmax=args.vmax,
        workers=args.workers,
    )
