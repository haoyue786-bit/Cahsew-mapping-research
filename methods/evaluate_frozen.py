"""Calculate Cashew point metrics after verifying the frozen selected maps."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import fiona
import rasterio
from rasterio.warp import transform


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_maps(job):
    frozen = read(job / "frozen_selection.json")
    receipt = read(job / "maps/export_receipt.json")
    identity = ("job_id", "selected_epoch", "checkpoint_sha256")
    if tuple(frozen[k] for k in identity) != tuple(receipt[k] for k in identity):
        raise ValueError("Frozen checkpoint and export identities differ")
    maps = {}
    for suffix, name in (("_argmax.tif", "argmax"), ("_uncertain10.tif", "p05")):
        matches = [p for p in receipt["files"] if p.endswith(suffix)]
        if len(matches) != 1 or Path(matches[0]).name != matches[0]:
            raise ValueError(f"Expected one published {name} TIFF")
        raster = job / "maps" / matches[0]
        if sha(raster) != receipt["files"][raster.name]:
            raise ValueError(f"Published map digest does not match {name}")
        maps[name] = raster
    return frozen, maps


def scores(c):
    tp, fp, fn = (c[tag] for tag in ("tp", "fp", "fn"))
    return dict(tp=tp, fp=fp, fn=fn, tn=c["tn"], uncertain=c["uncertain"],
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if tp+fn else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("points", nargs="+", type=Path)
    parser.add_argument("--label-field", default="class_id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Read neither point labels nor scores before validating the frozen deliverables.
    frozen, maps = verified_maps(args.job)
    result = {"job": frozen["job_id"], "epoch": frozen["selected_epoch"],
              "checkpoint_sha256": frozen["checkpoint_sha256"], "points": []}
    with ExitStack() as stack:
        rasters = {name: stack.enter_context(rasterio.open(path)) for name, path in maps.items()}
        reference = rasters["argmax"]
        if any((ds.crs, ds.shape, ds.transform) != (reference.crs, reference.shape, reference.transform)
               for ds in rasters.values()):
            raise ValueError("Published maps must share a grid")
        for path in args.points:
            counts = {name: Counter() for name in maps}
            total, covered = 0, 0
            with fiona.open(path) as points:
                source_crs = points.crs
                for point in points:
                    label = int(point["properties"][args.label_field])
                    if label not in (0, 1, 2, 3):
                        continue
                    geometry = point["geometry"]
                    if geometry is None or geometry["type"] != "Point":
                        raise ValueError(f"Expected a point geometry in {path.name}")
                    x, y = geometry["coordinates"][:2]
                    (x,), (y,) = transform(source_crs, reference.crs, [x], [y])
                    total += 1
                    values = {name: int(next(ds.sample([(x, y)]))[0]) for name, ds in rasters.items()}
                    raw, p05 = values["argmax"], values["p05"]
                    if raw == 255:
                        continue
                    if raw not in (0, 1, 2, 3) or not (raw == p05 or (raw, p05) == (2, 10)):
                        raise ValueError("Published decoders disagree with the map convention")
                    covered += 1
                    for name, value in values.items():
                        outcome = ("tp" if value == 2 else "fn") if label == 2 else (
                            "fp" if value == 2 else "tn")
                        counts[name][outcome] += 1
                        counts[name]["uncertain"] += value == 10
            result["points"].append(dict(
                point_file=path.name, supported_labels=total, covered=covered,
                coverage=covered/total if total else None,
                metrics={name: scores(count) for name, count in counts.items()}))
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
