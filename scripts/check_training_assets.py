#!/usr/bin/env python3
"""Check the files required by the released RAGNet training mixture."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def count_files(path: Path, pattern: str) -> int:
    return len(glob.glob(str(path / pattern), recursive=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/shared/rc/mairl/datasets/ragnet/processed/data"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/shared/rc/mairl/results/ragnet-reproduction/models/"
            "LLaVA-Lightning-7B-v1-1"
        ),
    )
    parser.add_argument(
        "--sam",
        type=Path,
        default=Path(
            "/shared/rc/mairl/results/ragnet-reproduction/models/"
            "sam_vit_h_4b8939.pth"
        ),
    )
    args = parser.parse_args()

    lisa = args.data_root / "lisa_data"
    checks = [
        ("LLaVA config", args.model / "config.json", None),
        ("LLaVA weight index", args.model / "pytorch_model.bin.index.json", None),
        ("SAM ViT-H", args.sam, None),
        ("ADE20K train images", lisa / "ade20k/images/training", "*.jpg"),
        ("ADE20K train labels", lisa / "ade20k/annotations/training", "*.png"),
        ("COCO 2017 train images", lisa / "coco/train2017", "*.jpg"),
        ("COCO-Stuff train labels", lisa / "cocostuff/train2017", "*.png"),
        (
            "LLaVA-Instruct-150K",
            lisa / "llava_dataset/llava_instruct_150k.json",
            None,
        ),
        ("Mapillary v2 train labels", lisa / "mapillary/training/v2.0/labels", "*.png"),
        ("PACO-LVIS train JSON", lisa / "vlpart/paco/annotations/paco_lvis_v1_train.json", None),
        ("PASCAL-Part train JSON", lisa / "vlpart/pascal_part/train.json", None),
        (
            "PASCAL VOC2010 images",
            lisa / "vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages",
            "*.jpg",
        ),
        ("ReasonSeg train", lisa / "reason_seg/ReasonSeg/train", "*.jpg"),
        ("ReasonSeg val", lisa / "reason_seg/ReasonSeg/val", "*.jpg"),
        (
            "RefCOCO 2014 train images",
            lisa / "refer_seg/images/mscoco/images/train2014",
            "*.jpg",
        ),
    ]
    for name in ("refclef", "refcoco", "refcoco+", "refcocog"):
        checks.append((f"{name} metadata", lisa / f"refer_seg/{name}", "*.p"))

    failed = False
    for name, path, pattern in checks:
        if pattern is None:
            ok = path.is_file() and path.stat().st_size > 0
            detail = f"{path.stat().st_size:,} bytes" if ok else "missing"
        else:
            total = count_files(path, pattern)
            ok = total > 0
            detail = f"{total:,} files" if ok else "missing"
        failed |= not ok
        print(f"{'OK' if ok else 'MISSING':7s} {name:32s} {detail}")

    refclef_instances = lisa / "refer_seg/refclef/instances.json"
    if refclef_instances.is_file():
        instances = json.loads(refclef_instances.read_text())
        images = instances.get("images", [])
        refclef_root = lisa / "refer_seg/images/saiapr_tc-12"
        missing_refclef = [
            image["file_name"]
            for image in images
            if not (refclef_root / image["file_name"]).is_file()
        ]
        ok = bool(images) and not missing_refclef
        detail = f"{len(images) - len(missing_refclef):,}/{len(images):,} referenced files"
    else:
        ok = False
        detail = "missing instances.json"
    failed |= not ok
    print(f"{'OK' if ok else 'MISSING':7s} {'RefCLEF images':32s} {detail}")

    index_path = args.model / "pytorch_model.bin.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing_shards = [
            shard
            for shard in shards
            if not (args.model / shard).is_file()
            or (args.model / shard).stat().st_size == 0
        ]
        if missing_shards:
            failed = True
            print("MISSING LLaVA weight shards:", ", ".join(missing_shards))
        else:
            print(f"OK      LLaVA weight shards             {len(shards)} files")

    for name in (
        "openx_train.pkl",
        "egoobjects_train.pkl",
        "graspnet_train.pkl",
        "rlbench_train.pkl",
        "handal_hard_reasoning_train.pkl",
        "egoobjects_easy_reasoning_train.pkl",
        "egoobjects_hard_reasoning_train.pkl",
    ):
        path = args.data_root / name
        ok = path.is_file() and path.stat().st_size > 0
        failed |= not ok
        detail = f"{path.stat().st_size:,} bytes" if ok else "missing"
        print(f"{'OK' if ok else 'MISSING':7s} {name:32s} {detail}")

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
