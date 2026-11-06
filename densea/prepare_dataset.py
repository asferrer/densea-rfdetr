"""Convert DenSea-Aggregated COCO annotations to the directory layout RF-DETR reads.

RF-DETR expects ``<out>/train``, ``<out>/valid`` and ``<out>/test``, each with an
``_annotations.coco.json``. File names are written as absolute paths (``image-root`` + the
``file_name`` of the annotation), so the images themselves are not copied.

    python densea/prepare_dataset.py --annotations-dir /data/annotations \
        --image-root /data --out-dir /runs/rfdetr_data/real_only

Optionally, extra training images (e.g. a synthetic batch in COCO format) are appended to the
training split with ``--extra-train-json`` and ``--extra-image-root``; validation and test are
never modified.
"""
import argparse
import json
from pathlib import Path

SPLITS = (("train", "train"), ("val", "valid"), ("test", "test"))


def load(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def absolutize(ds, root):
    root = str(root).replace("\\", "/").rstrip("/")
    for im in ds["images"]:
        im["file_name"] = f"{root}/{im['file_name']}"
    for cat in ds["categories"]:
        cat["supercategory"] = "debris"
    return ds


def append_extra(ds, extra):
    """Append the images and annotations of ``extra`` to ``ds`` with fresh ids."""
    img_off = max((im["id"] for im in ds["images"]), default=0) + 1
    ann_off = max((a["id"] for a in ds["annotations"]), default=0) + 1
    names = {c["id"]: c["name"] for c in ds["categories"]}
    for c in extra["categories"]:
        if names.get(c["id"]) != c["name"]:
            raise ValueError(f"category mismatch for id {c['id']}: {names.get(c['id'])!r} vs {c['name']!r}")
    for im in extra["images"]:
        ds["images"].append({**im, "id": im["id"] + img_off})
    for a in extra["annotations"]:
        ds["annotations"].append({**a, "id": a["id"] + ann_off, "image_id": a["image_id"] + img_off})
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-dir", required=True, help="directory with train.json, val.json and test.json")
    ap.add_argument("--image-root", required=True, help="directory that file_name entries are relative to")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-train-json", default=None)
    ap.add_argument("--extra-image-root", default=None)
    args = ap.parse_args()
    if bool(args.extra_train_json) != bool(args.extra_image_root):
        ap.error("--extra-train-json and --extra-image-root go together")

    for src_name, dst_name in SPLITS:
        ds = absolutize(load(Path(args.annotations_dir) / f"{src_name}.json"), args.image_root)
        if src_name == "train" and args.extra_train_json:
            extra = absolutize(load(args.extra_train_json), args.extra_image_root)
            ds = append_extra(ds, extra)
        out = Path(args.out_dir) / dst_name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "_annotations.coco.json", "w", encoding="utf8") as f:
            json.dump(ds, f)
        print(f"{dst_name}: {len(ds['images'])} images, {len(ds['annotations'])} annotations")


if __name__ == "__main__":
    main()
