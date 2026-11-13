#!/usr/bin/env python3
"""Export existing IQA SQLite records as portable, frozen split metadata."""

import argparse
import csv
import gzip
import io
import math
from pathlib import Path
import sqlite3


def export_scores(databases, images_root, output):
    root = images_root.resolve()
    rows = {}
    for database in databases:
        # Read-only mode prevents a misspelled database name creating an empty DB.
        # Completed score DBs are immutable snapshots. immutable=1 also avoids
        # creating WAL shared-memory files beside a read-only archived database.
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            for raw_path, crfiqa, musiq in db.execute(
                "SELECT crop_path, crfiqa, musiq FROM scores"
            ):
                if crfiqa is None or musiq is None:
                    raise ValueError("Missing quality score for {!r}".format(raw_path))
                source = Path(raw_path)
                if not source.is_absolute():
                    source = root / source
                # Lexical relative paths also work when source images are offline.
                relative = source.relative_to(root)
                if len(relative.parts) < 2 or any(p in (".", "..") for p in relative.parts):
                    raise ValueError("Expected a relative source path with an identity parent: {!r}".format(raw_path))
                key = relative.as_posix()
                values = (float(crfiqa), float(musiq))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Non-finite quality score for " + key)
                if key in rows:
                    raise ValueError("Duplicate score record: " + key)
                rows[key] = values
    if not rows:
        raise ValueError("No score records found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("path", "crfiqa", "musiq"))
                for path, values in sorted(rows.items()):
                    # repr(float) round-trips the original SQLite REAL value.
                    writer.writerow((path, repr(values[0]), repr(values[1])))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, nargs="+", required=True,
                        help="One completed immutable score DB OR its nonoverlapping shards")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="Original all_faces root used in crop_path records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_scores(args.database, args.images_root, args.output)
    print("Exported {:,} frozen quality records to {}".format(count, args.output))


if __name__ == "__main__":
    main()
