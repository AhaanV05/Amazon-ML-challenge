"""Build the final submission zip in the organizers' layout.

    python scripts/make_package.py SUBMISSION_OUTPUT_DIR
e.g. python scripts/make_package.py submissions/v7_final/output
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "business_entity_resolution"
OUT = ROOT / "submissions" / "Claudes_Plan_submission.zip"
SKIP = {"__pycache__", ".pytest_cache"}


def main(output_dir: str) -> None:
    out = ROOT / output_dir
    files = [(out / name, f"output/{name}") for name in ("matching_results.tsv", "candidate_pairs.tsv")]
    for p in sorted(CODE.rglob("*")):
        if p.is_file() and not SKIP & set(p.relative_to(CODE).parts):
            files.append((p, f"code/business_entity_resolution/{p.relative_to(CODE).as_posix()}"))
    files.append((ROOT / "docs" / "Documentation_template.md", "Documentation_template.md"))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {len(files)} files) with outputs from {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
