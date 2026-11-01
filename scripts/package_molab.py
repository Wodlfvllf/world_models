"""Build a small, source-only upload bundle for the hosted Molab notebook."""
from pathlib import Path
import hashlib
import json
import zipfile


def notebook_json(root):
    # VS Code's Jupyter serializer may strip per-cell language metadata. Supply
    # the requested explicit JSON-cell representation as a separate export.
    document = json.loads((Path(root) / "notebooks/run_prerequisites.ipynb").read_text())
    for index, cell in enumerate(document["cells"], 1):
        metadata = cell.setdefault("metadata", {})
        metadata["language"] = "markdown" if cell["cell_type"] == "markdown" else "python"
        metadata["id"] = cell.get("id", f"cell-{index}")
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
    return json.dumps({"cells": document["cells"]}, indent=2)


def package(root, output):
    root, output = Path(root), Path(output)
    files = [root / "pyproject.toml"]
    files += list(root.glob("*.md"))
    for directory in ("src", "scripts", "configs", "notebooks", "tests"):
        files += [p for p in (root / directory).rglob("*") if p.is_file()
                  and p.suffix in {".py", ".sh", ".json", ".ipynb", ".md"}
                  and "__pycache__" not in p.parts and ".egg-info" not in str(p)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            if path.is_symlink():
                raise ValueError(f"Refusing source symlink: {path}")
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("notebooks/molab_cells.json", notebook_json(root))
    with output.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    with output.with_suffix(".zip.sha256").open("x") as receipt:
        receipt.write(f"{checksum}  {output.name}\n")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(package(root, root / "dist/world-models-molab-source.zip"))
    with (root / "dist/molab_cells.json").open("x") as export:
        export.write(notebook_json(root))