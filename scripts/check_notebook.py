"""Check notebook structure/syntax without data downloads or model execution."""
from pathlib import Path
import ast
import json
import re

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "things_eeg2_clip.ipynb"


def python_source(cell):
    return "\n".join("pass" if line.lstrip().startswith(("%", "!")) else line
                     for line in "".join(cell["source"]).splitlines())


def check():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(set(ids)) == len(ids), "Duplicate cell IDs"
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert not cell["outputs"], "Published notebook contains execution output"
            assert cell["execution_count"] is None
            ast.parse(python_source(cell), filename=f"cell-{index}")
            count += 1
    serialized = NOTEBOOK.read_text()
    assert not re.search(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|/Users/)", serialized), "Possible secret or personal path"
    try:
        import nbformat
    except ImportError:
        print("nbformat not installed; JSON, syntax and clean-output checks completed.")
    else:
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
    print(f"Checked {count} code cells: valid syntax, unique IDs, no saved output or personal paths.")


if __name__ == "__main__":
    check()
