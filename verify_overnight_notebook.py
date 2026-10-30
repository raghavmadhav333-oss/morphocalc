import json
import ast
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

nb_path = r"d:\Research\morphocalc\MorphoCalc_Overnight_Grand_Campaign_Kaggle.ipynb"

print(f"[*] Validating {nb_path}...")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])
print(f"Total cells: {len(cells)}")

code_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"Code cells: {len(code_cells)}")

for i, cell in enumerate(code_cells):
    src = "".join(cell.get("source", []))
    try:
        ast.parse(src)
        print(f"  [OK] Code cell {i+1}: Valid Python syntax ({len(src.splitlines())} lines)")
    except SyntaxError as e:
        print(f"  [ERROR] Code cell {i+1} SyntaxError: {e}")
        sys.exit(1)

print("\n[OK] All notebook cells passed AST syntax validation successfully!")
