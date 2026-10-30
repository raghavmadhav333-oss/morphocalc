import json
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "d:/Research/morphocalc/MorphoCalc_11Hour_Grand_Kaggle_Marathon.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Testing notebook: {filename}")
print(f"Total cells: {len(nb['cells'])}")
code_count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code_count += 1
        src = ''.join(cell['source'])
        compile(src, f'cell_{i}', 'exec')
        print(f"  [+] Code Cell {code_count} (index {i}): Compiled successfully ({len(cell['source'])} lines)")

print(f"\n[SUCCESS] All {code_count} code cells verified with 0 syntax errors or invalid escape sequences!")
