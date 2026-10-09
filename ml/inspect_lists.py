import json, os

lists_path = r'd:\anvation\ml\data\TBX11K\TBX11K\lists'
for fname in os.listdir(lists_path):
    fpath = os.path.join(lists_path, fname)
    with open(fpath) as f:
        lines = f.readlines()
    first = lines[0].strip() if lines else "empty"
    print(f"{fname}: {len(lines)} lines, first: {first}")
