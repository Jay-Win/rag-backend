import os, re, sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
pat = re.compile(r'^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))')

imports = {}
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(dirpath, f)
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                lines = []
                for line in fh:
                    m = pat.match(line)
                    if m:
                        mod = m.group(1) or m.group(2)
                        lines.append(line.rstrip())
                if lines:
                    imports[p] = lines

for p, lines in sorted(imports.items()):
    print(f"\n# {p}")
    for l in lines:
        print(l)
