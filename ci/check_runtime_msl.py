#!/usr/bin/env python3
"""Compile every runtime MSL block (R"MSL( ... )MSL") found in overlay sources
with the iPhoneOS Metal compiler. Fails on the first error."""
import pathlib, re, subprocess, sys, tempfile, shutil, glob, os

root = pathlib.Path(sys.argv[1])
blocks = []
for path in sorted(root.rglob("*")):
    if path.suffix not in (".mm", ".m", ".cpp", ".h"): continue
    text = path.read_text(errors="ignore")
    for i, m in enumerate(re.finditer(r'R"MSL\((.*?)\)MSL"', text, re.S)):
        blocks.append((f"{path.name}#{i}", m.group(1)))
print(f"found {len(blocks)} runtime MSL programs")

def find_metal():
    try:
        out = subprocess.run(["xcrun", "-sdk", "iphoneos", "-f", "metal"], capture_output=True, text=True)
        cand = out.stdout.strip()
        if cand and subprocess.run([cand, "--version"], capture_output=True).returncode == 0:
            return [cand]
    except Exception:
        pass
    for pattern in ["/System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain/**/usr/bin/metal",
                    os.path.expanduser("~/Library/Developer/DVTDownloads/**/usr/bin/metal")]:
        for cand in glob.glob(pattern, recursive=True):
            if subprocess.run([cand, "--version"], capture_output=True).returncode == 0:
                return [cand]
    return None

metal = find_metal()
if not metal:
    print("::warning::no usable Metal compiler found; runtime MSL not compile-checked")
    sys.exit(0)
tmp = pathlib.Path(tempfile.mkdtemp())
failed = 0
for name, src in blocks:
    f = tmp / (re.sub(r"[^A-Za-z0-9_]", "_", name) + ".metal")
    f.write_text(src)
    r = subprocess.run(metal + ["-std=metal3.1", "-target", "air64-apple-ios18.0", "-Werror=return-type",
                                "-c", str(f), "-o", str(f) + ".air"], capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{name}: {status}")
    if r.returncode:
        failed += 1
        print(r.stderr)
    elif r.stderr.strip():
        print(r.stderr)
shutil.rmtree(tmp)
if failed:
    sys.exit(f"::error::{failed} runtime MSL program(s) failed to compile")
