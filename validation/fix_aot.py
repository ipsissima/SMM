"""Verify/discover AOT outputs after recompiling the independently published EDNM model."""
from pathlib import Path
import shutil,sys
root=Path(sys.argv[1]).resolve()
for name in ('aot_residual','aot_jacobian'):
    binaries=list(root.rglob(name+'*.so'))
    if not binaries: raise RuntimeError('No compiled binary for '+name)
    latest=max(binaries,key=lambda p:p.stat().st_mtime)
    dest=root/'src'/'ednm'/latest.name
    if latest.resolve()!=dest.resolve(): shutil.copy2(latest,dest)
    print(name,latest,dest,dest.stat().st_size)
