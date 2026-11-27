"""Checkpoint independent graphs in small, serially executed batches."""
from large_pair import *
import sys
n,ng,start=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);amp=float(sys.argv[4]);bd=sys.argv[5];ages=list(map(int,sys.argv[6:])) or DAYS
idx=[DAYS.index(d) for d in ages]
for g in range(start,start+ng):
 p=HERE/f'checkpoint_{n}_{amp:g}_{bd}_{"-".join(map(str,ages))}_{g:03}.json'
 if not p.exists():
  t=time.monotonic();o=job((n,g,amp,bd,idx,True));p.write_text(json.dumps(o,separators=(',',':')));print(g,round(time.monotonic()-t,2),flush=True)
