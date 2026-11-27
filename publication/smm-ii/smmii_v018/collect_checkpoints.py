from large_pair import *
import sys
n,ng,amp,bd=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),sys.argv[4];ages=list(map(int,sys.argv[5:])) or DAYS
s='-'.join(map(str,ages));a=sorted(HERE.glob(f'checkpoint_{n}_{amp:g}_{bd}_{s}_*.json'));assert len(a)==ng
r=[json.loads(p.read_text()) for p in a];assert [q['graph'] for q in r]==list(range(ng));assert all([z['day'] for z in q['rows']]==ages for q in r)
o=dict(n=n,sites=n**3,graphs=ng,amplitude=amp,boundary=bd,age_idx=[DAYS.index(d) for d in ages],records=r)
(HERE/f'large_{n}_{ng}_{amp:g}_{bd}_{s}.json').write_text(json.dumps(o,separators=(',',':')))
