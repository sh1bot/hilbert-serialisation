import unittest
import numpy as np
from pixel_bus import analyze


class IndependentAuditTests(unittest.TestCase):
    def test_scalar_reference(self):
        # Independent, direct pixel reads and scalar folding/counting. No production
        # packing, folding, traversal, or transition helpers are used.
        local_orders={'raster':[(0,0),(1,0),(0,1),(1,1)],
                      'z':[(0,0),(1,0),(0,1),(1,1)],
                      'hilbert':[(0,0),(0,1),(1,1),(1,0)]}
        checks=0
        for seed in range(8):
            plane=np.random.default_rng(seed).integers(0,256,(7,11),dtype=np.uint8)
            for cw,ch in [(1,1),(2,2),(3,2)]:
                mw,mh=2*cw,2*ch
                for lag in [1,2,5,100]:
                    report=analyze({'Y':plane},{'Y':(1,1)},{'width':11,'height':7},[(cw,ch)],[lag],(mw,mh))
                    for order,coords in local_orders.items():
                        raw=[];encoded=[]
                        for by in range(0,7,mh):
                            for bx in range(0,11,mw):
                                for cx,cy in coords:
                                    current=[int(plane[min(by+cy*ch+v,6),min(bx+cx*cw+u,10)])
                                             for v in range(ch) for u in range(cw)]
                                    raw.append(current)
                                    encoded.append([v ^ (127 if v & 128 else 0) for v in current])
                        for kind,stream in [('raw',raw),('fold',encoded)]:
                            counts=[[sum(((stream[t][lane]^stream[t-lag][lane])>>bit)&1
                                         for t in range(lag,len(stream))) for bit in range(8)]
                                    for lane in range(cw*ch)]
                            result=next(r for r in report['results'] if r['order']==order)
                            if kind == 'fold':
                                result = result['fold_only']
                            self.assertEqual(counts, result['channels']['Y']['by_lane_lsb_first'], (seed,cw,ch,lag,order,kind))
                            checks+=1
        self.assertEqual(checks, 576)
