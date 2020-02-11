#!/usr/bin/env python

import matplotlib.pyplot as plt, numpy as np

def errorcheck(overlaps):
  pulls = []

  for o in overlaps:
    printfirst = True
    if o.p1 > o.p2: continue
    if not np.isfinite(np.trace(o.result.covariance)): continue
    for o1 in overlaps:
      if o1.p1 > o1.p2: continue
      if o1.p1 == o.p1:
        if not np.isfinite(np.trace(o1.result.covariance)): continue
        for o2 in overlaps:
          if o2.p1 > o2.p2: continue
          if o2.p1 == o1.p2 and o2.p2 == o.p2:
            if not np.isfinite(np.trace(o2.result.covariance)): continue
            odx, ody = o.result.dxdy
            o1dx, o1dy = o1.result.dxdy
            o2dx, o2dy = o2.result.dxdy
            diffx, diffy = odx-o1dx-o2dx, ody-o1dy-o2dy
            if printfirst: print(f"  {o.p1:3d} -->         {o.p2:3d}: ({odx:10}, {ody:10})")
            print(f"      --> {o1.p2:3d} --> {o.p2:3d}: ({o1dx+o2dx:10}, {o1dy+o2dy:10})")
            print(f"           difference: ({diffx:10}, {diffy:10})")
            printfirst = False
            pulls.append(diffx.n / diffx.s)
            pulls.append(diffy.n / diffy.s)

  print()
  print()
  print()
  print("pulls:")
  plt.hist(pulls)
  print("mean:   ", np.mean(pulls))
  print("std dev:", np.std(pulls))
