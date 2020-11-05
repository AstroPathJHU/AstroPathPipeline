import argparse, numpy as np
from astropath_calibration.utilities import units
from .makeplots import alignmentset

def printrms(**kwargs):
  A = alignmentset(**kwargs)
  T = A.T
  dT = T - np.identity(2)
  rd = A.rectangledict
  raw = []
  shiftaffine = []
  shifted = []
  fields = A.fields
  for o in A.overlaps:
    if o.result.exit: continue
    dxvec = o.result.dxvec
    f1, f2 = fields[rd[o.p1]], fields[rd[o.p2]]
    raw.append(dxvec)
    shiftaffine.append(dxvec - dT @ (o.x1vec - o.x2vec))
    shifted.append(dxvec - ((f1.pxvec - f2.pxvec) - (f1.xvec - f2.xvec)))

  raw = units.nominal_values(raw)
  shiftaffine = units.nominal_values(shiftaffine)
  shifted = units.nominal_values(shifted)

  print(units.np.std(raw, axis=0))
  print(units.np.std(shiftaffine, axis=0))
  print(units.np.std(shifted, axis=0))

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("sample", choices=("M21_1", "M1_1", "M2_3", "TS19_0181_A_1_3_BMS_MITRE", "L1_4", "ML1603474_BMS069_5_21", "ML1603480_BMS078_5_22", "PZ1", "M115"))
  p.add_argument("--units", choices=("fast", "safe"), default="safe")
  args = p.parse_args()

  units.setup(args.units)

  printrms(samp=args.sample)
