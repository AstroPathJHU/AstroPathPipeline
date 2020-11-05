import argparse, numpy as np
from astropath_calibration.utilities import units
from .makeplots import alignmentset

def printmatrix(**kwargs):
  A = alignmentset(**kwargs)
  T = A.T

  print()
  print("Matrix:")
  print(T)

  print()
  print("Latex with siunitx:")
  for _ in np.ravel(T):
    print(units.drawing.siunitxformat(_, power=0, fmt=":f"))

  dT = T - np.identity(2)

  print()
  print("(Matrix - identity) * 10^3:")
  print(dT*1e3)

  print()
  print("Latex with siunitx:")
  for _ in np.ravel(dT*1e3):
    print(units.drawing.siunitxformat(_, power=0, fmt=":f"))

  print()
  print("Average shift when moving horizontally:")
  o = next(o for o in A.overlaps if o.tag == 4)
  print(dT @ (o.x2vec - o.x1vec))

  print()
  print("Average shift when moving vertically:")
  o = next(o for o in A.overlaps if o.tag == 2)
  print(dT @ (o.x2vec - o.x1vec))

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("sample", choices=("M21_1", "M1_1", "M2_3", "TS19_0181_A_1_3_BMS_MITRE", "L1_4", "ML1603474_BMS069_5_21", "ML1603480_BMS078_5_22", "PZ1"))
  p.add_argument("--units", choices=("fast", "safe"), default="safe")
  args = p.parse_args()

  units.setup(args.units)

  printmatrix(samp=args.sample)
