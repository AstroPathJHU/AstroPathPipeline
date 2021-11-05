import argparse, numpy as np
from astropath.utilities import units
from .makeplots import alignsample

def printmatrix(**kwargs):
  A = alignsample(**kwargs)
  T = A.T

  A.printlogger.debug()
  A.printlogger.debug("Matrix:")
  A.printlogger.debug(T)

  A.printlogger.debug()
  A.printlogger.debug("Latex with siunitx:")
  for _ in np.ravel(T):
    A.printlogger.debug(units.drawing.siunitxformat(_, power=0, fmt=":f"))

  dT = T - np.identity(2)

  A.printlogger.debug()
  A.printlogger.debug("(Matrix - identity) * 10^3:")
  A.printlogger.debug(dT*1e3)

  A.printlogger.debug()
  A.printlogger.debug("Latex with siunitx:")
  for _ in np.ravel(dT*1e3):
    A.printlogger.debug(units.drawing.siunitxformat(_, power=0, fmt=":f"))

  A.printlogger.debug()
  A.printlogger.debug("Average shift when moving horizontally:")
  o = next(o for o in A.overlaps if o.tag == 4)
  A.printlogger.debug(dT @ (o.x2vec - o.x1vec))

  A.printlogger.debug()
  A.printlogger.debug("Average shift when moving vertically:")
  o = next(o for o in A.overlaps if o.tag == 2)
  A.printlogger.debug(dT @ (o.x2vec - o.x1vec))

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("sample", choices=("M21_1", "M1_1", "M2_3", "TS19_0181_A_1_3_BMS_MITRE", "L1_4", "ML1603474_BMS069_5_21", "ML1603480_BMS078_5_22", "PZ1"))
  p.add_argument("--units", choices=("fast", "safe"), default="safe")
  args = p.parse_args()

  units.setup(args.units)

  printmatrix(samp=args.sample)
