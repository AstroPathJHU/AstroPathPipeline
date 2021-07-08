import numpy as np, skimage.transform
from skimage.transform._geometric import _center_and_normalize_points

class TranslationRotation(skimage.transform.AffineTransform):
  def __init__(self, matrix=None, rotation=None, translation=None):
    if matrix is not None:
      if not np.isclose(matrix[0,0], matrix[1,1]) and np.isclose(matrix[0,1], -matrix[1,0]):
        raise ValueError(f"{matrix} includes a scale and/or shear, not just translation and rotation")
      matrix[1,1] = matrix[0,0]
      matrix[1,0] = -matrix[0,1]
    super().__init__(matrix=matrix, rotation=rotation, translation=translation)
    self._coeffs = [0, 1, 2, 5]

  def estimate(self, src, dst):
    """
    https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    """
    n, d = src.shape
    p = src
    q = dst
    w = np.identity(n)

    pbar = np.mean(p, axis=0)
    qbar = np.mean(q, axis=0)

    x = (p - pbar).T
    y = (q - qbar).T

    print(x.shape, w.shape, y.shape)
    S = x @ w @ y.T

    U, Sigma, V = np.linalg.svd(S)
    R = V @ np.diag([1]*(len(V)-1)+[np.linalg.det(V@U.T)]) @ U.T
    t = qbar - R@pbar

    H = np.zeros((d+1, d+1))
    # solution is right singular vector that corresponds to smallest
    # singular value
    H[:d, :d] = R
    H[:d, d] = t
    H[d, d] = 1

    self.params = H

    return True
