import argparse, matplotlib.pyplot as plt, numpy as np, pathlib, pickle, sklearn.decomposition

from microscopealignment.baseclasses.sample import ReadRectanglesOverlapsIm3
from microscopealignment.utilities import units

here = pathlib.Path(__file__).parent

root1 = pathlib.Path(r"\\Bki02\e\Clinical_Specimen")
root2 = pathlib.Path(r"\\BKI04\flatw")
samp = "M41_1"

class MyReadRectanglesOverlapsIm3(ReadRectanglesOverlapsIm3):
  @property
  def filetype(self):
    return "flatWarp"
  @property
  def logmodule(self):
    return None
  multilayer = True

def makePCA():
  pca = sklearn.decomposition.IncrementalPCA(n_components=35)
  A = MyReadRectanglesOverlapsIm3(root1, root2, samp, layers=range(1, 36))

  n = max(len(A.rectangles), 1000)
  for i, r in enumerate(A.rectangles, start=1):
    A.logger.info("%d / %d", i, n)
    with r.using_image() as image:
      flattened = image.reshape(image.shape[0], image.shape[1]*image.shape[2]).T
      pca.partial_fit(flattened)
    if i == n: break

  with open(here/"PCA.pkl", "wb") as f:
    pickle.dump(pca, f)

def makeplots():
  with open(here/"PCA.pkl", "rb") as f:
    pca = pickle.load(f)

  A = MyReadRectanglesOverlapsIm3(root1, root2, samp, layers=range(1, 36))
  r = A.rectangles[1600]
  with r.using_image() as image, plt.rc_context({
      #"figure.figsize": (5, 5)
  }):
    flattened = image.reshape(image.shape[0], image.shape[1]*image.shape[2]).T
    transformed = pca.transform(flattened)

    for i, _ in enumerate(image, start=1):
      print(i)
      plt.imshow(_)
      plt.savefig(here/f"layer{i}.pdf")
      plt.close()

    for i, _ in enumerate(transformed.T, start=1):
      print(i)
      vmin, vmax = np.quantile(_, (0.25, 0.75))
      plt.imshow(_.reshape(image.shape[1], image.shape[2]), vmin=vmin, vmax=vmax)
      plt.savefig(here/f"PCA{i}.pdf")
      plt.close()

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--pca", action="store_true", default=not (here/"PCA.pkl").exists())
  args = p.parse_args()
  units.setup("fast")

  if args.pca:
    makePCA()
  makeplots()
