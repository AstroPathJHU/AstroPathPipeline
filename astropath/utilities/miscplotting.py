import cv2, matplotlib.pyplot as plt, numpy as np, scipy.stats, uncertainties as unc
from .miscpath import cd

def crop_and_overwrite_image(im_path,border=0.03) :
  """
  small helper function to crop white border out of an image
  """
  im = cv2.imread(im_path)
  y_border = int(im.shape[0]*(border/2))
  x_border = int(im.shape[1]*(border/2))
  min_y = 0; max_y = im.shape[0]
  min_x = 0; max_x = im.shape[1]
  while np.min(im[min_y:min_y+y_border,:,:])==255 :
      min_y+=1
  while np.min(im[max_y-y_border:max_y,:,:])==255 :
      max_y-=1
  while np.min(im[:,min_x:min_x+x_border,:])==255 :
      min_x+=1
  while np.min(im[:,max_x-x_border:max_x,:])==255 :
      max_x-=1
  cv2.imwrite(im_path,im[min_y:max_y+1,min_x:max_x+1,:])

def pullhist(array, *, binning=None, label="", stdinlabel=True, quantileforstats=1, logger=None, **kwargs):
  """
  Make a histogram of uncertainties.nominal_values(array) / uncertainties.std_dev(array)
  """
  pulls = np.array([_.n / _.s for _ in array], dtype=float)
  quantiles = np.array(sorted(((1-quantileforstats)/2, (1+quantileforstats)/2)))
  minpull, maxpull = np.quantile(pulls, quantiles)
  outliers = len(pulls[(minpull > pulls) | (pulls > maxpull)])
  pulls = pulls[(minpull <= pulls) & (pulls <= maxpull)]

  if stdinlabel:
    if label: label += ": "
    label += rf"$\text{{std dev}} = {np.std(pulls):.02f}$"
  if logger is not None:
    logger.info(f"mean of middle {100*quantileforstats}%:    {unc.ufloat(np.mean(pulls), scipy.stats.sem(pulls))}")
    logger.info(f"std dev of middle {100*quantileforstats}%: {unc.ufloat(np.std(pulls), np.std(pulls) / np.sqrt(2*len(pulls)-2))}")
    logger.info(f"n outliers: {outliers}")
  return plt.hist(pulls, bins=binning, label=label, **kwargs)

def save_figure_in_dir(pyplot_inst,figname,save_dirpath=None) :
  """
  Save the current figure in the given pyplot instance with a given name and crop it. 
  If save_dirpath is given the figure is saved in that directory (possibly creating it)
  """
  if save_dirpath is not None :
    if not save_dirpath.is_dir() :
      save_dirpath.mkdir(parents=True)
    with cd(save_dirpath) :
      pyplot_inst.savefig(figname)
      pyplot_inst.close()
      crop_and_overwrite_image(figname)
  else :
    pyplot_inst.savefig(figname)
    pyplot_inst.close()
    crop_and_overwrite_image(figname)
