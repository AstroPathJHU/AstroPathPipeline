import numpy as np

from .computeshift import mse

def plotintensityerror(overlap, aligned=True):
    print(overlap.result)
    if aligned:
        red, blue = overlap.shifted.copy()
    else:
        red, blue = overlap.cutimages.astype(float).copy()
    msered = mse(red)
    mseblue = mse(blue)
    red  *= (msered * mseblue) ** .25 / msered ** .5
    blue *= (msered * mseblue) ** .25 / mseblue ** .5
    print(np.mean(red-blue), np.std(red-blue))
    x = red#(red+blue)/2
    y = red-blue
    #x = np.sqrt(red*blue)
    #y = red/blue
    slc = slice(int(abs(overlap.result.dy)/2+1), -int(abs(overlap.result.dy)/2)-1 or None), slice(int(abs(overlap.result.dx)/2+1), -int(abs(overlap.result.dx)/2)-1 or None)
    x = x[slc].flatten()
    y = y[slc].flatten()
    #plt.scatter(x,y)
    print(np.mean(y), np.std(y))

    import seaborn as sns
    sns.regplot(x=x, y=y, x_bins=20, fit_reg=None)

