#miscellaneous plotting functions for warp methods

#imports
import matplotlib.pyplot as plt, numpy as np, seaborn as sns

#helper function to plot a curve fit to data
def plotPolyFit(x,y,c,squared,smooth_points=50) :
    plt.plot(x,y,".",label="Datapoints")
    fit_xs = np.linspace(x[0],x[-1],smooth_points)
    fit_ys = np.polynomial.polynomial.polyval(fit_xs,np.flip(c))
    plt.plot(fit_xs,fit_ys,label=f"Fit")
    deg=len(c)-1
    if squared :
        plt.title(f"Fit with degree {deg} squared polynomial")
    else :
        plt.title(f"Fit with degree {deg} polynomial")
    if squared : 
        plt.xlabel("scaled squared radial distance r^2") 
    else : 
        plt.xlabel("scaled radial distance r")
    plt.ylabel("warp amount")
    ftext="warp="
    for i,coeff in enumerate(np.flip(c)) :
        ftext+=f"{coeff:03f}"
        if i!=0 :
            ftext += "*r"
            if squared :
                ftext+='^2'
            if i!=1 :
                if squared :
                    ftext += f"^{2*i}"
                else :
                    ftext += f"^{i}"
        if i!=deg :
            ftext+=" + "
    plt.text(x[0],0.5*(fit_ys[-1]-fit_ys[0]),ftext)
    plt.legend()
    plt.show()

#helper function to plot r-dependent and x/y-dependent warp fields
def plotWarpFields(r,x,y) :
    f,(ax1,ax2,ax3) = plt.subplots(1,3)
    f.set_size_inches(20.,5.)
    #plot radial field as a heatmap
    g1 = sns.heatmap(r,ax=ax1)
    g1.set_title('radially-dependent warping shifts')
    #plot x and y shifts
    g2 = sns.heatmap(x,ax=ax2)
    g2.set_title('warping shift x components')
    g3 = sns.heatmap(y,ax=ax3)
    g3.set_title('warping shift y components')

#helper function to plot unwarped and warped checkerboard images next to one another
def plotCheckerboards(o,w) :
    f,(ax1,ax2) = plt.subplots(1,2)
    f.set_size_inches(20.,5.)
    ax1.imshow(o,cmap='gray')
    ax2.imshow(w,cmap='gray')
