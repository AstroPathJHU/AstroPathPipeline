#imports
import numpy as np, matplotlib.pyplot as plt

def plot_eigenvalue_coverage(pca) :
    """
    Plot a couple visualizations of the explained variance for the full set of eigenvalues for a PCA 
    """
    xs = np.array(list(range(1,len(pca.explained_variance_ratio_)+1)))
    ys_1 = pca.explained_variance_ratio_
    ys_2 = [np.sum(pca.explained_variance_ratio_[:i]) for i in xs]
    ys_3 = -1.*np.log(np.array([1.-y for y in ys_2]))
    f,ax=plt.subplots(figsize=(10.,4.))
    ax2=ax.twinx()
    ax2.bar(xs,ys_1,label='eigenvalue/sum(eigenvalues)',alpha=0.6)
    ax.bar(xs,ys_3,color='tab:red',alpha=0.65,label='-1*log(1.-cum. frac. variance)')
    ax2.step(xs,ys_2,label='cumulative explained variance')
    ax2.set_title('variance coverage per eigenvalue',fontsize=14)
    ax2.set_ylabel('explained variance',color='tab:blue',fontsize=13)
    ax2.tick_params(axis='y',labelcolor='tab:blue')
    ax.set_xlabel('eigenvalue',fontsize=13)
    ax.set_ylabel('-1*log(1.-cum. frac. variance)',color='tab:red',fontsize=13)
    ax.tick_params(axis='y',labelcolor='tab:red')
    ln1, lab1 = ax.get_legend_handles_labels()
    ln2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(ln1+ln2,lab1+lab2,loc='center')
    plt.show()

def plot_vector_components(pca) :
    """
    Plot the components of each eigenvector of a PCA
    """
    comp_vecs = pca.components_
    f,ax=plt.subplots(figsize=(12.,12.))
    pos = ax.imshow(comp_vecs,cmap='PiYG',vmax=0.5,vmin=-0.5)
    f.colorbar(pos,ax=ax)
    ax.plot((8.5,8.5),(-0.5,34.5),linewidth=3,color='k')
    ax.plot((17.5,17.5),(-0.5,34.5),linewidth=3,color='k')
    ax.plot((24.5,24.5),(-0.5,34.5),linewidth=3,color='k')
    ax.plot((31.5,31.5),(-0.5,34.5),linewidth=3,color='k')
    ax.set_xlabel('raw image layer',fontsize=13)
    ax.set_ylabel('PCA component vector',fontsize=13)
    ax.set_title('raw image layer components of PCA vectors',fontsize=14)
    plt.show()
