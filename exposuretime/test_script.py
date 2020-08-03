#imports
from ..alignment.alignmentset import AlignmentSet, AlignmentSetFromXML
from ..alignment.overlap import AlignmentOverlap
from ..warping.utilities import WarpImage
from ..utilities.img_file_io import getSampleMaxExposureTimesByLayer, getExposureTimesByLayer, getRawAsHWL
from ..utilities.misc import cd
import numpy as np, matplotlib.pyplot as plt
import scipy, os, glob, random, cv2, dataclasses, platform

#constants
if platform.system()=='Darwin' : #the paths on my Mac
    root1_dir = os.path.join(os.sep,'Volumes','E','Clinical_Specimen')
    root2_dir = os.path.join(os.sep,'Volumes','dat')
    fw01_root2_dir = os.path.join(os.sep,'Volumes','G','heshy','flatw')
else :
    root1_dir = r"W:\\Clinical_Specimen"
    root2_dir = r"X:\\"
    fw01_root2_dir = r"Z:\\heshy\\flatw"
#sample = 'M21_1'
sample = 'M41_1'
workingdir_name = 'EXPOSURE_TIME_TEST_SCRIPT_OUTPUT'
flatfield_file = os.path.join('flatfield_batch_3-9_samples_22692_initial_images','flatfield.bin')
layer = 1
nclip=8
#overlaps=list(range(200)) #only load 200 overlaps to test
raw_fit_1_dirname = 'raw_fit_1'
raw_fit_2_dirname = 'raw_fit_2'
fw_fit_1_dirname  = 'fw_fit_1'
fw_fit_2_dirname  = 'fw_fit_2'
#make the working directories
if not os.path.isdir(workingdir_name) :
    os.mkdir(workingdir_name)
with cd(workingdir_name) :
    dns = [raw_fit_1_dirname,raw_fit_2_dirname,fw_fit_1_dirname,fw_fit_2_dirname]
    for dn in dns :
        if not os.path.isdir(dn) :
            os.mkdir(dn)

#helper class for comparing overlap image exposure times
@dataclasses.dataclass
class ETOverlap :
    olap : AlignmentOverlap
    p1et : float
    p2et : float
    @property
    def p1_im(self):
        return self.olap.shifted[0]
    @property
    def p2_im(self):
        return self.olap.shifted[1]
    @property
    def et_diff(self):
        return self.p2et-self.p1et

#helper class for doing the fitting to find the optimal offset
class Fit :
    
    def __init__(self,etos,raw_or_fw,fitn) :
        self.etos = etos
        self.raw_or_fw = raw_or_fw
        self.fitn = fitn
        self.offsets = []
        self.costs = []
        self.iters=0
        self.n_olaps = len(self.etos)
        self.best_fit_offset=None

    def cost(self,pars) :
        self.iters+=1
        offset=pars[0]
        cost=0
        for eto in self.etos :
            corr_p1im, corr_p2im = self.__correctImages(eto,offset)
            cost+=self.__calcSingleCost(corr_p1im,corr_p2im)
        cost/=self.n_olaps
        self.offsets.append(offset); self.costs.append(cost)
        if self.iters%self.print_every==0 :
            print(f'iteration {self.iters}: offset = {offset:.4f}, cost = {cost:.4f}')
        return cost
    
    def doFit(self,initial_offset=10,max_iter=15000,print_every=10) :
        self.print_every=print_every
        msg = f'Starting fit at offset = {initial_offset}; will run for a max of {max_iter} '
        msg+= f'iterations printing every {print_every}'
        print(msg)
        self.result = scipy.optimize.minimize(self.cost,
                                              [initial_offset],
                                              method='L-BFGS-B',
                                              jac='3-point',
                                              bounds=[(0,100)],
                                              options={'disp':True,
                                                       'ftol':1e-20,
                                                       'gtol':1e-15,
                                                       'eps':2,
                                                       'maxiter':max_iter,
                                                       'iprint':self.print_every,
                                                       'maxls':2*self.n_olaps,
                                                       'finite_diff_rel_step':[0.2],
                                                      }
                                             )
        print(f'Done! Minimization terminated with exit {self.result.message}')
        print(f'Best-fit offset = {self.result.x[0]:.4f} (best cost = {self.result.fun:.4f})')
        self.best_fit_offset=self.result.x[0]
        
    def saveCorrectedImages(self,n_images,dirpath) :
        if self.best_fit_offset is None :
            raise Exception('ERROR: best fit offset not yet determined')
        for i,eto in enumerate(self.etos[:min(n_images,len(self.etos))],start=1) :
            orig = np.array([eto.p1_im, eto.p2_im, 0.5*(eto.p1_im+eto.p2_im)]).transpose(1, 2, 0) / 1000.
            orig_cost = self.__calcSingleCost(eto.p1_im,eto.p2_im)
            corr_p1im, corr_p2im = self.__correctImages(eto,self.best_fit_offset)
            corr = np.array([corr_p1im, corr_p2im, 0.5*(corr_p1im+corr_p2im)]).transpose(1, 2, 0) / 1000.
            corr_cost = self.__calcSingleCost(corr_p1im,corr_p2im)
            f,ax = plt.subplots(1,2,figsize=(2*6.4,4.6))
            ax[0].imshow(orig)
            ax[0].set_title(f'overlap {i} original (cost={orig_cost:.2f})')
            ax[1].imshow(corr)
            ax[1].set_title(f'overlap {i} corrected (cost={corr_cost:.2f})')
            with cd(dirpath) :
                plt.savefig(f'{self.raw_or_fw}_overlap_{eto.olap.n}_postfit_comparison_offset={self.best_fit_offset:.3f}.png')
            plt.close()
            
    def saveCostReduxes(self,dirpath) :
        cost_reduxes = []
        for eto in self.etos :
            orig_cost = self.__calcSingleCost(eto.p1_im,eto.p2_im)
            corr_p1im, corr_p2im = self.__correctImages(eto,self.best_fit_offset)
            corr_cost = self.__calcSingleCost(corr_p1im,corr_p2im)
            cost_reduxes.append((orig_cost-corr_cost)/(orig_cost))
        plt.hist(cost_reduxes,bins=60)
        plt.title('fractional cost reductions')
        with cd(dirpath) :
            plt.savefig(f'{self.raw_or_fw}_fit_{self.fitn}_cost_reductions.png')
        plt.close()
            
    def __correctImages(self,eto,offset) :
        corr_p1 = offset+(max_exp_times[0]/eto.p1et)*(eto.p1_im-offset)
        corr_p2 = offset+(max_exp_times[0]/eto.p2et)*(eto.p2_im-offset)
        return corr_p1, corr_p2
    
    def __calcSingleCost(self,p1im,p2im) :
        return np.sum(np.abs(p2im-p1im))/(p1im.shape[0]*p1im.shape[1])
        #return np.abs(np.mean(p2im)-np.mean(p1im))/(p1im.shape[0]*p1im.shape[1])

#first plot all of the exposure times
print('Plotting all exposure times....')
with cd(os.path.join(root2_dir,sample)) :
    all_rfps = [os.path.join(root2_dir,sample,fn) for fn in glob.glob('*.Data.dat')]
exp_times = {}
for rfp in all_rfps :
    exp_times[rfp]=getExposureTimesByLayer(rfp,35,root1_dir)[layer-1]
plt.hist([et for et in exp_times.values()])
plt.title(f'{sample} layer {layer} exposure times')
plt.xlabel('exposure time (ms)')
plt.ylabel('HPF count')
with cd(workingdir_name) :
    plt.savefig('all_exposure_times.png')
plt.close()

#get the max exposure times
max_exp_times = getSampleMaxExposureTimesByLayer(root1_dir,sample)

#make the alignmentset from the raw files
print('Making an AlignmentSet from the raw files....')
#a = AlignmentSetFromXML(root1_dir,root2_dir,sample,selectoverlaps=overlaps,onlyrectanglesinoverlaps=True,nclip=nclip,readlayerfile=False,layer=layer)
a = AlignmentSetFromXML(root1_dir,root2_dir,sample,nclip=nclip,readlayerfile=False,layer=layer)
a.getDAPI(filetype='raw')
#correct the rectangle images with the flatfield file
print('Correcting and updating rectangle images....')
flatfield_layer = (getRawAsHWL(flatfield_file,1004,1344,35,dtype=np.float64))[:,:,layer-1]
warp_images = []
for ri,r in enumerate(a.rectangles) :
    rfkey=r.file.rstrip('.im3')
    image = np.rint((r.image)/flatfield_layer).astype(np.uint16)
    warp_images.append(WarpImage(rfkey,cv2.UMat(image),cv2.UMat(np.empty_like(image)),False,ri))
a.updateRectangleImages(warp_images,usewarpedimages=False)
#align the overlaps
a.align(alreadyalignedstrategy='overwrite')
#make the exposure time comparison overlap objects
etolaps = []
for olap in a.overlaps :
    if olap.result.exit!=0 :
        continue
    p1et = exp_times[os.path.join(root2_dir,sample,(([r for r in a.rectangles if r.n==olap.p1])[0].file).replace('.im3','.Data.dat'))]
    p2et = exp_times[os.path.join(root2_dir,sample,(([r for r in a.rectangles if r.n==olap.p2])[0].file).replace('.im3','.Data.dat'))]
    if p2et-p1et!=0. :
        etolaps.append(ETOverlap(olap,p1et,p2et))
#sort the overlaps so those with the largest exposure time differences are first
print(f'Sorting list of {len(etolaps)} aligned overlaps with different exposure times....')
etolaps.sort(key=lambda x: abs(x.et_diff), reverse=True)
#save the overlay images for overlaps with the 20 greatest and 20 smallest differences in exposure time
print('Saving rawfile pre-correction overlap overlays....')
for io,eto in enumerate(etolaps[:min(20,len(etolaps))],start=1) :
    msg = f'exposure time diff. = {eto.et_diff:.2f}; sum(abs(p2-p1))/(total_pixels) = '
    msg+= f'{np.sum(np.abs(eto.p2_im-eto.p1_im))/(eto.p1_im.shape[0]*eto.p1_im.shape[1]):.2f}'
    img = eto.olap.getimage(normalize=1000.)
    plt.imshow(img)
    plt.title(msg)
    with cd(workingdir_name) :
        plt.savefig(f'raw_overlap_overlays_most_different_{io}.png')
    plt.close()
for io,eto in reversed(list(enumerate(etolaps[-min(20,len(etolaps)):],start=1))) :
    msg = f'exposure time diff. = {eto.et_diff:.2f}; sum(abs(p2-p1))/(total_pixels) = '
    msg+= f'{np.sum(np.abs(eto.p2_im-eto.p1_im))/(eto.p1_im.shape[0]*eto.p1_im.shape[1]):.2f}'
    img = eto.olap.getimage(normalize=1000.)
    plt.imshow(img)
    plt.title(msg)
    with cd(workingdir_name) :
        plt.savefig(f'raw_overlap_overlays_least_different_{abs(io-min(20,len(etolaps))-1)}.png')
    plt.close()
#shuffle the overlaps so they're not ordered
random.shuffle(etolaps)
#do a fit to half the overlaps
print('Doing first fit to raw file overlaps....')
fit_1 = Fit(etolaps[:int(len(etolaps)/2)],'raw','1')
fit_1.doFit(initial_offset=50)
f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
ax[0].plot(list(range(1,len(fit_1.costs)+1)),fit_1.costs,marker='*')
ax[0].set_title('costs')
ax[1].plot(list(range(1,len(fit_1.costs)+1)),fit_1.offsets,marker='*')
ax[1].set_title('offsets')
with cd(os.path.join(workingdir_name,raw_fit_1_dirname)) :
    plt.savefig(f'raw_fit_1_costs_and_offsets.png')
plt.close()
fit_1.saveCostReduxes(os.path.join(workingdir_name,raw_fit_1_dirname))
fit_1.saveCorrectedImages(25,os.path.join(workingdir_name,raw_fit_1_dirname))
#do a fit to the other half of the overlaps
print('Doing second fit to raw file overlaps....')
fit_2 = Fit(etolaps[int(len(etolaps)/2):],'raw','2')
fit_2.doFit(initial_offset=30)
f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
ax[0].plot(list(range(1,len(fit_2.costs)+1)),fit_2.costs,marker='*')
ax[0].set_title('costs')
ax[1].plot(list(range(1,len(fit_2.costs)+1)),fit_2.offsets,marker='*')
ax[1].set_title('offsets')
with cd(os.path.join(workingdir_name,raw_fit_2_dirname)) :
    plt.savefig(f'raw_fit_2_costs_and_offsets.png')
plt.close()
fit_2.saveCostReduxes(os.path.join(workingdir_name,raw_fit_2_dirname))
fit_2.saveCorrectedImages(25,os.path.join(workingdir_name,raw_fit_2_dirname))

#do all of the above except with an alignmentset made from the .fw01 files instead
print('Making an AlignmentSet from the .fw01 files....')
a = AlignmentSet(root1_dir,fw01_root2_dir,sample)
a.getDAPI(filetype='flatWarp')
a.align(write_result=False)
#make the exposure time comparison overlap objects
etolaps = []
for olap in a.overlaps :
    if olap.result.exit!=0 :
        continue
    p1et = exp_times[os.path.join(root2_dir,sample,(([r for r in a.rectangles if r.n==olap.p1])[0].file).replace('.im3','.Data.dat'))]
    p2et = exp_times[os.path.join(root2_dir,sample,(([r for r in a.rectangles if r.n==olap.p2])[0].file).replace('.im3','.Data.dat'))]
    if p2et-p1et!=0. :
        etolaps.append(ETOverlap(olap,p1et,p2et))
#sort the overlaps so those with the largest exposure time differences are first
etolaps.sort(key=lambda x: abs(x.et_diff), reverse=True)
#save the overlay images for overlaps with the 20 greatest and 20 smallest differences in exposure time
print('Saving .fw01 pre-correction overlay images....')
for io,eto in enumerate(etolaps[:min(20,len(etolaps))],start=1) :
    msg = f'exposure time diff. = {eto.et_diff:.2f}; sum(abs(p2-p1))/(total_pixels) = '
    msg+= f'{np.sum(np.abs(eto.p2_im-eto.p1_im))/(eto.p1_im.shape[0]*eto.p1_im.shape[1]):.2f}'
    img = eto.olap.getimage(normalize=1000.)
    plt.imshow(img)
    plt.title(msg)
    with cd(workingdir_name) :
        plt.savefig(f'fw_overlap_overlays_most_different_{io}.png')
    plt.close()
for io,eto in reversed(list(enumerate(etolaps[-min(20,len(etolaps)):],start=1))) :
    msg = f'exposure time diff. = {eto.et_diff:.2f}; sum(abs(p2-p1))/(total_pixels) = '
    msg+= f'{np.sum(np.abs(eto.p2_im-eto.p1_im))/(eto.p1_im.shape[0]*eto.p1_im.shape[1]):.2f}'
    img = eto.olap.getimage(normalize=1000.)
    plt.imshow(img)
    plt.title(msg)
    with cd(workingdir_name) :
        plt.savefig(f'fw_overlap_overlays_least_different_{abs(io-min(20,len(etolaps))-1)}.png')
    plt.close()
#shuffle the overlaps so they're not ordered
random.shuffle(etolaps)
#do a fit to half the overlaps
print('Doing first fit to .fw01 overlaps....')
fit_1 = Fit(etolaps[:int(len(etolaps)/2)],'fw','1')
fit_1.doFit(initial_offset=50)
f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
ax[0].plot(list(range(1,len(fit_1.costs)+1)),fit_1.costs,marker='*')
ax[0].set_title('costs')
ax[1].plot(list(range(1,len(fit_1.costs)+1)),fit_1.offsets,marker='*')
ax[1].set_title('offsets')
with cd(os.path.join(workingdir_name,fw_fit_1_dirname)) :
    plt.savefig(f'fw_fit_1_costs_and_offsets.png')
plt.close()
fit_1.saveCostReduxes(os.path.join(workingdir_name,fw_fit_1_dirname))
fit_1.saveCorrectedImages(25,os.path.join(workingdir_name,fw_fit_1_dirname))
#do a fit to the other half of the overlaps
print('Doing second fit to .fw01 overlaps....')
fit_2 = Fit(etolaps[int(len(etolaps)/2):],'fw','2')
fit_2.doFit(initial_offset=30)
f,ax=plt.subplots(1,2,figsize=(2*6.4,4.6))
ax[0].plot(list(range(1,len(fit_2.costs)+1)),fit_2.costs,marker='*')
ax[0].set_title('costs')
ax[1].plot(list(range(1,len(fit_2.costs)+1)),fit_2.offsets,marker='*')
ax[1].set_title('offsets')
with cd(os.path.join(workingdir_name,fw_fit_1_dirname)) :
    plt.savefig(f'fw_fit_2_costs_and_offsets.png')
plt.close()
fit_2.saveCostReduxes(os.path.join(workingdir_name,fw_fit_1_dirname))
fit_2.saveCorrectedImages(25,os.path.join(workingdir_name,fw_fit_1_dirname))

print('Done!')
