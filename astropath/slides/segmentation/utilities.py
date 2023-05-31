#imports
import re, ssl, requests
from urllib3 import poolmanager
import numpy as np, SimpleITK as sitk
from .config import SEG_CONST

def model_files_exist(model_dir_path=SEG_CONST.NNUNET_MODEL_DIR) :
    """
    Returns True if all of the pre-trained nnUNet model files exist and False otherwise
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for filepath in SEG_CONST.NNUNET_MODEL_FILES :
        check_filepath = model_dir_path/(filepath.relative_to(SEG_CONST.NNUNET_MODEL_DIR))
        if not check_filepath.is_file() :
            return False
    return True

class TLSAdapter(requests.adapters.HTTPAdapter):
    """
    Adapter needed to connect to the sciserver public website, see StackOverflow question about the error here:
    https://stackoverflow.com/questions/61631955/python-requests-ssl-error-during-requests
    """

    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=0')
        ctx.check_hostname = False
        self.poolmanager = poolmanager.PoolManager(
                num_pools=connections,
                maxsize=maxsize,
                block=block,
                ssl_version=ssl.PROTOCOL_TLS,
                ssl_context=ctx)

def download_model_files(model_dir_path=SEG_CONST.NNUNET_MODEL_DIR,logger=None) :
    """
    Downloads any missing pre-trained nnUNet model files from the public SciServer link
    """
    session = None
    for filepath in SEG_CONST.NNUNET_MODEL_FILES :
        check_filepath = model_dir_path/(filepath.relative_to(SEG_CONST.NNUNET_MODEL_DIR))
        if not check_filepath.is_file() :
            if not check_filepath.parent.is_dir() :
                check_filepath.parent.mkdir(parents=True)
            if session is None :
                session = requests.sessions.Session()
                session.verify=False
                session.mount('https://', TLSAdapter())
            dl_url = f'{SEG_CONST.NNUNET_MODEL_FILES_URL}{filepath.relative_to(SEG_CONST.NNUNET_MODEL_DIR).as_posix()}'
            if logger is not None :
                logger.debug(f'Downloading {check_filepath} from {dl_url}...')
            success = False
            n_retries = 2
            exc = None
            while (not success) and n_retries>0 :
                try :
                    resp = session.get(dl_url,verify=False)
                    with open(check_filepath,'wb') as wfp :
                        wfp.write(resp.content)
                    success = True
                    if logger is not None :
                        logger.debug(f'Done downloading {check_filepath}')
                except Exception as e :
                    n_retries-=1
                    exc = e
            if not success :
                msg = f'ERROR: {check_filepath} failed to be downloaded from {dl_url}! Exception will be reraised.'
                if logger is not None :
                    logger.error(msg)
                else :
                    print(msg)
                if session is not None :
                    try :
                        session.close()
                    except :
                        pass
                if exc is not None :
                    raise exc
    if session is not None :
        try :
            session.close()
        except :
            pass

def download_model_files_if_necessary(model_dir_path=SEG_CONST.NNUNET_MODEL_DIR,logger=None) :
    """
    Calls two functions above to make sure that the required model files exist
    """
    if not model_files_exist(model_dir_path) :
        download_model_files(model_dir_path,logger=logger)
    if not model_files_exist(model_dir_path) :
        raise FileNotFoundError(f'ERROR: failed to download nnUNet files to expected locations in {model_dir_path}!')

def write_nifti_file_for_rect_im(im,nifti_file_path) :
    """
    Convert a given rectangle image to the NIfTI format needed by nnU-Net and write it out
    """
    img = im[:,:,np.newaxis]
    img = img.transpose(2,0,1)
    itk_img = sitk.GetImageFromArray(img)
    itk_img.SetSpacing([1,1,999])
    sitk.WriteImage(itk_img, str(nifti_file_path))
    assert nifti_file_path.is_file()

def convert_nnunet_output(segmented_nifti_path,segmented_file_path) :
    """
    Convert the NIfTI output from nnU-Net to a compressed numpy file where 0=background, 1=boundary, and 2=nucleus
    """
    itk_read_img = sitk.ReadImage(str(segmented_nifti_path),imageIO='NiftiImageIO')
    output_img = np.zeros((itk_read_img.GetHeight(),itk_read_img.GetWidth()),dtype=np.float32)
    for ix in range(output_img.shape[1]) :
        for iy in range(output_img.shape[0]) :
            output_img[iy,ix] = itk_read_img.GetPixel((ix,iy,0))
    output_img[output_img>1] = 2
    output_img = output_img.astype(np.uint8)
    np.savez_compressed(segmented_file_path,output_img)
    assert segmented_file_path.is_file()

def run_deepcell_nuclear_segmentation(batch_ims,app,pscale,batch_segmented_file_paths) :
    """
    Run DeepCell nuclear segmentation for a given batch of images with a given application and write out the output
    """
    labeled_batch_ims = app.predict(batch_ims,image_mpp=1./pscale)
    for bi in range(batch_ims.shape[0]) :
        labeled_img = labeled_batch_ims[bi,:,:,0]
        np.savez_compressed(batch_segmented_file_paths[bi],labeled_img)
    for bi in range(batch_ims.shape[0]) :
        assert batch_segmented_file_paths[bi].is_file()

def run_mesmer_segmentation(batch_ims,app,pscale,batch_segmented_file_paths) :
    """
    Run Mesmer whole-cell and nuclear segmentations for a given batch of images 
    with a given application and write out the output
    """
    labeled_batch_ims = app.predict(batch_ims,image_mpp=1./pscale,compartment='both')
    for bi in range(batch_ims.shape[0]) :
        labeled_img = labeled_batch_ims[bi,:,:,:]
        np.savez_compressed(batch_segmented_file_paths[bi],labeled_img)
    for bi in range(batch_ims.shape[0]) :
        assert batch_segmented_file_paths[bi].is_file()

def initialize_app(appcls, *args, ntries=5, logger=None, **kwargs):
    try:
        return appcls(*args, **kwargs)
    except Exception as e:
        if ntries <= 1: raise
        errno = None
        try:
            errno = e.errno
        except AttributeError:
            match = re.search(r"\[Errno ([0-9-]+)\]", str(e))
            if match:
                errno = int(match.group(1))
        retry = False
        if errno == -3:  #Temporary failure in name resolution in aws download
            retry = True
        if retry:
            if logger is not None:
                logger.debug(f"initializing {appcls.__name__} failed")
                logger.debug(str(e))
                logger.debug("trying again")
            return initialize_app(appcls, *args, ntries=ntries-1, **kwargs)
        raise
