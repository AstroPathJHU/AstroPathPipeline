#imports
import shutil
import numpy as np, SimpleITK as sitk
from hashlib import sha512
from skimage.segmentation import find_boundaries
from .config import SEG_CONST

def split_model_files(model_dir_path=SEG_CONST.NNUNET_MODEL_TOP_DIR,
                      bytes_per_chunk=50000000,remove=True) :
    """
    Recursively search a directory for any "model_final_checkpoint.model" files 
    and split them into smaller chunks so they can be uploaded as part of the GitHub repo

    model_dir_path = the top directory beneath which anything called "model_final_checkpoint.model" should be split
    bytes_per_chunk = the maximum number of bytes that should be included in each individual chunk
    remove = True if the original, single, large "model_final_checkpoint.model" files should be removed 
        once the split files are all created
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for fp in model_dir_path.rglob('*') :
        if fp.name=='model_final_checkpoint.model' :
            ref_hash = sha512()
            with open(fp,'rb') as ifpo :
                ref_hash.update(ifpo.read())
            ref_hash = ref_hash.digest()
            new_hash_1 = sha512()
            created_fns = []
            with open(fp,'rb') as ifpo :
                chunk_i = 0
                new_bytes = ifpo.read(bytes_per_chunk)
                while len(new_bytes)>0 :
                    new_hash_1.update(new_bytes)
                    new_fn = f'{fp.name}_chunk_{chunk_i}'
                    created_fns.append(new_fn)
                    with open(fp.parent/new_fn,'wb') as ofpo :
                        ofpo.write(new_bytes)
                    chunk_i+=1
                    new_bytes = ifpo.read(bytes_per_chunk)
            new_hash_1 = new_hash_1.digest()
            new_hash_2 = sha512()
            if remove :
                for new_fn in created_fns :
                    if not (fp.parent/new_fn).is_file() :
                        remove = False
                        break
                    else :
                        with open(fp.parent/new_fn,'rb') as newfp :
                            new_hash_2.update(newfp.read())
            new_hash_2 = new_hash_2.digest()
            if not ref_hash==new_hash_1==new_hash_2 :
                errmsg = f'ERROR: hashes of original {fp} file and individual chunks are mismatched! '
                errmsg+= 'Original file will not be removed.'
                raise RuntimeError(errmsg)
            if remove :
                fp.unlink()

def assemble_model_files(model_dir_path=SEG_CONST.NNUNET_MODEL_TOP_DIR,remove=False) :
    """
    Recursively search a directory for any "fold_n" subdirectories and, if those subdirectories 
    contain any "model_final_checkpoint.model_chunk_n" files, assemble the individual files into 
    a single larger "model_final_checkpoint.model" file

    model_dir_path = the top directory beneath which any "fold_n" subdirectories should be searched for 
        "model_final_checkpoint.model_chunk_n" files to combine
    remove = True if the smaller individual files should be removed once they've been successfully rebuilt 
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for fold_dir in model_dir_path.rglob('fold_*') :
        if not (fold_dir/'model_final_checkpoint.model_chunk_0').is_file() :
            continue
        newfp = fold_dir/'model_final_checkpoint.model'
        ref_hash = sha512()
        with open(newfp,'wb') as newf :
            for fp in sorted(fold_dir.glob('model_final_checkpoint.model_chunk_*')) :
                with open(fp,'rb') as oldf :
                    shutil.copyfileobj(oldf,newf)
                with open(fp,'rb') as oldf :
                    ref_hash.update(oldf.read())
        ref_hash = ref_hash.digest()
        new_hash = sha512()
        if newfp.is_file() :
            with open(newfp,'rb') as newf :
                new_hash.update(newf.read())
        new_hash = new_hash.digest()
        if not ref_hash==new_hash :
            errmsg = f'ERROR: hashes of re-assembled {newfp} file and individual chunks are mismatched! '
            errmsg+= 'Individual chunk files will not be removed.'
            raise RuntimeError(errmsg)
        if remove :
            for fp in fold_dir.glob('model_final_checkpoint.model_chunk_*') :
                fp.unlink()

def model_files_exist(model_dir_path=SEG_CONST.NNUNET_MODEL_TOP_DIR) :
    """
    Returns True if every "model_final_checkpoint.model" file exists and False otherwise
    """
    if not model_dir_path.is_dir() :
        raise FileNotFoundError(f'ERROR: {model_dir_path} is not a directory!')
    for fold_dir in model_dir_path.rglob('fold_*') :
        if not (fold_dir/'model_final_checkpoint.model').is_file() :
            return False
    return True

def rebuild_model_files_if_necessary(model_dir_path=SEG_CONST.NNUNET_MODEL_TOP_DIR,remove=False) :
    """
    Calls two functions above to make sure that the required model files exist
    """
    if not model_files_exist(model_dir_path) :
        assemble_model_files(model_dir_path,remove)

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

def run_deepcell_nuclear_segmentation(im,app,pscale,segmented_file_path) :
    """
    Run DeepCell nuclear segmentation for a given image with a given application and write out the output
    """
    img = np.expand_dims(im,axis=-1)
    img = np.expand_dims(img,axis=0)
    labeled_img = app.predict(img,image_mpp=1./pscale)
    labeled_img = labeled_img[0,:,:,0]
    boundaries = find_boundaries(labeled_img)
    output_img = np.zeros(labeled_img.shape,dtype=np.uint8)
    output_img[labeled_img!=0] = 2
    output_img[boundaries] = 1
    np.savez_compressed(segmented_file_path,output_img)
    assert segmented_file_path.is_file()

def run_mesmer_segmentation(batch_ims,app,pscale,batch_segmented_file_paths) :
    """
    Run Mesmer whole-cell and nuclear segmentationss for a given batch of images 
    with a given application and write out the output
    """
    labeled_batch_ims = app.predict(batch_ims,image_mpp=1./pscale,compartment='both')
    for bi in range(batch_ims.shape[0]) :
        labeled_img = labeled_batch_ims[bi,:,:,:]
        np.savez_compressed(batch_segmented_file_paths[bi],labeled_img)
    for bi in range(batch_ims.shape[0]) :
        assert batch_segmented_file_paths[bi].is_file()
    pass
