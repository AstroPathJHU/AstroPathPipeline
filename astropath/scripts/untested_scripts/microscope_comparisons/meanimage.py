#imports
import pathlib, tifffile
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
from astropath.utilities.img_file_io import get_raw_as_hwl, write_image_to_file
from astropath.shared.image_masking.image_mask import ImageMask

#constants
DEF_COMP_TIFF_DIR=pathlib.Path('//bki-fs1.idies.jhu.edu/data02/Microscope_Comparison/Round_1/Component_Tiffs')
DEF_MASK_DIR=pathlib.Path('//bki07/')
DEF_OUTPUT_DIR=pathlib.Path('//bki-fs1.idies.jhu.edu/data02/maggie/microscope_comparison/mean_images')
DEF_N_PROCS = 8
#IMAGE_DIMS = (1404, 1876, 10)
IMAGE_DIMS = (1404, 1872, 10)

def write_output(output_dir,slideID,mean_image,mask_stack) :
    """
    Write out the mean image and the mask stack to the given location
    """
    print(f'Writing output to {output_dir}')
    if not output_dir.is_dir() :
        output_dir.mkdir(parents=True)
    write_image_to_file(mean_image,output_dir/f'{slideID}-mean_image.bin')
    write_image_to_file(mask_stack,output_dir/f'{slideID}-mask_stack.bin')

def get_mean_image(sum_of_masked_images,mask_stack) :
    """
    Return the mean image calculated from the given stack of masked images and the mask stack
    """
    zero_fixed_mask_stack = np.copy(mask_stack)
    zero_fixed_mask_stack[zero_fixed_mask_stack==0] = np.min(zero_fixed_mask_stack[zero_fixed_mask_stack!=0])
    mean_image = sum_of_masked_images/zero_fixed_mask_stack[:,:,np.newaxis]
    return mean_image

def queue_masked_image_and_mask(tiff_path,mask_path,masked_image_queue,mask_queue) :
    """
    Read the component tiff from the filepath and its corresponding tissue mask.
    Mask the component tiff image and add the masked image and the mask to their
    respective queues to be summed in other processes.
    """
    mask_path = '\\'+str(mask_path)
    tissue_mask = ImageMask.unpack_tissue_mask(mask_path,IMAGE_DIMS[:-1])
    mask_queue.put(tissue_mask)
    image = np.empty(IMAGE_DIMS,np.float32)
    with tifffile.TiffFile(tiff_path) as tif :
        for ip,page in enumerate(tif.pages[:10]) :
            image[:,:,ip] = page.asarray()
    masked_image = tissue_mask[:,:,np.newaxis]*image
    masked_image_queue.put(masked_image)

def sum_from_queue(queue) :
    """
    Sum a queue of numpy arrays together. Return the sum in the same queue after pulling None.
    """
    sum_array = None
    next_item = queue.get()
    while next_item is not None :
        if sum_array is None :
            if next_item.dtype==np.uint8 :
                sum_array = np.zeros(next_item.shape,dtype=np.uint64)
            else :
                sum_array = np.zeros(next_item.shape,dtype=np.float64)
        sum_array+=next_item
        next_item=queue.get()
    queue.put(sum_array)

def get_tissue_mask_path_for_comp_tiff_path(mask_dir,fp) :
    """
    Return the path to the tissue mask file corresponding to a given component tiff path
    """
    image_stem = fp.name[:-len('_component_data.tif')]
    tissue_mask_file_name = f'{image_stem}_tissue_mask.bin'
    tissue_mask_path = mask_dir/tissue_mask_file_name
    return tissue_mask_path

def sum_component_tiffs(slideID,comp_tiff_dir,mask_dir,nprocs) :
    """
    Use multiple processes to read component tiff images and masks, mask the component tiff images, 
    and append the mask and masked image to queues for summing in different processes
    """
    #figure out how many images there are and make sure all of their masks exist
    n_images = 0
    for fp in comp_tiff_dir.glob(f'{slideID}_*_component_data.tif') :
        tissue_mask_file_path = get_tissue_mask_path_for_comp_tiff_path(mask_dir,fp)
        if not tissue_mask_file_path.is_file() :
            raise FileNotFoundError(f'ERROR: could not find expected tissue mask {tissue_mask_file_path}!')
        n_images+=1
    if n_images<=0 :
        raise RuntimeError(f'Found {n_images} unmixed images in {comp_tiff_dir}')
    print(f'Found {n_images} total unmixed images to mask and sum for {slideID}')
    #start up the process to handle the sum of masked images
    m = mp.Manager()
    masked_image_queue = m.Queue(nprocs)
    sum_masked_images_proc = mp.Process(target=sum_from_queue,args=(masked_image_queue,))
    sum_masked_images_proc.start()
    print(f'Started sum masked images process for {slideID}')
    #start up the process to handle the sum of masks
    mask_queue = m.Queue(nprocs)
    sum_masks_proc = mp.Process(target=sum_from_queue,args=(mask_queue,))
    sum_masks_proc.start()
    print(f'Started sum masks process for {slideID}')
    read_procs = []
    for ifp,fp in enumerate(comp_tiff_dir.glob(f'{slideID}_*_component_data.tif'),start=1) :
        while len(read_procs)>=nprocs-2 :
            p = read_procs.pop(0)
            p.join(0.1)
            if p.is_alive() :
                read_procs.append(p)
        pct = 100.*ifp/n_images
        print(f'Reading component tiff and image mask for {fp.name} ({ifp}/{n_images}, {pct:.1f}%)')
        tissue_mask_file_path = get_tissue_mask_path_for_comp_tiff_path(mask_dir,fp)
        new_p = mp.Process(target=queue_masked_image_and_mask,
                           args=(fp,
                                 tissue_mask_file_path,
                                 masked_image_queue,
                                 mask_queue))
        new_p.start()
        read_procs.append(new_p)
    print('Joining all read processes...')
    for p in read_procs :
        p.join()
    masked_image_queue.put(None)
    mask_queue.put(None)
    print(f'finishing summing masks...')
    sum_masks_proc.join()
    print(f'finishing summing masked images...')
    sum_masked_images_proc.join()
    print(f'Getting sum of masked images and mask stack...')
    sum_masked_images = masked_image_queue.get()
    mask_stack = mask_queue.get()
    return sum_masked_images, mask_stack

def get_arguments(args=None) :
    """
    Get the command line arguments to use
    """
    #parse the arguments
    parser = ArgumentParser()
    parser.add_argument('slideID',help='ID of the slide to correct')
    parser.add_argument('--comp_tiff_dir',type=pathlib.Path,default=DEF_COMP_TIFF_DIR,
        help=f"path to the directory to read component tiff input from (default = {DEF_COMP_TIFF_DIR/'Polaris_#'})")
    parser.add_argument('--mask_dir',type=pathlib.Path,default=DEF_MASK_DIR,
        help=f'''path to the directory to read image masks from 
                 (default = {DEF_MASK_DIR/"JHU_Polaris_#"/"slideID"/"im3"/"meanimage"/"image_masking"})''')
    parser.add_argument('--output_dir',type=pathlib.Path,default=DEF_OUTPUT_DIR,
        help=f'path to the directory to put output in (default = {DEF_OUTPUT_DIR/"slideID"})')
    parser.add_argument('--nprocs',type=int,default=DEF_N_PROCS,
        help=f'Number of parallel processes to use for writing out corrected files (default = {DEF_N_PROCS})')
    args = parser.parse_args(args)
    #adjust some of them for defaults
    if args.nprocs<4 :
        print(f'WARNING: setting number of processes to 4 (the minimum) instead of {args.nprocs}')
        args.nprocs = 4
    polaris_number = None 
    if args.slideID.startswith('AP018') :
        polaris_number = 1
    elif args.slideID.startswith('AP017') :
        polaris_number = 2
    elif args.slideID.startswith('AP019') :
        polaris_number = 3
    if polaris_number not in (1,2,3) :
        raise ValueError(f'ERROR: could not determine polaris number from slide ID {args.SlideID}')
    if args.comp_tiff_dir==DEF_COMP_TIFF_DIR :
        args.comp_tiff_dir = args.comp_tiff_dir/f'Polaris{polaris_number}'
    if args.mask_dir==DEF_MASK_DIR :
        args.mask_dir = args.mask_dir/f'JHU_Polaris_{polaris_number}'/args.slideID/'im3'/'meanimage'/'image_masking'
    if args.output_dir==DEF_OUTPUT_DIR :
        args.output_dir = args.output_dir/args.slideID
    #print out the paths
    print(f'Will read component tiffs from {args.comp_tiff_dir}')
    print(f'Will read tissue masks from {args.mask_dir}')
    print(f'Will write output to {args.output_dir}')
    #return them
    return args

def main(args=None) :
    #get the arguments
    args = get_arguments(args)
    #add all of the masks and masked images
    sum_masked_images, mask_stack = sum_component_tiffs(args.slideID,
                                                        args.comp_tiff_dir,
                                                        args.mask_dir,
                                                        args.nprocs)
    #calculate the mean image
    mean_image = get_mean_image(sum_masked_images,mask_stack)
    #write out the mean image and the mask stack 
    write_output(args.output_dir,args.slideID,mean_image,mask_stack)
    print('Done : )')

if __name__=='__main__' :
    main()