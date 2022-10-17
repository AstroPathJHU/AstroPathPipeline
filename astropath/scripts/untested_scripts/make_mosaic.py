""" A script to create a mosaic flatfield image given a single tile """

#imports
import pathlib, numpy as np
from argparse import ArgumentParser
from astropath.utilities.img_file_io import get_raw_as_hwl, write_image_to_file

#some constants
TILE_DIMENSIONS = (1004,1344,35)
MOSAIC_DIMENSIONS = (3008,4028,35)
HORIZONTAL_EDGES = (1008,2016)
VERTICAL_EDGES = (1344,2688)
IMG_DTYPE = np.float64

def main() :
    #create the argument parser and get the command line arguments
    parser = ArgumentParser()
    parser.add_argument('input_file', type=pathlib.Path,
                        help='Path to the input file: a single tile flatfield image')
    parser.add_argument('--output_file', type=pathlib.Path, 
                        help='(optional) path to the output file: the full mosaic flatfield image')
    args = parser.parse_args()
    #make sure the input file exists
    if not args.input_file.is_file() :
        raise FileNotFoundError(f'ERROR: {args.input_file} not found!')
    #set the output file path
    output_file = args.output_file
    if output_file is None :
        output_file = args.input_file.parent / f'{args.input_file.name[:-len(".bin")]}_mosaic.bin'
    #load the tile image and create the mosaic image
    print(f'loading tile image from {args.input_file}....')
    tile_image = get_raw_as_hwl(args.input_file,*TILE_DIMENSIONS,IMG_DTYPE)
    print(f'creating placeholder output image....')
    mosaic_image = np.empty(MOSAIC_DIMENSIONS,dtype=IMG_DTYPE)
    #build the mosaic image from the tiles
    print('creating mosaic image from tiles....')
    slices = {
        #slices for replacing the entire tile image (upper left four grid sections)
        'four upper left grid sections' : {
            'tileslice' : np.s_[:,:,:],
            'mosaicslices' : [
                np.s_[:TILE_DIMENSIONS[0],:VERTICAL_EDGES[0],:],
                np.s_[:TILE_DIMENSIONS[0],VERTICAL_EDGES[0]:VERTICAL_EDGES[1],:],
                np.s_[HORIZONTAL_EDGES[0]:HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0],:VERTICAL_EDGES[0],:],
                np.s_[HORIZONTAL_EDGES[0]:HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0],VERTICAL_EDGES[0]:VERTICAL_EDGES[1],:],
            ],
        },
        #slices for the extra bottom edges in the upper left four grid sections
        'bottom edges of upper left sections' : {
            'tileslice' : np.s_[-(HORIZONTAL_EDGES[0]-TILE_DIMENSIONS[0]):,:,:],
            'mosaicslices' : [
                np.s_[TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[0],:VERTICAL_EDGES[0],:],
                np.s_[TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[0],VERTICAL_EDGES[0]:VERTICAL_EDGES[1],:],
                np.s_[HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[1],:VERTICAL_EDGES[0],:],
                np.s_[HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[1],VERTICAL_EDGES[0]:VERTICAL_EDGES[1],:],
            ],
        },
        #slices for the upper two right column tiles, with the right edges cut off slightly
        'upper two right column sections' : {
            'tileslice' : np.s_[:,:MOSAIC_DIMENSIONS[1]-VERTICAL_EDGES[1],:],
            'mosaicslices' : [
                np.s_[:TILE_DIMENSIONS[0],VERTICAL_EDGES[1]:,:],
                np.s_[HORIZONTAL_EDGES[0]:HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0],VERTICAL_EDGES[1]:,:],
            ],
        },
        #slices for the extra bottom edges in the upper two right column tiles, with the right edges cut off
        'bottom edges of upper right column tiles' : {
            'tileslice' : np.s_[-(HORIZONTAL_EDGES[0]-TILE_DIMENSIONS[0]):,:MOSAIC_DIMENSIONS[1]-VERTICAL_EDGES[1],:],
            'mosaicslices' : [
                np.s_[TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[0],VERTICAL_EDGES[1]:,:],
                np.s_[HORIZONTAL_EDGES[0]+TILE_DIMENSIONS[0]:HORIZONTAL_EDGES[1],VERTICAL_EDGES[1]:,:],
            ],
        },
        #slices for the left two bottom row tiles, with the bottoms cut off slightly
        'left two bottom row sections' : {
            'tileslice' : np.s_[:MOSAIC_DIMENSIONS[0]-HORIZONTAL_EDGES[1],:,:],
            'mosaicslices' : [
                np.s_[HORIZONTAL_EDGES[1]:,:VERTICAL_EDGES[0],:],
                np.s_[HORIZONTAL_EDGES[1]:,VERTICAL_EDGES[0]:VERTICAL_EDGES[1],:],
            ],
        },
        #slice for the bottom right corner tile (bottom and right both cut off slightly)
        'bottom right corner tile' : {
            'tileslice' : np.s_[:MOSAIC_DIMENSIONS[0]-HORIZONTAL_EDGES[1],:MOSAIC_DIMENSIONS[1]-VERTICAL_EDGES[1],:],
            'mosaicslices' : [
                np.s_[HORIZONTAL_EDGES[1]:,VERTICAL_EDGES[1]:,:],
            ],
        },
    }
    for slicename in slices :
        print(f'    replacing the {slicename}....')
        for mosaic_slice in slices[slicename]['mosaicslices'] :
            mosaic_image[mosaic_slice] = tile_image[slices[slicename]['tileslice']]
    #write out the mosaic image
    print(f'Writing output mosaic image to {output_file}....')
    write_image_to_file(mosaic_image,output_file)
    print('Done!')

if __name__=='__main__' :
    main()
