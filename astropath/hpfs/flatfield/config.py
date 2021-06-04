#class for shared constant variables
class Const :
    @property
    def MEANIMAGE_DIRNAME(self) :
        return 'meanimage' #name of the output directory for slidemeanimage results
    @property
    def IMAGE_MASKING_SUBDIR_NAME(self) :
        return 'image_masking' #name of the image masking subdirectory in the workingdirectory (if masking is run)
    @property
    def FIELDS_USED_CSV_FILENAME(self) :
        return 'fields_used.csv' #name of the .csv file listing every HPF used to make a meanimage/flatfield model
    @property
    def METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME(self) :
        return 'metadata_summary_stacked_images.csv' #name of the .csv file giving the metadata summary for the HPFs that were stacked 
                                                     #to make the meanimage/flatfield model
    @property
    def LABELLED_MASK_REGIONS_CSV_FILENAME(self) :
        return 'labelled_mask_regions.csv' #name of the .csv file listing information about every region of every HPF masked due to blur or saturation 
                                           #(like a key for the mask files)
    @property
    def BLUR_AND_SATURATION_MASK_FILE_NAME_STEM(self) :
        return '_full_mask.bin' #end of the filename for the blur and saturation mask files
    @property
    def TISSUE_MASK_FILE_NAME_STEM(self) :
        return '_tissue_mask.bin' #end of the filename for the blur and saturation mask files

CONST=Const()