#class for shared constant variables
class Const :
    @property
    def IMAGE_MASKING_SUBDIR_NAME(self) :
        return 'image_masking' #name of the image masking subdirectory in the workingdirectory (if masking is run)
    @property
    def FIELDS_USED_CSV_FILENAME(self) :
        return 'fields_used.csv' #name of the .csv file listing every HPF used to make a meanimage/flatfield model
    @property
    def MEAN_IMAGE_BIN_FILE_NAME_STEM(self) :
        return 'mean_image.bin' #suffix to name of the meanimage .bin file created by meanimagesample
    @property
    def SUM_IMAGES_SQUARED_BIN_FILE_NAME_STEM(self) :
        return 'sum_images_squared.bin' #suffix to name of the sum of images squared .bin file created by meanimagesample
    @property
    def STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM(self) :
        return 'std_err_of_mean_image.bin' #suffix to name of the standard error of the meanimage .bin file created by meanimagesample
    @property
    def BACKGROUND_THRESHOLD_TEXT_FILE_NAME_STEM(self) :
        return 'background_thresholds.txt' #suffix to name of the background threshold .txt file created by meanimagesample
    @property
    def METADATA_SUMMARY_STACKED_IMAGES_CSV_FILENAME(self) :
        return 'metadata_summary_stacked_images.csv' #name of the .csv file giving the metadata summary for the HPFs that were stacked 
                                                     #to make the meanimage/flatfield model
    @property
    def METADATA_SUMMARY_THRESHOLDING_IMAGES_CSV_FILENAME(self) :
        return 'metadata_summary_thresholding_images.csv' #name of the .csv file giving the metadata summary for the HPFs that were used 
                                                          #to find the optimal background thresholds
    @property
    def THRESHOLDING_DATA_TABLE_CSV_FILENAME(self) :
        return 'thresholding_data_table.csv' #name of the .csv file listing optimal background thresholds found for individual images
    @property
    def THRESHOLDING_SUMMARY_PDF_FILENAME(self) :
        return 'thresholding_summary.pdf' #name of the .pdf file containing plots of how the thresholding algorithm worked for this sample
    @property
    def LABELLED_MASK_REGIONS_CSV_FILENAME(self) :
        return 'labelled_mask_regions.csv' #name of the .csv file listing information about every region of every HPF masked due to blur or saturation 
                                           #(like a key for the mask files)
    @property
    def BLUR_AND_SATURATION_MASK_FILE_NAME_STEM(self) :
        return 'full_mask.bin' #end of the filename for the blur and saturation mask files
    @property
    def TISSUE_MASK_FILE_NAME_STEM(self) :
        return 'tissue_mask.bin' #end of the filename for the blur and saturation mask files

CONST=Const()