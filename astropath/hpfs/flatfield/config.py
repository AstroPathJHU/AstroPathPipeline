#class for shared constant variables
class Const :
    @property
    def DEFAULT_N_THREADS(self) :
        return 10 #default number of threads to use for parallelized portions of the code
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
    def MASK_STACK_BIN_FILE_NAME_STEM(self) :
        return 'mask_stack.bin' #suffix to the name of the mask stack .bin file
    @property
    def BACKGROUND_THRESHOLD_CSV_FILE_NAME_STEM(self) :
        return 'background_thresholds.csv' #suffix to name of the background threshold .txt file created by meanimagesample
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
    @property
    def MASKING_SUMMARY_PDF_FILENAME(self) :
        return 'masking_summary.pdf' #name of the .pdf file containing plots of how the image masking worked for this sample
    @property
    def MIN_PIXEL_FRAC(self) :
        return 0.8 #minimum fraction of pixels that must be "good tissue" in a given image to add it to an image stack
    @property
    def MEANIMAGE_SUMMARY_PDF_FILENAME(self) :
        return 'meanimage_summary.pdf' #name of the .pdf file containing plots of layers of the meanimage for this sample, etc.
    @property
    def FLATFIELD_WIDE_GAUSSIAN_FILTER_SIGMA(self) :
        return 100 #Sigma (in pixels) for the wide Gaussian filter applied to smooth the meanimage when creating a flatfield model
    @property
    def FLATFIELD_DIRNAME_STEM(self) :
        return 'flatfield_BatchID_' #prepend for name of the directory holding batch flatfield results (also used for some other image/plot names)
    @property
    def FLATFIELD_SUMMARY_PDF_FILENAME_STEM(self) :
        return 'flatfield_summary' #prepend for the name of the batch flatfield summary pdf file
    @property
    def APPLIED_FLATFIELD_SUMMARY_PDF_FILENAME_STEM(self) :
        return 'applied_flatfield_summary' #prepend for the name of the applied flatfield summary pdf file

CONST=Const()