class Const :
    @property
    def OCTET_SUBDIR_NAME(self) :
        return 'octets' #name of the subdirectory in a cohort's warping directory that 
                        #contains the lists of octets for every sample 
    @property
    def OCTET_FILENAME_STEM(self) :
        return 'all_overlap_octets.csv' #stem of the filename that holds the list of 
                                                     #a sample's warping octets to use
    @property
    def REQ_OVERLAP_PIXEL_FRAC(self) :
        return 0.85 #the required fraction of pixels whose intensities must be above the background 
                    #threshold for an overlap to be used for fitting

CONST = Const()