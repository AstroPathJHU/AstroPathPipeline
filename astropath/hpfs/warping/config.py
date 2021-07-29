class Const :
    @property
    def OCTET_SUBDIR_NAME(self) :
        return 'octets' #name of the subdirectory in a cohort's warping directory that 
                        #contains the lists of octets for every sample 
    @property
    def OCTET_FILENAME_STEM(self) :
        return 'overlap_octets_for_warping_fits.csv' #stem of the filename that holds the list of 
                                                     #a sample's warping octets to use

CONST = Const()