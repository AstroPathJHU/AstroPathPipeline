#class for shared constant variables
class Const :
    #file extensions
    @property
    def RAW_EXT(self) :
        return '.Data.dat' # extension of completely raw image files
    @property
    def WARP_EXT(self) :
        return '.camWarp_layer' #extension stem for the warped files
    @property
    def FW_EXT(self) :
        return '.fw' #extension stem for the flatfielded/warped files
    #other image properties
    @property
    def CORNER_OVERLAP_TAGS(self):
        return [1,3,7,9] #list of tags representing overlaps that are corners 
    #files that get created
    @property 
    def OCTET_OVERLAP_CSV_FILE_NAMESTEM(self) :
        return '_overlap_octets.csv' #stem for the name of the octet overlap csv file name
    
CONST=Const()
