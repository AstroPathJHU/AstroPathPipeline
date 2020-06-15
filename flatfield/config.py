#class for shared constant variables
class Const :
    #final overall outputs
    @property
    def FILE_EXT(self) :
        return '.bin' #file extension for the main output files
    @property
    def FLATFIELD_FILE_NAME_STEM(self) :
        return 'flatfield' #what the flatfield file is called
    @property
    def THRESHOLD_TEXT_FILE_NAME_STEM(self) :
        return 'background_thresholds.txt' #name of the text file holding each layer's background threshold flux
    #image smoothing
    @property
    def GENTLE_GAUSSIAN_SMOOTHING_SIGMA(self) :
        return 5 #the sigma, in pixels, of the gentle gaussian smoothing applied to images before thresholding/masking

CONST=Const()
