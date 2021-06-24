#imports
from ...shared.logging import dummylogger
import numpy as np

class Flatfield :
	"""
	Class representing the flatfield model for a group of slides
	The model is created by reading the meanimages and other data for each slide
	"""

	#################### PUBLIC FUNCTIONS ####################

	def __init__(self,logger=dummylogger) :
		"""
        logger = the logging object to use (passed from whatever is using this meanimage)
        """
        self.__image_stack = None
        self.__image_squared_stack = None
        self.__mask_stack = None
        self.__flatfield_image = None

    def add_sample(self,sample) :
    	"""
		Add a given sample's information to the model
    	"""
    	pass

    def create_flatfield_model(self) :
    	"""
		After all the samples have been added, this method creates the actual flatfield object
    	"""
    	pass

    def write_output(self,batchID,workingdirpath) :
    	"""
		Write out the flatfield image and all other output

		batchID = the batchID to use for the model (in filenames, etc.)
		workingdirpath = path to the directory where the output should be saved (the actuall flatfield is saved in this directory's parent)
    	"""
    	pass
