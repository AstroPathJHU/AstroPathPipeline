#imports
from ...utilities.dataclasses import MyDataClass

#dataclass to organize entries in the correction model .csv file
class CorrectionModelTableEntry(MyDataClass) :
    SlideID : str
    Project : int
    Cohort  : int
    BatchID : int
    FlatfieldVersion : str
    WarpingFile : str
    