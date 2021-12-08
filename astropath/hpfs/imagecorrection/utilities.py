#imports
from ...utilities.dataclasses import MyDataClass

#dataclass to organize entries in the correction model .csv file
class CorrectionModelTableEntry(MyDataClass) :
    SlideID : str
    Project : str
    Cohort  : str
    BatchID : str
    FlatfieldVersion : str
    WarpingFile : str
    