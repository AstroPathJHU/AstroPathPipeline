#class for shared constant variables
class Const :
    #overlap cost parameterization
    @property
    def OVERLAP_COST_PARAMETERIZATION_N_POINTS(self) :
        return 100 #number of points to test between the bounds for each overlap's cost parameterization
    
CONST=Const()