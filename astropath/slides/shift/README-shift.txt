%%-----------------------------------------------------------------
%% README-shift.txt
%%
%% Solves for the optimal field placement by figuring out
%% the shifts that minimize the rmns alignment error
%%
%% Alex Szalay, Baltimore, 2019-02-03
%%-----------------------------------------------------------------

Main modules:

	C = shiftSample(root1,root2,sample [,opt]);
	
		It will create a C struct, which contains everything
		needed to perform the solution. The code will read
		all the annotations, QPTIFF, pairwise alignments. It
        computes the distinct partitions, the affine transformation
        mapping form microscope to pixel coordinates, and the field
        positions that minimize the rms scatter of the residual
        pairwise alignments. It writes the results as csv files 
        into the sample\dbload directory. The following files are 
        written (* represents the sample):
			*_fields.csv
			*_affine.csv
			*_sigma.csv
		
