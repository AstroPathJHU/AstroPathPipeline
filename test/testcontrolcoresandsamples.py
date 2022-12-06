#imports
import pathlib, unittest, os, shutil
from astropath.slides.batch_correction.make_cores_and_samples import main

#some constants
folder = pathlib.Path(__file__).parent
tissue_types_file = folder/'data'/'astropath_processing'/'batch_correction'/'TMA_tissuetypes.xlsx'
outdir = folder/'test_for_jenkins'/'testcontrolcoresandsamples'
project_folder = folder/'data'
ref_dir = folder/'data'/'reference'/'controlcoresandsamples'

class TestControlCoresAndSamples(unittest.TestCase) :
    """
    A class to test the batch_corrections/make_cores_and_samples.py script
    """

    def test_control_cores_and_samples(self) :
        try :
            #run the script
            args = [
                0, #project number
                '--tissue_types_file', os.fspath(tissue_types_file),
                '--outdir', os.fspath(outdir),
                '--project_folder', os.fspath(project_folder),
                '--cohort_number', '0',
            ]
            main(args)
            #make sure the expected output files exist
            filenames = [
                'Control_TMA1372_all_info.xlsx',
                'Control_TMA1372_info.xlsx',
                'Project0_ctrlcores_all.csv',
                'Project0_ctrlcores.csv',
                'Project0_ctrlsamples_ext.xlsx',
                'Project0_ctrlsamples.csv',
            ]
            for filename in filenames :
                path = outdir / 'Ctrl' / filename
                print(f'Testing that {filename} exists')
                self.assertTrue(path.is_file())
            #compare the lines of the csv files to the references
            for ref_path in ref_dir.glob('*') :
                print(f'Testing that {ref_path.name} matches reference file')
                test_path = outdir / 'Ctrl' / ref_path.name
                with open(ref_path,'r') as rfp :
                    ref_lines = rfp.readlines()
                with open(test_path,'r') as tfp :
                    test_lines = tfp.readlines()
                for ref_line, test_line in zip(ref_lines,test_lines) :
                    self.assertEqual(test_line,ref_line)
        except Exception as e :
            raise e
        finally :
            if outdir.is_dir() :
                shutil.rmtree(outdir)
