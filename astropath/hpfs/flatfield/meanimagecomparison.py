#imports
import os, pathlib, logging, re
from argparse import ArgumentParser
import numpy as np
from ...utilities.config import CONST as UNIV_CONST
from ...utilities.misc import split_csv_to_list, split_csv_to_list_of_floats
from ...utilities.tableio import readtable,writetable
from ...utilities.img_file_io import get_image_hwl_from_xml_file, get_raw_as_hwl
from ...shared.samplemetadata import SampleDef
from .config import CONST
from .utilities import FlatfieldModelTableEntry, ComparisonTableEntry, normalize_mean_image
from .plotting import meanimage_comparison_plot
from .batchflatfieldmulticohort import BatchFlatfieldMultiCohort

class MeanImageComparison :
    """
    class to manage making/adding to the datatable/plots of mean image patterns compared to one another
    """

    #################### PUBLIC FUNCTIONS ####################

    def __init__(self,root_dirs,sampleregex,workingdir,sort_by,use_flatw,min_images_stacked) :
        """
        root_dirs:   list of paths to root directories for all samples that should be included/appended
        sampleregex: regular expression matching all slides that should be used
        workingdir:  path to directory that should hold output (may already hold some output that should be appended to)
        sort_by:     string indicating how slides should be ordered on the plot
        use_flatw:   True if meanimages from .fw files should be used instead of default meanimages (from raw files)
        min_images_stacked: Skip any slides that stacked less than this number of images anywhere
        """
        #create the working directory
        self.workingdir = workingdir
        if not self.workingdir.is_dir() :
            self.workingdir.mkdir(parents=True)
        #create a general logger
        self.logger = self.__get_logger()
        #set the name of the meanimage subdirectory to search
        self.meanimage_subdir_name = self.FLATW_MEANIMAGE_SUBDIR_NAME if use_flatw else self.MEANIMAGE_SUBDIR_NAME
        #create a dictionary keyed by root directory paths, with values that are lists of slide IDs at those paths
        self.slide_ids_by_rootdir = self.__get_slides_by_rootdir(root_dirs,sampleregex,min_images_stacked)
        #get a list of all the slide IDs and their mean image filepaths, sorted in the order they'll be plotted
        #(also returns the list of slide IDs after which lines would go if sorted by project/cohort/batch)
        self.ordered_slide_tuples,self.lines_after = self.__get_sorted_slide_tuples(sort_by)

    def calculate_comparisons(self) :
        """
        Calculate delta/sigma values for every slide compared to every other slide, in each layer of each meanimage
        Write results to a datatable
        """
        #start up an array to hold all of the necessary values and a list of table entries
        n_slides = len(self.ordered_slide_tuples)
        layers = list(range(1,self.dims[-1]+1))
        self.dos_std_dev_values = np.zeros((n_slides,n_slides,self.dims[-1]))
        #read any values that have already been calculated from the datatable they're stored in
        if self.datatable_path.is_file() :
            existing_entries = readtable(self.datatable_path,ComparisonTableEntry)
        else :
            existing_entries = []
        #populate the array with all of the values needed
        n_pairs_to_do = int(((len(self.ordered_slide_tuples)**2)-len(self.ordered_slide_tuples))/2)
        pairs_done = set()
        for is1,(sid1,mid1) in enumerate(self.ordered_slide_tuples) :
            mi1fp = mid1/f'{sid1}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
            semi1fp = mid1/f'{sid1}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'
            for is2,(sid2,mid2) in enumerate(self.ordered_slide_tuples) :
                mi2fp = mid2/f'{sid2}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
                semi2fp = mid2/f'{sid2}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'
                #put zeroes along the diagonal
                if sid2==sid1 :
                    self.dos_std_dev_values[is1,is2,:] = 0.
                    continue 
                #if this pair of slide IDs has been done in the opposite order, copy the values across the diagonal
                elif (sid2,sid1) in pairs_done :
                    self.dos_std_dev_values[is1,is2,:] = self.dos_std_dev_values[is2,is1,:]
                    continue
                else :
                    #check if the requested values are available from the file that was read
                    n_existing_entries = len([e for e in existing_entries 
                                              if ((e.slide_ID_1==sid1 and e.slide_ID_2==sid2) or 
                                                  (e.slide_ID_1==sid2 and e.slide_ID_2==sid1))])
                    if n_existing_entries==self.dims[-1] :
                        added=set()
                        for entry in existing_entries :
                            if ( (entry.slide_ID_1==sid1 and entry.slide_ID_2==sid2) or 
                                 (entry.slide_ID_1==sid1 and entry.slide_ID_2==sid2) ) :
                                self.dos_std_dev_values[is1,is2,entry.layer_n-1] = entry.delta_over_sigma_std_dev
                                added.add(entry.layer_n)
                        if len(added)==self.dims[-1] :
                            msg = f'Std. devs. of delta/sigma for {sid1} vs {sid2} read from {self.datatable_path}'
                            self.logger.debug(msg)
                    else :
                        #otherwise calculate them
                        msg = f'Finding std. devs. of delta/sigma for {sid1} vs. {sid2} '
                        msg+= f'({len(pairs_done)+1}/{n_pairs_to_do})...'
                        self.logger.info(msg)
                        mi1   = get_raw_as_hwl(mi1fp,*(self.dims),np.float64)
                        semi1 = get_raw_as_hwl(semi1fp,*(self.dims),np.float64)
                        mi2   = get_raw_as_hwl(mi2fp,*(self.dims),np.float64)
                        semi2 = get_raw_as_hwl(semi2fp,*(self.dims),np.float64)
                        dossd_list = self.__get_delta_over_sigma_std_devs_by_layer(mi1,semi1,mi2,semi2)
                        self.dos_std_dev_values[is1,is2,:] = dossd_list
                        #append the values to the table
                        new_entries = []
                        for ln in layers :
                            new_entries.append(ComparisonTableEntry(mid1.parent.parent.parent,sid1,
                                                                    mid2.parent.parent.parent,sid2,
                                                                    ln,self.dos_std_dev_values[is1,is2,ln-1]))
                        writetable(self.datatable_path,new_entries,append=True)
                    pairs_done.add((sid1,sid2))

    def create_plots(self,to_plot,lines_after,bounds) :
        """
        Create comparison plot images

        to_plot:     whether to save plots for "all" layers individually, for the "brightest" layers, or for the 
                     "average" over all layers
        lines_after: list of slideIDs after which dividing lines should be drawn on the plot
        bounds:      the lower and upper bounds on the imshow scale for the plot(s)
        """
        #make the list of ordered slide IDs to send to the plotting function
        slide_ids = [st[0] for st in self.ordered_slide_tuples]
        #reset the lines_after variable if it was externally supplied
        if lines_after!=[''] :
            self.lines_after = lines_after
        #for each image layer requested, plot a grid of the delta/sigma comparisons
        if to_plot=='all' :
            layers = list(range(1,self.dims[-1]+1))
        elif to_plot=='brightest' :
            layers = UNIV_CONST.BRIGHTEST_LAYERS_35 if self.dims[-1]==35 else UNIV_CONST.BRIGHTEST_LAYERS_43
        else :
            layers = []
        for ln in layers : 
            self.logger.debug(f'Saving plot for layer {ln}...')
            meanimage_comparison_plot(slide_ids,
                                      self.dos_std_dev_values[:,:,ln-1],
                                      f'mean image delta/sigma std. devs. in layer {ln}',
                                      f'meanimage_comparison_layer_{ln}.png',
                                      self.workingdir,
                                      self.lines_after,
                                      bounds)
        if to_plot=='average' :
            #save a plot of the average over all considered layers and also over the filter groups individually
            self.logger.debug('Saving plots of values averaged over all layers and within filter groups...')
            layer_groups = None
            if self.dims[-1]==35 :
                layer_groups = UNIV_CONST.LAYER_GROUPS_35
            elif self.dims[-1]==43 :
                layer_groups = UNIV_CONST.LAYER_GROUPS_43
            else :
                raise RuntimeError(f'ERROR: no broadband filter groups defined for images with {self.dims[-1]} layers')
            layer_groups = [(1,self.dims[-1])]+layer_groups
            for ilg,lg in enumerate(layer_groups) :
                average_values = np.zeros_like(self.dos_std_dev_values[:,:,0])
                for i in range(len(slide_ids)) :
                    for j in range(len(slide_ids)) :
                        num = 0; den = 0
                        for li in range(lg[0]-1,lg[1]) :
                            if self.dos_std_dev_values[i,j,li]!=0. :
                                num+=self.dos_std_dev_values[i,j,li]
                                den+=1
                        if den==0 :
                            average_values[i,j]=0.
                        else :
                            average_values[i,j]=num/den
                title = 'mean image delta/sigma std. devs. (averaged over '
                filename = 'meanimage_comparison_average_over_'
                if ilg==0 :
                    title+='all'
                    filename+='all'
                else :
                    title+=f'filter group {ilg}'
                    filename+=f'filter_group_{ilg}'
                title+=' layers)'
                filename+='_layers.png'
                meanimage_comparison_plot(slide_ids,
                                          average_values,
                                          title,
                                          filename,
                                          self.workingdir,
                                          self.lines_after,
                                          bounds)

    def save_model(self,root_dirs,version_tag,skip_creation) :
        """
        Add lines to the flatfield models .csv file defining a model with this group of slides under the given 
        version_tag, and then run BatchFlatfieldMultiCohort to actually create the model in the default location
        """
        all_lines = []
        #read anything already in the table
        if self.MODEL_TABLE_PATH.is_file() :
            all_lines = readtable(self.MODEL_TABLE_PATH,FlatfieldModelTableEntry)
        #add a line for every slide used
        for sid,mid in self.ordered_slide_tuples :
            sd = SampleDef(SlideID=sid,root=mid.parent.parent.parent)
            all_lines.append(FlatfieldModelTableEntry(version_tag,sd.Project,sd.Cohort,sd.BatchID,sd.SlideID))
        #write out the table
        writetable(self.MODEL_TABLE_PATH,all_lines)
        #run batchflatfieldmulticohort if requested
        if not skip_creation :
            batchflatfieldmulticohortargs = []
            for root_dir in root_dirs :
                batchflatfieldmulticohortargs.append(os.fspath(root_dir))
            batchflatfieldmulticohortargs+=['--version',version_tag]
            batchflatfieldmulticohortargs+=['--flatfield-model-file',os.fspath(self.MODEL_TABLE_PATH)]
            BatchFlatfieldMultiCohort.runfromargumentparser(args=batchflatfieldmulticohortargs)

    #################### CLASS VARIABLES & PROPERTIES ####################

    MEANIMAGE_SUBDIR_NAME = UNIV_CONST.MEANIMAGE_DIRNAME
    FLATW_MEANIMAGE_SUBDIR_NAME = f'{UNIV_CONST.MEANIMAGE_DIRNAME}_from_fw_files'
    DATATABLE_NAME = 'meanimagecomparison_table.csv'
    MODEL_TABLE_PATH = CONST.DEFAULT_FLATFIELD_MODEL_FILEPATH

    #################### CLASS METHODS ####################

    @classmethod
    def get_args(cls,given_args=None) :
        parser = ArgumentParser()
        # root: some number of root directories whose sampledef.csv file(s) list a group of slides that could be used
        parser.add_argument('root', type=pathlib.Path, nargs='+',
                            help='''Paths to directories with [slideID]/im3/meanimage 
                                    subdirectories and sampledef.csv files in them''')
        # sampleregex: a regular expression matching all of the slide IDs that should be used
        parser.add_argument('--sampleregex', type=re.compile, help='only run on SlideIDs that match this regex')
        # workingdir: the directory the output should go in (with a default location)
        def_workingdir = UNIV_CONST.ASTROPATH_PROCESSING_DIR/'meanimagecomparison'
        parser.add_argument('--workingdir', type=pathlib.Path, default=def_workingdir,
                            help='Path to the working directory where results will be saved')
        #other optional arguments
        parser.add_argument('--sort-by', choices=['order','project_cohort_batch'], default='project_cohort_batch',
                            help='how to sort the order of the slide IDs')
        parser.add_argument('--flatw', action='store_true',
                            help='''Add this flag to use meanimages from "meanimage_from_fw_files" instead of 
                                    "meanimage" subdirectories''')
        parser.add_argument('--min-images-stacked', type=int, default=1,
                            help='Skip any slides that stacked less than this number of HPF images everywhere')
        parser.add_argument('--plot', choices=['all','brightest','average','none'], default='average',
                            help='''Add one of these choices to save the plots of some individual layers, the average
                                    over all of them/filter groups (default), or no plots at all''')
        parser.add_argument('--lines-after', type=split_csv_to_list, default='',
                            help="""Comma-separated list of slides to put separating lines after on the plot 
                                    (default is project/cohort/batch transitions, adding this argument will overwrite 
                                    those automatic options)""")
        parser.add_argument('--bounds', type=split_csv_to_list_of_floats, default='0.85,1.15',
                            help='Hard limits to the imshow scale for the plot (given as lower_bound,upper_bound')
        parser.add_argument('--store-as', 
                            help='''Flatfield model version tag to store the group of slides as. Including this argument 
                                    writes out all of the slides in the plot as part of a flatfield model in the
                                    flatfield models .csv file''')
        parser.add_argument('--skip-creation',action='store_true',
                            help='''Add this flag to skip creating the flatfield model in the default location, 
                                    if "store-as" is given. 
                                    (BatchFlatfieldMultiCohort can be run standalone later on.)''')
        args = parser.parse_args(args=given_args)
        #make sure some arguments make sense
        for root_dir in args.root :
            if not root_dir.is_dir() :
                raise ValueError(f'ERROR: root directory {root_dir} not found!')
        if len(args.bounds)!=2 or args.bounds[0]>=args.bounds[1] :
            raise ValueError(f'ERROR: invalid bounds argument {args.bounds}')
        #return the parsed arguments
        return args

    @classmethod
    def run(cls,args=None) :
        """
        Actually run the code start to finish
        """
        #get and parse the command-line arguments
        args = cls.get_args(args)
        #set up the Comparison
        mic = cls(args.root,args.sampleregex,args.workingdir,args.sort_by,args.flatw,args.min_images_stacked)
        #create (or append to) the datatable of comparison values
        mic.calculate_comparisons()
        #create any requested plots
        if args.plot!='none' :
            mic.create_plots(args.plot,args.lines_after,args.bounds)
        #if requested, add some lines to the flatfield models .csv file
        if args.store_as is not None :
            mic.save_model(args.root,args.store_as,args.skip_creation)

    #################### PRIVATE HELPER FUNCTIONS ####################

    def __get_logger(self) :
        """
        Return the general logger to use
        """
        logger = logging.getLogger("meanimagecomparison")
        logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        logging_formatter = logging.Formatter("[%(asctime)s] %(message)s","%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(logging_formatter)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(self.workingdir/'meanimagecomparison.log')
        file_handler.setFormatter(logging_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        return logger

    def __get_slides_by_rootdir(self,root_dirs,sampleregex,min_images_stacked) :
        """
        Return a dictionary whose keys are root directory paths and whose values are lists of slide IDs in them to use
        Also sets the dimensions of the images that will be used
        """
        self.logger.debug('Finding slides to use...')
        self.dims = None
        slide_ids_by_rootdir = {}
        #start by adding any slide IDs from the file in the working directory, if it exists
        self.datatable_path = self.workingdir/self.DATATABLE_NAME
        if self.datatable_path.is_file() :
            all_entries = readtable(self.datatable_path,ComparisonTableEntry)
            for entry in all_entries :
                rd1  = pathlib.Path(entry.root_dir_1)
                sid1 = entry.slide_ID_1
                rd2  = pathlib.Path(entry.root_dir_2)
                sid2 = entry.slide_ID_2
                if (sampleregex is None) or (sampleregex.match(sid1)) :
                    if rd1 not in slide_ids_by_rootdir.keys() :
                        slide_ids_by_rootdir[rd1] = []
                    if sid1 not in slide_ids_by_rootdir[rd1] :
                        this_slide_dims = get_image_hwl_from_xml_file(rd1,sid1)
                        if self.dims is None :
                            self.dims = this_slide_dims
                        elif this_slide_dims!=self.dims :
                            errmsg = f'ERROR: slide {sid1} has dimensions {this_slide_dims},'
                            errmsg+= f' mismatched to {self.dims}'
                            raise RuntimeError(errmsg)
                        self.logger.debug(f'{sid1} found in existing datatable at {self.datatable_path}')
                        slide_ids_by_rootdir[rd1].append(sid1)
                if (sampleregex is None) or (sampleregex.match(sid2)) :
                    if rd2 not in slide_ids_by_rootdir.keys() :
                        slide_ids_by_rootdir[rd2] = []
                    if sid2 not in slide_ids_by_rootdir[rd2] :
                        this_slide_dims = get_image_hwl_from_xml_file(rd2,sid2)
                        if self.dims is None :
                            self.dims = this_slide_dims
                        elif this_slide_dims!=self.dims :
                            errmsg = f'ERROR: slide {sid2} has dimensions {this_slide_dims},'
                            errmsg+= f' mismatched to {self.dims}'
                            raise RuntimeError(errmsg)
                        self.logger.debug(f'{sid2} found in existing datatable at {self.datatable_path}')
                        slide_ids_by_rootdir[rd2].append(sid2)
        else :
            if len(root_dirs)<1 :
                errmsg = 'ERROR: no root directories given and no existing data table defining them found at '
                errmsg+= f'{self.datatable_path}'
                raise ValueError(errmsg)
        for root_dir in root_dirs :
            samps = readtable(pathlib.Path(root_dir)/'sampledef.csv',SampleDef)
            sids = slide_ids_by_rootdir[root_dir] if root_dir in slide_ids_by_rootdir.keys() else []
            sids_to_check = [s.SlideID for s in samps]+sids
            for sid in sids_to_check :
                if (sampleregex is None) or (sampleregex.match(sid)) :
                    this_slide_dims = get_image_hwl_from_xml_file(root_dir,sid)
                    if self.dims is None :
                        self.dims = this_slide_dims
                    elif this_slide_dims!=self.dims :
                        errmsg = f'ERROR: slide {sid} has dimensions {this_slide_dims},'
                        errmsg+= f' mismatched to {self.dims}'
                        raise RuntimeError(errmsg)
                    midfp = root_dir/sid/UNIV_CONST.IM3_DIR_NAME/self.meanimage_subdir_name
                    mifp = midfp/f'{sid}-{CONST.MEAN_IMAGE_BIN_FILE_NAME_STEM}'
                    semifp = midfp/f'{sid}-{CONST.STD_ERR_OF_MEAN_IMAGE_BIN_FILE_NAME_STEM}'
                    msfp = midfp/f'{sid}-{CONST.MASK_STACK_BIN_FILE_NAME_STEM}'
                    if not mifp.is_file() :
                        self.logger.warning(f'WARNING: expected mean image {mifp} not found! ({sid} will be skipped!)')
                        continue
                    if not semifp.is_file() :
                        warnmsg = f'WARNING: expected std. err. of mean image {semifp} not found!'
                        warnmsg+= f' ({sid} will be skipped!)'
                        self.logger.warning(warnmsg)
                        continue
                    if not msfp.is_file() :
                        self.logger.warning(f'WARNING: expected mask stack {mifp} not found! ({sid} will be skipped!)')
                        continue
                    mi   = get_raw_as_hwl(mifp,*(self.dims),np.float64)
                    semi = get_raw_as_hwl(semifp,*(self.dims),np.float64)
                    ms   = get_raw_as_hwl(msfp,*(self.dims),np.uint64)
                    if np.min(mi)==np.max(mi) or np.max(semi)==0. or np.min(ms)<min_images_stacked :
                        warnmsg = f'WARNING: slide {sid} will be skipped because not enough images were stacked!'
                        self.logger.warning(warnmsg)
                        if sid in sids :
                            sids.pop(sid)
                    else :
                        self.logger.debug(f'{sid} is valid and will be used')
                        sids.append(sid)
            slide_ids_by_rootdir[root_dir] = sids
        n_total_slides = sum([len(sids) for sids in slide_ids_by_rootdir.values()])
        if n_total_slides<2 :
            raise RuntimeError(f'ERROR: only {n_total_slides} valid slides were selected!')
        msg = f"The following {n_total_slides} slides' meanimages will be compared: "
        for sids in slide_ids_by_rootdir.values() :
            for sid in sids :
                msg+=f'{sid}, '
        self.logger.info(msg[:-2])
        return slide_ids_by_rootdir

    def __get_sorted_slide_tuples(self,sort_by) :
        """
        Return a list of tuples of (slide_ID,path_to_meanimage_directory) sorted by the order in which they'll be 
        plotted. Also returns a list of slide IDs after which lines would be placed on the plot, based on 
        project/cohort/batch divisions
        """
        sampledefs = []
        for rd,sids in self.slide_ids_by_rootdir.items() :
            for sid in sids :
                sampledefs.append(SampleDef(SlideID=sid,root=rd))
        ordered_tuples = []
        lines_after = []
        if sort_by=='project_cohort_batch' :
            projects = list(set([sd.Project for sd in sampledefs]))
            cohorts = list(set([sd.Cohort for sd in sampledefs]))
            batches = list(set([sd.BatchID for sd in sampledefs]))
            projects.sort(); cohorts.sort(); batches.sort()
            for p in projects :
                if len(ordered_tuples)>0 and ordered_tuples[-1][0] not in lines_after :
                    lines_after.append(ordered_tuples[-1][0])
                for c in cohorts :
                    if len(ordered_tuples)>0 and ordered_tuples[-1][0] not in lines_after :
                        lines_after.append(ordered_tuples[-1][0])
                    for b in batches :
                        if len(ordered_tuples)>0 and ordered_tuples[-1][0] not in lines_after :
                            lines_after.append(ordered_tuples[-1][0])
                        for sd in sampledefs :
                            if sd.Project==p and sd.Cohort==c and sd.BatchID==b :
                                root = ([rd for rd,sids in self.slide_ids_by_rootdir.items() if sd.SlideID in sids])[0]
                                mid = root/sd.SlideID/UNIV_CONST.IM3_DIR_NAME/self.meanimage_subdir_name
                                ordered_tuples.append((sd.SlideID,mid))
            return ordered_tuples, lines_after[:-1]
        elif sort_by=='order' :
            for sd in sampledefs :
                mid = sd.root/sd.SlideID/UNIV_CONST.IM3_DIR_NAME/self.meanimage_subdir_name
                ordered_tuples.append((sd.SlideID,mid))
            return ordered_tuples, lines_after
        else :
            raise RuntimeError(f'ERROR: sort option {sort_by} is not recognized!')

    def __get_delta_over_sigma_std_devs_by_layer(self,mi1,semi1,mi2,semi2) :
        """
        get the standard deviation of the delta/sigma distribution for every layer 
        of two meanimages compared to one another
        """
        #normalize the images by their means in each layer
        mi1,semi1 = normalize_mean_image(mi1,semi1)
        mi2,semi2 = normalize_mean_image(mi2,semi2)
        #make the delta/sigma image
        delta_over_sigma = (mi1-mi2)/(np.sqrt(semi1**2+semi2**2))
        std_devs = []
        for li in range(self.dims[-1]) :
            std_devs.append(np.std(delta_over_sigma[:,:,li][delta_over_sigma[:,:,li]!=0]))        
        return std_devs

#################### MAIN SCRIPT ####################

def main(args=None) :
    MeanImageComparison.run(args)

if __name__=='__main__' :
    main()
