def addCommonArgumentsToParser(parser,positional_args=True,et_correction=True,flatfielding=True,warping=True) :
  """
  helper function to mutate an argument parser for some very generic options
  """
  #positional arguments
  if positional_args :
    parser.add_argument('rawfile_top_dir',  help='Path to the directory containing the raw files')
    parser.add_argument('root_dir',         help='Path to the Clinical_Specimen directory with info for the given slide')
    parser.add_argument('workingdir',       help='Path to the working directory (will be created if necessary)')
  #mutually exclusive group for how to handle the exposure time correction
  if et_correction :
    et_correction_group = parser.add_mutually_exclusive_group(required=True)
    et_correction_group.add_argument('--exposure_time_offset_file',
                                     help="""Path to the .csv file specifying layer-dependent exposure time correction offsets for the slides in question
                                    [use this argument to apply corrections for differences in image exposure time]""")
    et_correction_group.add_argument('--skip_exposure_time_correction', action='store_true',
                                     help='Add this flag to entirely skip correcting image flux for exposure time differences')
  #mutually exclusive group for how to handle the flatfielding
  if flatfielding :
    flatfield_group = parser.add_mutually_exclusive_group(required=True)
    flatfield_group.add_argument('--flatfield_file',
                                 help='Path to the flatfield.bin file that should be applied to the files in this slide')
    flatfield_group.add_argument('--skip_flatfielding', action='store_true',
                                 help='Add this flag to entirely skip flatfield corrections')
  #mutually exclusive group for how to handle the warping corrections
  if warping :
    warping_group = parser.add_mutually_exclusive_group(required=True)
    warping_group.add_argument('--warp_def',   
                               help="""Path to the weighted average fit result file of the warp to apply, 
                                    or to the directory with the warp's dx and dy shift fields""")
    warping_group.add_argument('--skip_warping', action='store_true',
                               help='Add this flag to entirely skip warping corrections')
