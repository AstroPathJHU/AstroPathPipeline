Import-Module 'H:\andrew\AstroPathPipelinePrivate\astropath'
$task = ('1', 'M21_1', '\\bki08\h$\testing\testing_meanimage', '\\bki08\h$\testing\astropath_processing')
$inp = meanimage $task

$inp.DownloadFiles()

$inp.ShredDat()

$inp.GetMeanImage()

$inp.returndata()

$inp.cleanup()