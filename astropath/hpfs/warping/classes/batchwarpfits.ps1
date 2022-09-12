<#
--------------------------------------------------------
batchwarpfits
Benjamin Green, Andrew Jorquera
Last Edit: 02.16.2022
--------------------------------------------------------
#>
class batchwarpfits : moduletools {
    #
    batchwarpfits([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task,[launchmodule]$sample){
       # $this.processloc = '\\' + $this.sample.project_data.fwpath +
       #     '\warpfits\Batch_' + $this.sample.BatchID
        $this.processvars[0] = $this.sample.basepath
        $this.processvars[1] = $this.processloc
        $this.sample.createnewdirs($this.processloc)
    }
    #
    <# -----------------------------------------
     Runbatchwarpfits
     ------------------------------------------
     Usage: $this.Runbatchwarpfits()
    ----------------------------------------- #>
    [void]Runbatchwarpfits(){
        #
        $this.getslideidregex('batchwarpfits')
        $this.getwarpdats()
        $this.Getbatchwarpfits()
        $this.copyfinalfile()
        $this.cleanup()
        #
    }
    <# -----------------------------------------
     getwarpdats
     get the data.dat files to be used in the
     warp fits
     ------------------------------------------
     Usage: $this.getwarpdats()
    ----------------------------------------- #>
    [void]getwarpdats(){
        #
        $this.sample.info('shredding neccessary files to flatw\warpfits')

        $image_keys = $this.sample.GetContent($this.getkeysloc())
        #
        if (!$image_keys){
            Throw ('no keys found in: ' + $this.getkeysloc())
        }
        #
        $this.batchslides | foreach-object{
            $this.sample.info(('shredding files for: '+ $_))
            $images = $this.getslidekeypaths($_, $image_keys)
            if ($images){
                $this.shreddat($_, $images)
            }
        }
    }
    #
    [string]getkeysloc(){
        if ($this.all){
            $image_keys_file = $this.sample.warpprojectoctetsfolder() + '\image_keys_needed.txt'
        } else {
            $image_keys_file = $this.sample.warpbatchoctetsfolder() + '\image_keys_needed.txt'
        }
        return $image_keys_file
    }
    <# -----------------------------------------
     getslidekeypaths
     get the paths from the keys particular slide
     ------------------------------------------
     Usage: $this.getslidekeypaths()
    ----------------------------------------- #>
    [array]getslidekeypaths($slideid, $image_keys){
        $sid = $this.sample.slideid
        $this.sample.slideid = $slideid
        $current_keys = $image_keys -match ($slideid +'_')
        if (!$current_keys){
            $this.sample.slideid = $sid 
            return @()
        }
        $current_keys = $current_keys |
         foreach-object {
              $this.sample.im3folder() + '\' + $_ + '.im3'
        }
        $this.sample.slideid = $sid 
        return $current_keys
    }
    <# -----------------------------------------
     Getbatchwarpfits
     ------------------------------------------
     Usage: $this.Getbatchwarpfits()
    ----------------------------------------- #>
    [void]Getbatchwarpfits(){
        #
        $this.sample.info('start fits')
        $this.getmodulename()
        $taskname = $this.pythonmodulename
        $dpath = $this.processvars[0]
        $rpath = $this.processvars[1]
        #
        $pythontask = $this.getpythontask($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info('fits finished')
        #
    }
    #
    [string]getpythontask($dpath, $rpath){
        #
        $pythontask = (
            $this.pythonmodulename,
            $dpath,
            '--shardedim3root',  $rpath, 
            '--sampleregex',  ('"'+($this.batchslides -join '|')+'"'), 
            '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), 
            $this.gpuopt(),'--no-log',
            '--ignore-dependencies',
            $this.buildpyopts('cohort'),
            '--workingdir', $this.workingdir()
        ) -join ' '
        #
       return $pythontask
       #
    }
    #
    [string]workingdir(){
        if ($this.all){
            return $this.sample.warpprojectfolder()
        } else {
            return $this.sample.warpbatchfolder()
        }
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = 'warpingcohort'
    }
    #
    [void]copyfinalfile(){
        if (!$this.all){
            $file = $this.workingdir(), 'weighted_average_warp.csv' -join '\'
            $this.sample.copy($file, $this.sample.warpfolder())
            $this.sample.renamefile($this.sample.warpfolder(),
                'weighted_average_warp.csv',
                $this.sample.batchwarpingfile())
        }
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory
     ------------------------------------------
     Usage: $this.cleanup()
    ----------------------------------------- #>
    [void]cleanup(){
        #
        $this.sample.info("cleanup started")
        $this.silentcleanup()
        $this.sample.info("cleanup finished")
        #
    }
    <# -----------------------------------------
     silentcleanup
        silentcleanup
     ------------------------------------------
     Usage: $this.silentcleanup()
    ----------------------------------------- #>
    [void]silentcleanup(){
        #
        $this.sample.removedir($this.processloc)
        #
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testbatchwarpfitsfiles()){
            throw 'Output files are not correct'
        }
    }
}