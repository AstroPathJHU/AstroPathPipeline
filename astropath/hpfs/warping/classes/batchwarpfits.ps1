<#
--------------------------------------------------------
batchwarpfits
Benjamin Green, Andrew Jorquera
Last Edit: 02.16.2022
--------------------------------------------------------
#>
class batchwarpfits : moduletools {
    #
    [string]$project
    [switch]$all = $false
    batchwarpfits([array]$task,[launchmodule]$sample) : base ([array]$task,[launchmodule]$sample){
        $this.processloc = '\\' + $this.sample.project_data.fwpath + '\warpfits'
        $this.processvars[0] = $this.sample.basepath
        $this.processvars[1] = $this.processloc
        $this.sample.createdirs($this.processloc)
        $this.flevel = [FileDownloads]::IM3 
    }
    #
    batchwarpfits([array]$task,[launchmodule]$sample, $all) : base ([array]$task,[launchmodule]$sample){
        $this.processloc = '\\' + $this.sample.project_data.fwpath + '\warpfits'
        $this.processvars[0] = $this.sample.basepath
        $this.processvars[1] = $this.processloc
        $this.sample.createdirs($this.processloc)
        $this.flevel = [FileDownloads]::IM3 
    }
    <# -----------------------------------------
     Runbatchwarpfits
     ------------------------------------------
     Usage: $this.Runbatchwarpfits()
    ----------------------------------------- #>
    [void]Runbatchwarpfits(){
        $this.getslideidregex()
        $this.getwarpdats()
        $this.Getbatchwarpfits()
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
        $image_keys_file = $this.sample.mpath + '\warping\octets\image_keys_needed.txt'
        $image_keys = $this.sample.GetContent($image_keys_file)
        $this.batchslides | foreach-object{
            $this.sample.info(('shredding files for: '+ $_))
            $images = $this.getslidekeypaths($_, $image_keys)
            if ($images){
                $this.shreddat($_, $images)
            }
        }
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
              $this.sample.MSIfolder() + '\' + $_ + '.im3'
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
        $taskname = 'batchwarpfits'
        $dpath = $this.processvars[0]
        $rpath = '\\' + $this.processvars[1]
        $this.getmodulename()
        #
        $pythontask = $this.getpythontask($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.silentcleanup()
        #
    }
    #
    [string]getpythontask($dpath, $rpath){
        #
        $this.sample.info('start fits')
        #
        $pythontask = (
            $this.pythonmodulename,
            $dpath,
            '--shardedim3root',  $rpath, 
            '--sampleregex',  ('"'+($this.batchslides -join '|')+'"'), 
            '--flatfield-file',  $this.sample.pybatchflatfieldfullpath(), 
            '--noGPU --no-log',
            '--ignore-dependencies',
            $this.buildpyopts('cohort') 
        ) -join ' '
        #
        $this.sample.info('fits finished')
       #
       return $pythontask
       #
    }
    #
    [void]getslideidregex(){
        #
        $this.sample.info('selecting samples for sample regex')
        #
        $nbatchslides = @()
        $sid = $this.sample.slideid
        #
        if ($this.all){
            $aslides = $this.sample.importslideids($this.sample.mpath)
            $aslides = $aslides | where-object {$_.Project -match $this.sample.project}
            $slides = $aslides.SlideID
        } else {
            $slides = $this.sample.batchslides.slideid
        }
        #
        foreach ($slide in $slides){
            $this.sample.slideid = $slide
            if ($this.sample.testwarpoctetsfiles()){
                $nbatchslides += $slide
            }
        }
        #
        $this.sample.slideid = $sid
        $this.sample.info(([string]$nbatchslides.length +
             ' sample(s) selected for sampleregex'))
        $this.batchslides = $nbatchslides
        #
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = 'warpingcohort'
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
}