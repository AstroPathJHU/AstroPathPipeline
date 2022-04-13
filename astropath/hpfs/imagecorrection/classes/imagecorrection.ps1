<#
--------------------------------------------------------
imagecorrection
Created By: Benjamin Green -JHU
Last Edit: 07/23/2021
--------------------------------------------------------
Description
Task to be launched remotely to ANY computer from ANYWHERE
--------------------------------------------------------
Input:
$task[array]: the 3 part array of project, slideid, process loc
    E.g. @('7','M18_1','\\bki08\e$')
$sample[launchmodule]: A launchmodule object 
--------------------------------------------------------
Usage: $a = [imagecorrection]::new($task, $sample)
       $a.runimagecorrection()
--------------------------------------------------------
#>
Class imagecorrection : moduletools {
    #
    [string]$pytype = 'cohort'
    #
    imagecorrection([hashtable]$task,[launchmodule]$sample) : base ([hashtable]$task, [launchmodule]$sample){
        $this.flevel = [FileDownloads]::BATCHID + 
            [FileDownloads]::IM3 + 
            [FileDownloads]::FLATFIELD + 
            [FileDownloads]::XML
        $this.funclocation = '"'+$PSScriptRoot + '\..\funcs"'  
    }
    <# -----------------------------------------
     RunImageCorrection
     Run image correction
     ------------------------------------------
     Usage: $this.runimagecorrection()
    ----------------------------------------- #>
    [void]RunImageCorrection(){
        $this.TestPaths()
        $this.fixM2()
        $this.DownloadFiles()
        $this.ShredDat()
        $this.ApplyFlatw()
        $this.InjectDat()
        $this.cleanup()
        $this.datavalidation()
    }
    <# -----------------------------------------
     TestPaths
     Test that the batch flatfield and im3 
     folder exists in the correct locations
     ------------------------------------------
     Usage: $this.TestPaths()
    ----------------------------------------- #>
    [void]TestPaths(){
        $this.sample.info('run applyflatw for:')
        $s = $this.sample.basepath, $this.sample.flatwfolder(), 
            $this.sample.slideid -join ' '
        $this.sample.info($s)
        $this.sample.testim3mainfolder()
        $this.sample.testbatchflatfield()
    }
   <# -----------------------------------------
     ApplyCor
        apply the correction
     ------------------------------------------
     Usage: $this.ApplyCor()
    ----------------------------------------- #>
    [void]ApplyFlatw(){
        #
        if ($this.sample.vers -match '0.0.1'){
            $this.applyflatwmatlab()
            $this.ExtractLayer(1)
        } else {
            $this.applyflatwpy()
            $this.renamefw2dat()
        }
        #
    }
    [void]applyflatwmatlab(){
        $this.sample.info("started applying correction -- matlab")
        $taskname = 'applycorr'
        $matlabtask = ";runFlatw('"+$this.processvars[0] + 
            "', '" + $this.processvars[1] + "', '" + 
            $this.sample.slideid + "');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("finished applying correction -- matlab")
    }
    [void]applyflatwpy(){
        #
        $this.sample.info("started applying correction -- python")
        $this.getmodulename()
        $taskname = $this.pythonmodulename
        #
        $dpath = $this.sample.basepath + ' '
        $rpath = $this.processvars[1]
        $pythontask = $this.('getpythontask' + $this.pytype)($dpath, $rpath)
        #
        $this.runpythontask($taskname, $pythontask)
        $this.sample.info("finished applying correction -- python")
        #
    }
    #
    [string]getpythontaskcohort($dpath, $rpath){
        #
        $globalargs = $this.buildpyopts('cohort')
        $pythontask = ($this.pythonmodulename,
            $dpath, 
            '--sampleregex', $this.sample.slideid,
            '--shardedim3root', $rpath, 
            '--flatfield-file', $this.sample.pybatchflatfieldfullpath(),
            '--warping-file', ('warping_BatchID_', $this.sample.batchid.padleft(2, '0'), '.csv' -join ''), 
            "--njobs '8' --no-log --layers -1 1", $globalargs,
            '--workingdir', ($rpath + '\' + $this.sample.slideid) -join ' ')
        #
        return $pythontask
    }
    #
    [void]getmodulename(){
        $this.pythonmodulename = ('applyflatw', $this.pytype -join '')
    }
   <# -----------------------------------------
     ExtractLayer
        Extract the particular layer of interest
     ------------------------------------------
     Usage: $this.ExtractLayer($layer)
    ----------------------------------------- #>
    [void]ExtractLayer([int]$layer){
        $this.sample.info("Extract Layer started")
        $taskname = 'extractlayer'
        $matlabtask = ";extractLayer('" + $this.processvars[1] + 
            "', '" + $this.sample.slideid + "', '" + $layer + "');exit(0);"
        $this.runmatlabtask($taskname, $matlabtask)
        $this.sample.info("Extract Layer finished")
    }
    <#
        in order to run the inject written by richard
        you have to use the Data.dat files. With the 'layers'
        option for py applyflatw the files are appended with 
        .fw, instead of the original .Data.dat. Remove old 
        .Data.dat and rename .fw to Data.dat to run inject 
    #>
    [void]renamefw2dat(){
        #
        $this.sample.info("rename fw 2 dat started")
        #
        $files = $this.sample.listfiles(($this.processvars[1], $this.sample.slideid -join '\'), 'fw')
        if (!$files){
            #
            $this.sample.error('no fw files applyflatw failed without an error, will rerun and print entire applyflatw log')
            $this.applyflatwpy()
            #
            $this.getmodulename()
            $taskname = $this.pythonmodulename
            $externallog = $this.ProcessLog($taskname)
            $this.logoutput = $this.sample.GetContent($externallog)
            $this.sample.error($this.logoutput)
            throw 'end rerun for no fw files after appyflatw'
            #
        }

        #
        $this.sample.removefile(
            ($this.processvars[1], $this.sample.slideid -join '\'), '.Data.dat')
        if ($this.sample.listfiles(
            ($this.processvars[1], $this.sample.slideid -join '\'), '.Data.dat')
        ){
            Throw 'data.dat file(s) not properly deleted'
        }
        $this.sample.renamefile(
            ($this.processvars[1], $this.sample.slideid -join '\'), 'fw', 'Data.dat')
        $this.sample.info("rename fw 2 dat finished")
        #
    }
    <# -----------------------------------------
     cleanup
     cleanup the data directory and return the 
     data to the dpath locations
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
    #
    [void]loggedcopy($sor, $des, $type, $filespec){
        #
        $this.sample.info("return $type")
        $this.sample.info('source: '+ $sor)
        $sorcount = (Get-ChildItem ($sor + '\*') -Include "*$filespec").Count
        $this.sample.info('source file(s): ' + $sorcount)
        $this.sample.info('destination: '+ $des)
        $this.sample.copy($sor, $des, $filespec, 10)
        $descount = (Get-ChildItem ($des + '\*') -Include "*$filespec").Count
        $this.sample.info('destintation file(s): ' + $descount)
        #
        if(!($sorcount -eq $descount)){
            Throw "$type did not upload correctly." 
        }
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
        if ($this.processvars[4]){
            #
            # im3s
            #
            $sor = $this.processvars[0] + '\' + 
                $this.sample.slideid + '\im3\flatw'
            $des = $this.sample.flatwim3folder()
            $this.loggedcopy($sor, $des, 'corrected im3s', 'im3')
            #
            $this.sample.copy($sor, $des, '.log')
            #
            # fw files
            #
            $sor = $this.processvars[1] + '\' + $this.sample.slideid
            $des = $this.sample.flatwfolder()
            #
            @('fw', 'fw01') | foreach-object {
                #
                $this.loggedcopy($sor, $des, "corrected $_", $_)
                #
            }
            #
            $this.sample.copy($sor, $des, '.log')
            $this.sample.removedir($this.processloc)
            #
        }
        #
    }
    <# -----------------------------------------
     datavalidation
     Validation of output data
     ------------------------------------------
     Usage: $this.datavalidation()
    ----------------------------------------- #>
    [void]datavalidation(){
        if (!$this.sample.testimagecorrectionfiles()){
            throw 'Output files are not correct'
        }
    }
}