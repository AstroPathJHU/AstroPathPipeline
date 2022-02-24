 <# -------------------------------------------
 testpssharedtools
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test the shared tools utilities 
 -------------------------------------------#>
#
 Class testpssharedtools {
    #
    [string]$mpath 
    [string]$process_loc
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    #
    testpssharedtools(){
        #
        Write-Host '---------------------test ps [sharedtools]---------------------'
        $this.importmodule()
        $this.testconstructor()
        $tools = sharedtools
        $this.testcondaenvir($tools)
        $this.testcheckgitrepo($tools)
        $this.testcreatedirs($tools)
        $this.testcopy($tools)
        Write-Host '.'
        #
    }
    #
    importmodule(){
        Import-Module $this.uncpath($this.apmodule)
        $this.mpath = $PSScriptRoot + '\data\astropath_processing'
        $this.process_loc = $PSScriptRoot + '\test_for_jenkins\testing'
    }
    #
    [string]uncpath($str){
        $r = $str -replace( '\\', '/')
        if ($r[0] -ne '/'){
            $root = ('//' + $env:computername+'/'+$r) -replace ":", "$"
        } else{
            $root = $r -replace ":", "$"
        }
        return $root
    }
    #
    testconstructor(){
        #
        Write-Host '.'
        Write-Host 'test [sharedtools] constructor started'
        try {
            sharedtools | Out-Null
        } catch {
            Throw ('cannot create a [sharedtools] object. ' + $_.Exception.Message)
        }
        #
        Write-Host 'test [sharedtools] constructor finished'
        #
    }
    #
    [void]testcheckgitrepo($tools){
        #
        Write-Host '.'
        Write-Host 'start get info for git versioning'
        Write-Host '    root: ' $tools.defRoot()
        Write-Host '    Git repo path: ' $tools.pypackagepath() 
        Write-Host '    Git installed: ' $tools.checkgitinstalled()
        Write-Host '    Git repo: ' $tools.checkgitrepo()
        Write-Host '    Git version: ' $tools.getgitversion()
        Write-Host '    Git status: ' $tools.checkgitstatus() 
        Write-Host '    Git full version: ' $tools.getfullversion()
        #
    }
    #
    [void]testcreatedirs($tools){
        #
        Write-Host '.'
        Write-Host 'test create dirs and files started'
        #
        $logpath = $PSScriptRoot + '\data\logfiles'
        #
        Write-Host '    '$logpath
        #
        $tools.createdirs($logpath)
        #
        if (!(test-path $logpath)){
            Throw 'could not create folder in data'
        }
        #
        Write-Host '    dir created'
        #
        $logfile = $logpath + '\logfile.log'
        $content = 'log file contents'
        #
        set-content $logfile $content -EA Stop
        #
        Write-Host '    file created'
        #
        $tools.setfile($logfile, $content)
        #
        Write-Host '    setfile checked'
        #
        $tools.popfile($logfile, $content)
        #
        Write-Host '    popfile checked'
        #
        $tools.removedir($logpath)
        #
        Write-Host '    remove dir checked'
        #
        $tools.popfile($logfile, $content)
        #
        Write-Host '    pop file without dir checked'
        #
        Write-Host 'test create dirs and files finished'
        #
    }
    #
    [void]testcopy($tools){
        #
        Write-Host '.'
        Write-Host 'test copy files started'
        #
        $sor = $PSScriptRoot + '\data\logfiles'
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $content = 'log file contents'
        $tools.popfile($sorfile, $content)
        #
        if (!(Test-Path $sorfile)){
            Throw 'write failed in copy tests'
        }
        #
        $des = $PSScriptRoot + '\data\logfiles2'
        $tools.createdirs($des)
        $desfile = $des + '\logfile.log'
        #
        Write-Host '    testing single copy'
        #
        $tools.copy($sorfile, $des)
        $tools.popfile($sorfile,'edited')
        #
        Write-Host '    testing single checksum'
        #
        $tools.verifyChecksum($sor, $des, '*', 0)
        $this.checkdesfiles($des, $desfile, $tools)
        #
        Write-Host '    testing robo copy'
        #
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile2.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile3.log'
        $tools.popfile($sorfile, $content)
        #
        $tools.copy($sor, $des, 'log')
        $this.checkdesfiles($des, $desfile, $tools)
        #
        Write-Host '    testing robo copy 2'
        #
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile2.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile3.log'
        $tools.popfile($sorfile, $content)
        $tools.copy($sor, $des, 'log')
        #
        Write-Host '    testing checksum multiple files'
        #
        $tools.popfile($sorfile, 'new contents')
        $tools.verifyChecksum($sor, $des, '*', 0)
        $this.checkdesfiles($des, $desfile, $tools, 3)
        #
        Write-Host '    testing checksum on one file'
        #
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile2.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile3.log'
        $tools.popfile($sorfile, $content)
        $tools.copy($sor, $des, 'log')
        $tools.popfile($sorfile, $content)
        $tools.verifyChecksum($sorfile, $des, '*', 0)
        $this.checkdesfiles($des, $desfile, $tools, 3)
        #
        Write-Host '    testing checksum no files in des'
        #
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile2.log'
        $tools.popfile($sorfile, $content)
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile3.log'
        $tools.popfile($sorfile, $content)
        $tools.verifyChecksum($sor, $des, '*', 0)
        $this.checkdesfiles($des, $desfile, $tools, 3)
        #
        Write-Host '    testing robo copy all files'
        #
        $tools.copy($sor, $des, '*')
        #
        $this.checkdesfiles($des, $desfile, $tools)
        #
        Write-Host 'test copy files finished'
        #
    }
    #
    [void]checkdesfiles($des, $desfile, $tools){
        #
        if (!(Test-Path $desfile)){
            Throw 'could not copy using robocopy'
        }
        #
        $files = get-childitem $des
        Write-Host ($files.FullName)
        #
        $tools.removedir($des)
        if (Test-Path $des){
        Throw 'could not remove directory in copy tests'
        }
        #
    }
    #
    [void]checkdesfiles($des, $desfile, $tools, $n){
        #
        if (!(Test-Path $desfile)){
            Throw 'could not copy using robocopy'
        }
        #
        $files = get-childitem $des -File
        Write-Host ($files.FullName)
        $nfiles = ($files).Length
        if ($nfiles -ne $n){
            Throw 'wrong number of files for test'

        }
        $tools.removedir($des)
        if (Test-Path $des){
        Throw 'could not remove directory in copy tests'
        }
        #
    }
    #
    [void]testfilewatchers($tools){
        #
        Write-Host '.'
        Write-Host 'test file watcher started'
        #
        # $sor = $PSScriptRoot + '\data\logfiles'
        $sorfile = $PSScriptRoot + '\data\logfiles\logfile.log'
        $content = 'log file contents'
        $tools.popfile($sorfile, $content)
        #
    }
    #
    [void]testcondaenvir($tools){
        #
        Write-Host '.'
        Write-Host 'test that the conda enviroment can be imported'
        #
        Write-Host '    is Windows: ' $tools.isWindows()
        #
        if ($tools.isWindows()){
            $tools.CheckConda()
            if ((get-module).name -notcontains 'Conda'){
                Throw 'Conda not installed correctly'
            }
        } else {
            #
            Write-Host '    OS is not windows test that we can run astropath from python'
            #
            write-host '1'
            write-host (Invoke-Expression 'meanimagesample -h')
            write-host '2'
            write-host (& meanimagesample -h)
            #
            $pymod = "from astropath.hpfs.flatfield.meanimagesample import MeanImageSample"
            $pyargs = "args = ['-h']"
            $testval = $pymod, $pyargs, "MeanImageSample.runfromargumentparser(args=args)" -join "; "
            <#
            $pymod = "import astropath"
            $pyargs = "args = ['-h']"
            $testval = $pymod, $pyargs, "astropath.hpfs.flatfield.MeanImageCohort.runfromargumentparser(args=args)" -join "; "
            #>
            Write-Host '   '$pymod
            Write-Host '   '$pyargs
            Write-Host '   '$testval
            #
            $output = python -c $testval
            Write-Host $output
            <#
            if ($output -notmatch 'usage'){
                Throw 'error launching py test wihout conda'
            }
            #>
            #
            Write-Host '    python finished'
            #     
        }
        #
    }
}

#
# launch test and exit if no error found
#
[testpssharedtools]::new() | Out-Null
exit 0
