﻿using module .\testtools.psm1
<# testpsmoduletools
 testpslogger
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
 Class testpsmoduletools : testtools {
    #
    [string]$class = 'moduletools'
    [string]$module = 'shredxml'
    #
    testpsmoduletools() : base(){
        #
        $this.testmodulecontruction()
        #
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $inp = meanimage $task
        #
        # $this.testslidelist()
        #
        $this.TestPaths($inp)
        Write-Host '.'
        #
    }
    #
    [void]testmodulecontruction(){
        #
        Write-Host '.'
        Write-Host 'building a shredxml module object'
        try {
            $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
            shredxml $task | Out-Null
        } catch {
            Throw 'module could not be constructed'
        }
        #
    }
    #
    [void]TestPaths($inp){
        #
        Write-Host '.'
        Write-Host 'Starting Paths Testing'
        #
        $testloc = $this.processloc + '\astropath_ws\meanimage\' + $this.slideid
        #
        if (!([regex]::Escape($inp.processvars[0]) -contains [regex]::Escape($testloc))){
            Throw ('processvars[0] not correct: ' + $inp.processvars[0] + '~=' + $testloc)
        }
        #
        if (!([regex]::Escape($inp.processvars[1]) -contains [regex]::Escape(($testloc + '\flatw')))){
            Throw ('processvars[1] not correct: ' + $inp.processvars[1] + '~=' + $testloc + '\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[2]) -contains [regex]::Escape(($testloc + '\' + $this.slideid + '\im3\flatw')))){
            Throw ('processvars[2] not correct: ' + $inp.processvars[2] + '~=' + $testloc + '\' + $this.slideid + '\im3\flatw')
        }
        #
        if (!([regex]::Escape($inp.processvars[3]) -contains [regex]::Escape(($testloc + '\flatfield\flatfield_BatchID_08.bin')))){
            Write-Host 'batch flatfield file:' $inp.sample.batchflatfield()
            Throw ('processvars[3] not correct: ' + $inp.processvars[3] + '~=' + $testloc + '\flatfield\flatfield_BatchID_08.bin')
        }
        #
        $sor = $this.basepath, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        $des = $this.processloc, $this.slideid, 'im3\meanimage\image_masking' -join '\'
        #
        Write-Host '   source:' $sor
        Write-Host '   destination:' $des
        #
        $filespec = '*'
        $des1 = $des -replace '\\', '/'
        $sor1 = ($sor -replace '\\', '/')
        $files = $inp.sample.listfiles($sor, $filespec)

        $find = ('"'+$filespec+'"')
        $files = find $sor1 -name $find
        Write-Host '    files:' $files
        mkdir -p $des1
        cp $files -r -t ($des1 + '/')
        #
        Write-host '.'
        $files = find $des1 -name $find
        Write-Host '    files:' $files
        Throw 'stop'
        <#
        Write-Host '   source:' $sor
        Write-Host '   destination:' $des
        $inp.sample.copy($sor, $des, '*')
        #
        if (!(test-path -LiteralPath ($sor + '\.gitignore'))){
            Throw 'da git ignore is not correct in meanimage source'
        }
        #
        if (!(test-path -LiteralPath ($des + '\.gitignore'))){
            Throw 'da git ignore is not correct in meanimage destination'
        }
        #
        $this.comparepaths($sor, $des, $inp)
        $inp.sample.removedir($des)
        #>
        Write-Host 'Passed Paths Testing'
        #
    }
    #
    <#---------------------------------------------
    testslidelist
    ---------------------------------------------#>
    [void]testslidelist(){
        #
        Write-Host "."
        Write-Host 'test building slide list started'
        #
        $task = ($this.project, $this.batchid, $this.processloc, $this.mpath)
        $inp = batchwarpkeys $task  
        #
        Write-Host '    test one batch slidelist no dep'
        $inp.getslideidregex()
        Write-Host '    slides in batch list:'
        Write-Host '   '$inp.batchslides
        Write-Host '    slide id is:' $inp.sample.slideid
        Write-Host '    batch id is:' $this.batchid
        if ($inp.sample.slideid -notcontains $this.batchid.PadLeft(2,'0')){
            Throw 'slide id wrong'
        }
        #
        Write-Host '    test one batch slidelist'
        #
        $this.addwarpoctetsdep($inp)
        #
        $inp.getslideidregex('batchwarpkeys')
        #
        Write-Host '    slides in batch list:'
        Write-Host '   '$inp.batchslides
        Write-Host '    slide id is:' $inp.sample.slideid
        Write-Host '    batch id is:' $this.batchid
        if ($inp.sample.slideid -notcontains $this.batchid.PadLeft(2,'0')){
            Throw 'slide id wrong'
        }
        #
        if (!$inp.batchslides){
            Throw 'no batch slides found!!!'
        }
        #
        Write-Host '    test all slides slidelist'
        $inp.all = $true
        $this.addwarpoctetsdep($inp)
        $inp.getslideidregex('batchwarpkeys')
        Write-Host '    slides in batch list:'
        Write-Host '   '$inp.batchslides
        Write-Host '    slide id is:' $inp.sample.slideid
        Write-Host 'test building slide list finished'
        #
        if (!$inp.batchslides){
            Throw 'no batch slides found!!!'
        }
        #
        $this.removewarpoctetsdep($inp)
        #
        Write-Host 'test building slide list finished'
    }
    #
    [void]addwarpoctetsdep($inp){
        #
        $sor = $this.basepath, 'reference', 'warpingcohort',
         'M21_1-all_overlap_octets.csv' -join '\'
        #
        $inp.getslideidregex()
        #
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping', 'octets' -join '\'
            $inp.sample.copy($sor, $des)
            rename-item ($des + '\M21_1-all_overlap_octets.csv') ($_ + '-all_overlap_octets.csv') -EA stop
        }
    }
    #
    [void]removewarpoctetsdep($inp){
        #
        $inp.getslideidregex()
        $inp.batchslides | ForEach-Object{
            $des = $this.basepath, $_, 'im3', 'warping' -join '\'
            $inp.sample.removedir($des)
        }
    }
    #
}
#
# launch test and exit if no error found
#
[testpsmoduletools]::new() | Out-Null
exit 0
