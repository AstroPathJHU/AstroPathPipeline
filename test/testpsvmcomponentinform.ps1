using module .\testtools.psm1
<# -------------------------------------------
 testvmcomponentinform
 created by: Andrew Jorquera
 Last Edit: 12.13.2022
 --------------------------------------------
 Description
 test if the methods of vmcomponentinform are 
 functioning as intended
 -------------------------------------------#>
#
Class testpsvmcomponentinform : testtools {
    #
    [string]$module = 'vmcomponentinform'
    [string]$outpath = "C:\Users\Public\BatchProcessing"
    [string]$referenceim3
    [string]$protocolcopy
    [string]$placeholder
    [switch]$jenkins = $false
    [switch]$versioncheck = $false
    [string]$class = 'vmcomponentinform'
    [string]$informantibody = 'Component'
    [string]$informproject = 'Component_08.ifr'
    #
    testpsvmcomponentinform() : base(){
        #
        $this.launchtests()
        #
    }
    testpsvmcomponentinform($jenkins) : base(){
        #
        $this.jenkins = $true
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $this.testvmcomponentinformconstruction($this.task)
        $inp = vmcomponentinform $this.task
        $this.setupjenkinspaths($inp)
        $this.comparevmcomponentinforminput($inp)
        Write-Host '.'
        #
    }
    <# --------------------------------------------
    setupjenkinspaths
    set up output paths for when tests are being 
    run on jenkins
    --------------------------------------------#>
    [void]setupjenkinspaths($inp){
        
        if ($this.jenkins) {
            $this.outpath = $this.basepath + '\..\test_for_jenkins\BatchProcessing'
            $inp.outpath = $this.basepath + '\..\test_for_jenkins\BatchProcessing'
            $inp.informoutpath = $this.outpath + '\' + $this.informantibody + '_0'
            $inp.image_list_file = $this.outpath + '\image_list.tmp'
            $inp.informprocesserrorlog =  $this.outpath + "\informprocesserror.log"
            $inp.processvars[0] = $this.outpath
            $inp.processvars[1] = $this.outpath
            $inp.processvars[2] = $this.outpath
        }
        $this.protocolcopy = $this.basepath + '\..\test_for_jenkins\testing_vmcomponentinform'
        $inp.islocal = $false
        $inp.inputimagepath = $inp.outpath + '\' + $inp.sample.slideid + '\im3\flatw'
        $this.placeholder = $this.basepath + '\..\test_for_jenkins\testing_vmcomponentinform'
    }
    <# --------------------------------------------
    testvmcomponentinformconstruction
    test that the vmcomponentinform object can be constucted
    --------------------------------------------#>
    [void]testvmcomponentinformconstruction($task){
        #
        Write-Host "."
        Write-Host 'test [vmcomponentinform] constructors started'
        try {
            vmcomponentinform $task | Out-Null
        } catch {
            Throw ('[vmcomponentinform] construction with [1] input(s) failed. ' + $_.Exception.Message)
        }
        Write-Host 'test [vmcomponentinform] constructors finished'
        #
    }
    <# --------------------------------------------
    comparevmcomponentinforminput
    check that vmcomponentinform input is what is expected
    from the vmcomponentinform module object
    --------------------------------------------#>
    [void]comparevmcomponentinforminput($inp){
        #
        Write-Host '.'
        Write-Host 'compare [vmcomponentinform] expected input to actual started'
        #
        $informoutpath = $this.outpath, ($this.informantibody + '_0') -join '\'
        $md_imageloc = $this.outpath, 'image_list.tmp' -join '\'
        $algpath = $this.basepath, 'tmp_inform_data', 'Project_Development', $this.informproject -join '\'
        $informpath = '"'+"C:\Program Files\Akoya\inForm\" + $this.informvers + "\inForm.exe"+'"'
        $informprocesserrorlog =  $this.outpath, "informprocesserror.log" -join '\'
        #
        $processoutputlog =  $this.outpath + '\processoutput.log'
        $arginput = ' -a',  $algpath, `
                    '-o',  $informoutpath, `
                    '-i', $md_imageloc -join ' '
        #
        [string]$userinformtask = $informpath,
                                  '-NoNewWindow',
                                  '-RedirectStandardError', $informprocesserrorlog,
                                  '-PassThru',
                                  '-ArgumentList',  $arginput,
                                  '*>>', $processoutputlog -join ' '
        #
        $informtask = $inp.getinformtask()
        #
        $this.compareinputs($userinformtask, $informtask)
        #
    }
}
#
# launch test and exit if no error found
#
#[testpsvmcomponentinform]::new() | Out-Null

#
# add $jenkins parameter to constructor if testing on jenkins
#
[testpsvmcomponentinform]::new($jenkins) | Out-Null

exit 0