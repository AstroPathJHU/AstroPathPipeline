using module .\testtools.psm1
<# -------------------------------------------
 testsegmentationcheck
 created by: Andrew Jorquera
 Last Edit: 03.24.2022
 --------------------------------------------
 Description
 test if the methods of segmentation are 
 functioning as intended
 -------------------------------------------#>
#
Class testsegmentationcheck : testtools {
    #
    [string]$module = 'vminform'
    [string]$class = 'segmentationcheck'
    #
    testsegmentationcheck() : base(){
        #
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $task = ($this.basepath, $this.slideid, $this.antibody, $this.algorithm, $this.informver, $this.mpath)
        $inp = vminform $task
        $this.testsegmentation($inp)
        Write-Host '.'
    }
    <# --------------------------------------------
    testsegmentation
    test that the checking of inform files output
    from the expected outcome works correctly
    --------------------------------------------#>
    [void]testsegmentation($inp) {
        #
        Write-Host '.'
        Write-Host 'test segmentation check started'
        <#
        Write-Host '    error number at start:' $inp.err
        $this.setupexpected($inp)
        $inp.StartInForm()
        $inp.WatchBatchInForm()
        Write-Host '    batch process complete'
        #>
        Write-Host '--------------------'
        $inp.sample.findantibodies()
        Write-Host '    Anitibodies:' $inp.sample.antibodies
        
        #
        Write-Host 'test segmentation check done'
        #
    }
}
#
# launch test and exit if no error found
#
[testsegmentationcheck]::new() | Out-Null
exit 0