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
    [string]$module = 'meanimage'
    [string]$class = 'segmentationcheck'
    [string]$antibody = 'CD8'
    [string]$algorithm = 'CD8_Prototype.ifr'
    [string]$informver = '2.4.8'
    #
    testsegmentationcheck() : base(){
        #
        $this.launchtests()
        #
    }
    #
    [void]launchtests(){
        #
        $task = ($this.project, $this.slideid, $this.processloc, $this.mpath)
        $inp = meanimage $task
        $this.testfindsegmentationtargets($inp)
        Write-Host '.'
    }
    <# --------------------------------------------
    testsegmentation
    test that the checking of inform files output
    from the expected outcome works correctly
    --------------------------------------------#>
    [void]testfindsegmentationtargets($inp) {
        #
        Write-Host '.'
        Write-Host 'test segmentation check started'
        Write-Host '--------------------'
        $inp.sample.findsegmentationtargets()
        Write-Host '    Segmentation Targets:' $inp.sample.segmentationtargets
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