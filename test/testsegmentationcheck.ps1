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
        Write-Host '    Binary Targets:' $inp.sample.binarysegtargets
        Write-Host '    Component Target:' $inp.sample.componenttarget
        Write-Host 'test segmentation check done'
        #
    }
    <# --------------------------------------------
    testsegmentation
    test that the transferring the merge config
    file from excel to csv works correctly
    --------------------------------------------#>
    [void]testmergeconfigtocsv($inp) {
        #
        Write-Host '.'
        Write-Host 'test merge config to csv started'
        #
        $testbatchid = 1
        $inp.sample.MergeConfigToCSV($inp.sample.basepath, $testbatchid)
        #
        Write-Host 'test merge config to csv done'
        #
    }
}
#
# launch test and exit if no error found
#
[testsegmentationcheck]::new() | Out-Null
exit 0