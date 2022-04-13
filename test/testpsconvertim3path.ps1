using module .\testtools.psm1
<# -------------------------------------------
 testpsconvertim3path
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
Class testpsconvertim3path : testtools{
    #
    [string]$module = 'convertim3path'
    [string]$scanfolder
    [string]$class = 'convertim3path'
    [string]$im3folderexten = '\im3\Scan1\MSI'
    #
    testpsconvertim3path(){
        $this.launchtests()
    }
    #
    testpsconvertim3path($dryrun) : base ('1', 'M21_1', $dryrun){
        #
        $this.dryrun = $true
        $this.basepath = '\\bki04\Clinical_Specimen'
        $this.launchtests()
        #
    }
    [void]launchtests(){
        #
        $this.scanfolder = $this.basepath + '\' + $this.slideid + $this.im3folderexten
        #
        $this.cleanup()
        $this.launchshreddat()
        $this.testnormal()
        $this.testzero()
        $this.testzero2()
        $this.cleanup()
        #
        $this.launchshredfullxml()
        $this.cleanup()
        $this.launchshredxml()
        $this.cleanup()
        $this.launchshredall()
        $this.cleanup()
        #
        $this.launchinject()
        $this.cleanup()
        #
        Write-Host '.'
        #
    }
    #
    [void]launchshreddat(){
        ConvertIM3Path $this.basepath $this.processloc $this.slideid -shred -dat -verbose -debug
    }
    #
    [void]launchshredxml(){
        ConvertIM3Path $this.basepath $this.processloc $this.slideid -shred -xml -verbose -debug
    }
    #
    [void]launchshredfullxml(){
        ConvertIM3Path $this.basepath $this.processloc $this.slideid -shred -xmlfull -verbose -debug
    }
    #
    [void]launchshredall(){
        ConvertIM3Path $this.basepath $this.processloc $this.slideid -shred -all -verbose -debug
    }
    #
    [void]launchinject(){
        #
        write-host '.'
        write-host 'test inject started'
        #
        $des = $this.processloc + '\test_data\' + 
            $this.slideid + $this.im3folderexten
        write-host '    copy' $this.scanfolder
        write-host '    to:' $des
        if (!(test-path $des)){
            new-item $des -ItemType 'directory'
        }
        robocopy $this.scanfolder $des -r:3 -w:3 -np -E -mt:10
        #
        $this.basepath = $this.processloc + '\test_data'
        $this.launchshreddat()
        #
        ConvertIM3Path $this.basepath $this.processloc $this.slideid -inject -verbose -debug
        write-host 'test inject finished'
        #
    }
    #
    [void]testnormal(){
        Write-Host '.'
        Write-Host 'test normal execution on [search failed] returns nothing started'
        #
        [array]$images = (Get-ChildItem ($this.scanfolder +'\*') '*.im3').FullName
        #
        Write-Host '    image 1:' $images[0]
        #
        $dest = ($this.processloc + '\' + $this.slideid)
        $filespec = '.Data.dat'
        #
        $output = Get-ChildItem ($dest + '\*') ('*' + $filespec)
        $outputnames = $output.Name
        Write-Host '    output name 1:' $outputnames[0]
        #
        Write-Host '    running SEARCH-FAILED'
        $outimages = Search-Failed $images $dest '.Data.dat'
        Write-Host '    output:' $outimages
        #
        if ($outimages){
            Throw 'outimages is not empty'
        }
        #
        Write-Host '.'
        Write-Host 'test normal execution on [search failed] returns nothing finished'
    }
    #
    [void]testzero(){
        #
        Write-Host '.'
        Write-Host 'test that the zero byte dats are found correctly started'
        #
        $outputimages = Get-ChildItem ($this.processloc + '\' + $this.slideid + '\*') '*Data.dat'   
        Write-Host '    remove:' $outputimages[0].FullName
        remove-item -LiteralPath $outputimages[0].FullName -Force -EA Stop
        Write-Host '    create file'
        new-item -Path $outputimages[0].Directory -Name $outputimages[0].Name -ItemType 'file' -EA Stop
        #
        $filespec = '.Data.dat'
        $dest = ($this.processloc + '\' + $this.slideid)
        [array]$images = (Get-ChildItem ($this.scanfolder +'\*') '*.im3').FullName
        Write-Host '    image 1:' $images[0]
        #
        $output = Get-ChildItem ($dest + '\*') ('*' + $filespec)
        $outputnames = $output.Name
        $imagepath = Split-Path $images[0]
        Write-Host '    output name 1:' $outputnames[0]
        #
        Write-Host '    filespec match?' ($filespec -match '.Data.dat')
        #
        $output2 = $output | Where-Object {$_.Length -eq 0kb}
        Write-Host (($output2 -replace [regex]::escape($dest), $imagepath) -replace $filespec, '.im3')
        #
        Write-Host '    running SEARCH-FAILED'
        $outimages = Search-Failed $images $dest '.Data.dat'
        Write-Host '    output:' $outimages
        #
        Write-Host 'test that the zero byte dats are found correctly finished'
        #
    }
    #
    [void]testzero2(){
        #
        Write-Host '.'
        Write-Host 'test that the zero byte dats are found correctly started'
        #
        $outputimages = Get-ChildItem ($this.processloc + '\' + $this.slideid + '\*') '*Data.dat'   
        Write-Host '    remove:' $outputimages[0].FullName
        remove-item -LiteralPath $outputimages[0].FullName -Force -EA Stop
        Write-Host '    create file'
        new-item -Path $outputimages[0].Directory -Name $outputimages[0].Name -ItemType 'file' -EA Stop
        #
        Write-Host '    remove:' $outputimages[1].FullName
        remove-item -LiteralPath $outputimages[1].FullName -Force -EA Stop
        Write-Host '    create file'
        new-item -Path $outputimages[1].Directory -Name $outputimages[1].Name -ItemType 'file' -EA Stop
        #
        $filespec = '.Data.dat'
        $dest = ($this.processloc + '\' + $this.slideid)
        [array]$images = (Get-ChildItem ($this.scanfolder +'\*') '*.im3').FullName
        Write-Host '    image 1:' $images[0]
        #
        $output = Get-ChildItem ($dest + '\*') ('*' + $filespec)
        $outputnames = $output.Name
        $imagepath = Split-Path $images[0]
        Write-Host '    output name 1:' $outputnames[0]
        #
        Write-Host '    filespec match?' ($filespec -match '.Data.dat')
        #
        $output2 = $output | Where-Object {$_.Length -eq 0kb}
        Write-Host (($output2 -replace [regex]::escape($dest), $imagepath) -replace $filespec, '.im3')
        #
        Write-Host '    running SEARCH-FAILED'
        $outimages = Search-Failed $images $dest '.Data.dat'
        Write-Host '    output:' $outimages
        #
        Write-Host 'test that the zero byte dats are found correctly finished'
        #
    }
    #
    [void]cleanup(){
        #
        $dir = $this.processloc
        #
        if (test-path -literalpath $dir){
            Get-ChildItem -Directory $dir | Remove-Item -force -Confirm:$false -recurse
            remove-item $dir -force -Confirm:$false -Recurse
        }
        #
    }
}
#
# launch test and exit if no error found
#
[testpsconvertim3path]::new($true) | Out-Null
exit 0
