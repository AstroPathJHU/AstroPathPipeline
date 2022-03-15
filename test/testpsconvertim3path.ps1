<# -------------------------------------------
 testpsconvertim3path
 Benjamin Green - JHU
 Last Edit: 02.09.2022
 --------------------------------------------
 Description
 test if the module can be imported or not
 -------------------------------------------#>
#
Class testpsconvertim3path {
    #
    [string]$apmodule = $PSScriptRoot + '/../astropath'
    [string]$basepath
    [string]$flatwpath
    [string]$slideid
    [string]$scanfolder
    [string]$dryrun
    #
    testpsconvertim3path($dryrun){
        #
        $this.basepath = '\\bki04\Clinical_Specimen'
        $this.flatwpath = '\\bki08\e$\astropath_ws\test'
        $this.slideid = 'M21_1'
        $this.scanfolder = $this.basepath + '\' + $this.slideid + '\im3\Scan1\MSI'
        $this.dryrun = $true
        $this.launchtests()
        #
    }
    [void]launchtests(){
        Write-Host '---------------------test [convertim3path]---------------------'
        $this.importmodule()
        $this.launchshred()
        $this.testnormal()
        $this.testzero()
        $this.testzero2()
        Write-Host '.'
        #
    }
    #
    [void]importmodule(){
        Write-Host '.'
        Write-Host 'importing astropath ... '
        Import-Module $this.apmodule
    }
    #
    [void]launchshred(){
        ConvertIM3Path $this.basepath $this.flatwpath $this.slideid -shred -dat -verbose
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
        $dest = ($this.flatwpath + '\' + $this.slideid)
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
        $outputimages = Get-ChildItem ($this.flatwpath + '\' + $this.slideid + '\*') '*Data.dat'   
        Write-Host '    remove:' $outputimages[0].FullName
        remove-item -LiteralPath $outputimages[0].FullName -Force -EA Stop
        Write-Host '    create file'
        new-item -Path $outputimages[0].Directory -Name $outputimages[0].Name -ItemType 'file' -EA Stop
        #
        $filespec = '.Data.dat'
        $dest = ($this.flatwpath + '\' + $this.slideid)
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
        $outputimages = Get-ChildItem ($this.flatwpath + '\' + $this.slideid + '\*') '*Data.dat'   
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
        $dest = ($this.flatwpath + '\' + $this.slideid)
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
}
#
# launch test and exit if no error found
#
[testpsconvertim3path]::new($true) | Out-Null
exit 0
