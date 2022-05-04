<#--------------------------------------------------------------------------------------------
ConvertIm3Path.ps1

"shred" or "inject" the im3 files for a whole sample depending on options provided. This code
is part of the flat fielding work flow for the JHU astropath pipeline. 

Created by: Alex Szalay, Benjamin Green - JHU - 04/14/2020

Usage: 
 To "shred" a directory of im3s in the CS format use:
    Im3ConvertPath -dataroot -fwpath -sample -s [-a -d -xml]
    Optional arguements:
	-d: only extract the binary bitmap for each image into the output directory
	-xml: extract the xml information only for each image, xml information includes:
		1) one <sample>.Parameters.xml: sample location, shape, and scale
		2) one <sample>.Full.xml: the full xml of an im3 without the bitmap
		3) an .SpectralBasisInfo.Exposure.xml for each image containing the 
			exposure times of the image
 To "inject" a directory of .fw binary blobs for each image back into the directory of im3s use:
    Im3ConvertPath -datapath -fwpath -sample -i
    Exports the new '.im3s' into the flatw directory
#--------------------------------------------------------------------------------------------#>
function ConvertIm3Path{ 
    #
    param ([Parameter(Position=0)][string] $root1 = '',
           [Parameter(Position=1)][string] $root2 = '', 
           [Parameter(Position=2)][string] $sample = '',
           [Parameter()][array] $images,
           [Parameter()][switch]$inject,
           [Parameter()][switch]$shred, 
           [Parameter()][switch]$all,
           [Parameter()][switch]$xml,
           [Parameter()][switch]$xmlfull,
           [Parameter()][switch]$dat)
    #
    test-convertim3params $PSBoundParameters
    $scan = search-scan $root1 $sample
    $IM3 = search-im3 $scan
    $flatw = search-flatw $root2 $sample -inject:$inject

    #
    write-convertim3log -myparams $PSBoundParameters -IM3_fd $IM3 -Start 
    #
    if (!($PSBoundParameters.ContainsKey('images'))){
        $images = (get-childitem "$IM3\*" '*.im3').FullName
    }
    #
    if ($images.Count -eq 0){
         write-convertim3log -myparams $PSBoundParameters -IM3_fd $IM3 -Finish 
         return
    }
    #
    if ($shred) {
        #
        # for shred: extract the bit map and xml for each image. 
        # then extract the full xml and rename to 'Full.xml' and 
        # extract additional sample information like shape, etc
        # optional inputs are applied
        #
        if ($all -or $dat) { Invoke-IM3Convert -images $images -dest $flatw -BIN }
        #
        if ($all -or $xml) {
            Invoke-IM3Convert $images $flatw -XML
        }
        #
        if ($all -or $xml -or $xmlfull) {
            Invoke-IM3Convert $images $flatw -FULL
            Invoke-IM3Convert $images $flatw -PARMS
        }
        #
    } elseif ($inject) {
        #
        # for inject check for '.dat' files then inject
        # back to im3 into the flatw folder
        #
        $dats = get-childitem "$flatw\*" '*.dat'
        if (!($dats.Count -eq $images.Count)) { 
            Write-Verbose "$flatw\*.fw N File(s) and $IM3\*im3 N File(s) do not match"
        }
        #
        $dest = "$root1\$sample\im3\flatw"
        if (!(test-path $dest)) {
            new-item $dest -itemtype directory | Out-Null
        }
        #
        Invoke-IM3Convert $images "$root1\$sample\im3\flatw" -inject -IM3 $IM3 -flatw $flatw
        # 
    } 
    #
    write-convertim3log -myparams $PSBoundParameters -IM3_fd $IM3 -Finish 
    #
}
#
function test-convertim3params{
    #
    param ([Parameter(Position=0)][hashtable] $myparams)
    #
    if (
        !($myparams.ContainsKey('root1')) -OR 
        !($myparams.ContainsKey('root2')) -OR 
        !($myparams.ContainsKey('sample')) -OR
        (!($myparams.inject) -AND !($myparams.shred))
    ) {
        Throw "Usage: ConvertIm3Path dataroot dest sample -inject -shred:[-all -dat -xml -xmlfull]"
    }
    #
    # set default to all for shred if no other value given
    #
    if ($myparams.shred -and !$myparams.all -and !$myparams.dat -and !$myparams.xml -and !$myparams.xmlfull) { $myparams.all = $true }
    #
    # if option is set for inject send a warning message as the option params are not valid
    #
    if ($myparams.inject) {
        #
        if ($myparams.all) {
            Write-Verbose "WARNING: '-all' not valid for inject. IGNORING"
        } elseif ($myparams.dat) {
            Write-Verbose "WARNING: '-dat' not valid for option inject. IGNORING"
        } elseif ($myparams.xml) {
            Write-Verbose "WARNING: '-xml' not valid for option inject. IGNORING"
        } elseif ($myparams.xmlfull) {
            Write-Verbose "WARNING: '-xmlfull' not valid for option inject. IGNORING"
        }
        #
    }
    #
}
#
function search-scan {
    #
    param ([Parameter(Position=0)][string] $root1 = '',
        [Parameter(Position=2)][string] $sample = '')
    #
    # find highest scan folder, exit if im3 directory not found
    #
    $IM3 = "$root1\$sample\im3"
    if (!(test-path $IM3)) { 
       Throw "IM3 root path $IM3 not found"
        }
    #
    $sub = get-childitem $IM3 -Directory
        foreach ($sub1 in $sub) {
            if($sub1.Name -like "Scan*") { 
                $scan = $IM3 + "\" + $sub1.Name
            }
        }
    #
    return $scan
    #
}
#
function search-im3 {
    #
    param ([Parameter(Position=0)][string] $scan = '')
    #
    # build full im3 path, exit if not found
    #
    $IM3 = "$scan\MSI"
    if (!(test-path $IM3)) { 
        Throw "IM3 subpath $IM3 not found"
    }
    #
    return $IM3
    #
}
#
function search-flatw {
    #
    param ([Parameter(Position=0)][string] $root2 = '',
        [Parameter(Position=2)][string] $sample = '',
        [parameter(Mandatory=$false)][Switch]$inject)
    #
    # build flatw path, and create folders if they do not exist for shred
    # exit if not found on inject
    #
    $flatw = "$root2\$sample"
    if (!(test-path $flatw) -and !$inject) {
        new-item $flatw -itemtype directory | Out-Null
    } elseif (!(test-path $flatw) -and $inject){
        Throw "flatw path $flatw not found"; return
    }
    #
    return $flatw
    #
}
#
function write-convertim3log {
    <# ----------------------------------------------------- 
     Part of the shredPath workflow. This function
     writes to the log using either a -Start or -Finish Switch
    -----------------------------------------------------
     Usage: Write-Log -Start OR Write-Log -Finish
    # ----------------------------------------------------- #>
    [CmdletBinding(PositionalBinding=$false)]
    #
    param([parameter(Mandatory=$false)][hashtable]$myparams,
          [parameter(Mandatory=$false)][String]$IM3_fd,
          [parameter(Mandatory=$false)][Switch]$Start,
          [parameter(Mandatory=$false)][Switch]$Finish)
    #
    $s = $myparams.shred
    $i = $myparams.inject
    $d = $myparams.dat
    $xml = $myparams.xml
    $xmlfull = $myparams.xmlfull
    $a = $myparams.all
    $root1 = $myparams.root1
    $root2 = $myparams.root2
    $sample = $myparams.sample
    #
    # if Start switch is active write the start error messaging for shred
    #
    if ($Start) {
        #
        Write-Verbose ". `r"
        #
        if ($s) {
            #
            $appendargs = @()
            #
            if ($d){
                $appendargs += '-dat'
            }
            #
            if ($xml){
                $appendargs += '-xml'
            }
            #
            if ($xmlfull){
                $appendargs += '-xmlfull'
            }
            #
            if ($all){
                $appendargs += '-all'
            }
            #
            $appendargs = ($appendargs -join ' ')
            Write-Verbose "shredPath $root1 $root2 $sample $appendargs `r"
            If (test-path "$root2\$sample\doShred.log") {
                 Remove-Item "$root2\$sample\doShred.log" -Force
                 }
            #
        } else {
            #
            Write-Verbose "injectPath $root1 $root2 $sample `r"
            If (test-path "$root1\$sample\im3\flatw\doInject.log") {
                 Remove-Item "$root1\$sample\im3\flatw\doInject.log" -Force
                 }
            #
        }
        #
        Write-Verbose (" "+(get-date).ToString('T')+"`r")
        #
        if (!$s) {
            Write-Verbose "  src path $root2\$sample `r"
            $stats = get-childitem "$root2\$sample\*" '*.dat' | Measure-Object Length -sum
            Write-Verbose ('     '+$stats.Count+' File(s) '+$stats.Sum+' byte(s)'+"`r")
        }
        #
        Write-Verbose "  im3 path $IM3_fd `r"
        $stats = get-childitem "$IM3_fd\*" '*.im3' | Measure-Object Length -sum
        Write-Verbose ('     '+$stats.Count+' File(s) '+$stats.Sum+' byte(s)'+"`r")
        #
    }
    #
    # if finish switch is active write the finish error messaging for 
    #    
    if ($Finish){
        #
        if ($s) { $dest = "$root2\$sample"
        } else { $dest = "$root1\$sample\im3\flatw" }
        #
        Write-Verbose "  dst path $dest `r"
        #
        if ($s) {
            #
            if($a -or $d) {
                $stats = get-childitem "$dest\*" '*.dat' | Measure-Object Length -sum
                Write-Verbose ('     '+$stats.Count+' File(s) '+$stats.Sum+' byte(s)'+"`r")
            }
            #
            if ($a -or $xml -or $xmlFull){
                $stats = get-childitem "$dest\*" '*.xml' | Measure-Object Length -sum
                Write-Verbose ('     '+$stats.Count+' File(s) '+$stats.Sum+' byte(s)'+"`r")
            }
            #
        } else {
            $stats = get-childitem "$dest\*" '*.im3' | Measure-Object Length -sum
            Write-Verbose ('     '+$stats.Count+' File(s) '+$stats.Sum+' byte(s)'+"`r")
            #
        }
        #
        Write-Verbose (" "+(get-date).ToString('T')+"`r")
        # 
    }
    #
}
#
function search-imagenames{
    #
    param([parameter(Position=0)][String[]]$IM3,
          [parameter(Position=1)][String[]]$images
        )
    #

    #
    return $images
    #
}
#
function Invoke-IM3Convert {
    <# ----------------------------------------------------- 
    # Part of the shredPath workflow. This function
    # runs the IM3Convert utility for each of the different
    # instances desired
    #
    # ----------------------------------------------------- #>
    param([parameter(Position=0)][array]$images,
          [parameter(Position=1)][String]$dest,
          [parameter(Mandatory=$false)][Switch]$BIN,
          [parameter(Mandatory=$false)][Switch]$XML,
          [parameter(Mandatory=$false)][Switch]$FULL,
          [parameter(Mandatory=$false)][Switch]$PARMS, 
          [parameter(Mandatory=$false)][Switch]$inject,
          [parameter(Mandatory=$false)][String[]]$IM3,
          [parameter(Mandatory=$false)][String[]]$flatw)
    #
    # Set up variables
    #
    $code = "$PSScriptRoot\ConvertIm3.exe"
    $dat = ".//D[@name='Data']/text()"
    $exp = '"' + ".//G[@name='SpectralBasisInfo']//D[@name='Exposure'] " + '"' #| " + 
            # "(.//G[@name='Protocol']//G[@name='DarkCurrentSettings'])" + '"'
    $glb_prms =  '"' + "//D[@name='Shape']  | " +
                 "//D[@name='SampleLocation'] | " +
                 "//D[@name='MillimetersPerPixel'] | " +
                 "(.//G[@name='Protocol']//G[@name='CameraState'])[1]" + '"'
    $injecttxt = ".//D[@name='Data']/text()"
    #
    # extracts the binary bit map
    #
    if ($BIN) {
        #
        $log = $dest + '\doShred.log'
        $cnt = 0
        #
        while($images -and ($cnt -lt 5)){
            #
            Write-Debug ('       attempt:' + $cnt)
            #
            $images | foreach-object -Parallel {
                & $using:code $_ DAT -x $using:dat -o $using:dest # 2>&1>> $log
            } -ThrottleLimit 5| Out-File -append $log
            #
            Start-Sleep 2
            #
            $images = SEARCH-FAILED $images $dest '.Data.dat' 
            $cnt += 1
        }
        #
    }
    #
    # extracts the xml file for the exposure times
    #
    if ($XML) {
        #
        $log = $dest + '\doShred.log'
        $cnt = 0
        #
        while($images -and ($cnt -lt 5)){
            #
            Write-Debug ('       attempt:' + $cnt)
            #
            $images | foreach-object -Parallel {
                & $using:code $_ XML -x $using:exp -o $using:dest # 2>&1>> $log
            } -ThrottleLimit 5| Out-File -append $log
            #
            Start-Sleep 2
            #
            $images = SEARCH-FAILED $images $dest '.SpectralBasisInfo.Exposure.xml' 
            $cnt += 1
        }
        #
    }
    #
    # for full switch extract the full xml from the first IM3 
    # in the directory
    #
    if ($FULL){
        #
        $im1 = $images[0]
        & $code $im1 XML -t 64 -o $dest 2>&1>> "$dest\doShred.log"
        #
        $f = (get-childitem "$dest\*].xml")[0].Name
        $f2 = "$dest\$sample.Full.xml"
        if (test-path $f2) {Remove-Item $f2 -Force}
        Rename-Item "$dest\*].xml" $f2 -Force
        "$f Renamed to $sample.Full.xml" | Out-File "$dest\doShred.log" -Append
        #
    }
    #
    # for parms switch extract the global parameters from the first IM3
    # in the directory
    #
    if ($PARMS) {
        #
        $im1 = $images[0]
        & $code $im1 XML -x $glb_prms -o $dest 2>&1>> "$dest\doShred.log"
        # 
        $f = (get-childitem "$dest\*State.xml")[0].Name
        $f2 = "$dest\$sample.Parameters.xml"
        if (test-path $f2) {Remove-Item $f2 -Force}
        Rename-Item "$dest\*State.xml" $f2 -Force
        "$f Renamed to $sample.Parameters.xml" | Out-File "$dest\doShred.log" -Append
        #
    }
    #
    # for inject switch inject the .dat back into the flatw files
    # 
    if ($inject) {
        #
        $log = $dest + '\doInject.log'
        $cnt = 0
        #
        $savedimagenames = $images
        #
        while($images -and ($cnt -lt 5)){
            #
            Write-Debug ('       attempt:' + $cnt)
            #
            $images | foreach-object -Parallel {
                #
                $in = $_.Replace($using:IM3, $using:flatw)
                $in = $in.Replace('.im3', '.Data.dat')
                #
                & $using:code $_ IM3 -x $using:injecttxt -i $in -o $using:dest # 2>&1>> $log
            } -ThrottleLimit 5| Out-File -append $log
            #
            Start-Sleep 2
            #
            $images = SEARCH-FAILED $images $dest '.injected.im3' 
            $cnt += 1
        }
        #
        $savedimagenames | foreach-object {
            #
            # renamed injected.im3s to im3s
            #    
            $f2 = $_.replace($IM3, $dest)
            $f = $f2.replace('.im3', '.injected.im3')
            $f2log = $f2.replace("$dest\", '') 
            if (test-path -LiteralPath $f2) {Remove-Item -LiteralPath $f2 -Force}
            Rename-Item -LiteralPath $f $f2 -Force
            "$f Renamed to $f2log" | Out-File $log -Append
            #
            # renamed Data.dat to .fw
            #    
            $f2 = $_.replace($IM3, $flatw)
            $f = $f2.replace('.im3', '.Data.dat')
            $f2 = $f2.replace('.im3', '.fw')
            #
            $f2log = $f2.replace("$flatw\", '') 
            if (test-path -LiteralPath $f2) {Remove-Item -LiteralPath $f2 -Force}
            Rename-Item -LiteralPath $f $f2 -Force
            "$f Renamed to $f2log" | Out-File $log -Append
            #
        }
        #
    }
    #
}
#
function SEARCH-FAILED {
    #
    param([parameter(Position=0)][array]$images,
    [parameter(Position=1)][String]$dest,
    [parameter(Position=2)][String]$filespec)
    #
    # find images that did not extract at all
    #
    $output = Get-ChildItem ($dest + '\*') ('*' + $filespec)
    if (!$output){
        return $images
    }
    #
    write-debug '       search failed'
    #
    $outputnames = $output.Name
    $compareimagenames = (Split-Path $images -Leaf) -replace '.im3', $filespec
    $imagepath = Split-Path $images[0]
    #
    write-debug ('        filespec: ' + $filespec)
    write-debug ('        comparename ex: ' + $compareimagenames[0])
    write-debug ('        expected ex: ' + $outputnames[0])
    #
    $comparison = Compare-Object -ReferenceObject $compareimagenames `
        -DifferenceObject $outputnames |
        Where-Object -FilterScript {$_.SideIndicator -eq '<='}
    #
    [array]$outimages = @()
    #
    if ($comparison.InputObject){
        ($comparison.InputObject) | foreach-Object{
            $outimages += $imagepath + '\' + ($_  -replace $filespec, '.im3')
        }
    }
    #
    write-debug ('        n files after not found file check: ' + $outimages.Length)
    #
    # Find potential corrupt files
    #
    if ($filespec -match '.Data.dat'){
        #
        if (($output | measure-object length  -maximum).maximum -gt 200000kb){
            $min = 220000kb
        } else {
            $min = 90000kb
        }
    } elseif ($filespec -match '.SpectralBasisInfo.Exposure.xml') {
        $min = 500
    } else {
        $min = 90000kb
    }
    #
    Write-Debug ('        min size filter: ' + $min)
    Write-Debug ('        min file size: ' + ($output | measure-object length  -Minimum).Minimum)
    #
    $filteredoutput = $output | Where-Object {$_.Length -lt $min}
    $outimages += (($filteredoutput -replace [regex]::escape($dest), $imagepath) `
        -replace $filespec, '.im3')
    #
    write-debug ('        n files after wrong file size check: ' + $outimages.Length)
    #
    return $outimages
    #
}
