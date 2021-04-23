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
param ([Parameter(Position=0)][string] $root1 = '',
       [Parameter(Position=1)][string] $root2 = '', 
       [Parameter(Position=2)][string] $sample = '',
       [Parameter()][switch]$i,
       [Parameter()][switch]$s, 
       [Parameter()][switch]$a,
       [Parameter()][switch]$xml,
       [Parameter()][switch]$d)
#
# check input parameters
#
if (
    !($PSBoundParameters.ContainsKey('root1')) -OR 
    !($PSBoundParameters.ContainsKey('root2')) -OR 
    !($PSBoundParameters.ContainsKey('sample')) -OR
    (!($i) -AND !($s))
    ) {
    Write-Host "Usage: ConvertIm3Path dataroot dest sample -s [-a -d -xml]"
    Write-Host "Usage: ConvertIm3Path dataroot dest sample -i"; return
    }
#
function Im3ConvertPath{ 
    #
    param ([Parameter(Position=0)][string] $root1 = '',
           [Parameter(Position=1)][string] $root2 = '', 
           [Parameter(Position=2)][string] $sample = '',
           [Parameter()][switch]$i,
           [Parameter()][switch]$s, 
           [Parameter()][switch]$a,
           [Parameter()][switch]$xml,
           [Parameter()][switch]$d)
    #
    # set default to all for shred if no other value given
    #
    if ($s -and !$a -and !$d -and !$xml) { $a = $true }
    #
    # if option is set for inject send a warning message as the option params are not valid
    #
    if ($i) {
        #
        if ($a) {
            Write-Host "WARNING: '-a' not valid for inject. IGNORING"
        } elseif ($d) {
            Write-Host "WARNING: '-d' not valid for option inject. IGNORING"
        } elseif ($xml) {
            Write-Host "WARNING: '-xml' not valid for option inject. IGNORING"
        }
        #
    }
    #
    # find highest scan folder, exit if im3 directory not found
    #
    $IM3 = "$root1\$sample\im3"
    if (!(test-path $IM3)) { 
        Write-Host "IM3 root path $IM3 not found"; return
        }
    #
    $sub = gci $IM3 -Directory
        foreach ($sub1 in $sub) {
            if($sub1.Name -like "Scan*") { 
                $scan = $IM3 + "\" + $sub1.Name
            }
        }
    #
    # build full im3 path, exit if not found
    #
    $IM3 = "$scan\MSI"
    if (!(test-path $IM3)) { 
        Write-Host "IM3 subpath $IM3 not found"; return
        }
    #
    # build flatw path, and create folders if they do not exist for shred
    # exit if not found on inject
    #
    $flatw = "$root2\$sample"
    if (!(test-path $flatw) -and !$i) {
        new-item $flatw -itemtype directory | Out-Null
    } elseif (!(test-path $flatw) -and $i){
        Write-Host "flatw path $flatw not found"; return
    }
    #
    Write-To-Log -root1 $root1 -root2 $root2 -sample $sample `
                 -IM3_fd $IM3 -Start -s:$s -a:$a -d:$d -xml:$xml
    #
    $images = gci "$IM3\*" '*.im3'
    if (!($images.Count -eq 0) -and $s) {
        #
        # for shred: extract the bit map and xml for each image. 
        # then extract the full xml and rename to 'Full.xml' and 
        # extract additional sample information like shape, etc
        # optional inputs are applied
        #
        if ($a -or $d) { Run-IM3Convert $images $flatw -BIN }
        #
        if ($a -or $xml) {
            Run-IM3Convert $images $flatw -XML
            Run-IM3Convert $images $flatw -FULL
            Run-IM3Convert $images $flatw -PARMS
        }
        #
    } elseif (!($images.Count -eq 0) -and $i) {
        #
        # for inject check for '.dat' files then inject
        # back to im3 into the flatw folder
        #
        $dats = gci "$flatw\*" '*.fw'
        if (!($dats.Count -eq $images.Count)) { 
            Write-Host "$flatw\*.fw N File(s) and $IM3\*im3 N File(s) do not match"
            #return 
        }
        #
        $dest = "$root1\$sample\im3\flatw"
        if (!(test-path $dest)) {
            new-item $dest -itemtype directory | Out-Null
        }
        #
        Run-IM3Convert $images "$root1\$sample\im3\flatw" -i -IM3 $IM3 -flatw $flatw
        # 
    } 
    #
    Write-To-Log -root1 $root1 -root2 $root2 -sample $sample `
                 -Finish -s:$s -a:$a -d:$d -xml:$xml
    #
}
#
function Write-To-Log {
    <# ----------------------------------------------------- 
    # Part of the shredPath workflow. This function
    # writes to the log using either a -Start or -Finish Switch
    #
    # Usage: Write-To-Log -Start OR Write-To-Log -Finish
    #
    # ----------------------------------------------------- #>
    [CmdletBinding(PositionalBinding=$false)]
    #
    param([parameter(Mandatory=$false)][String[]]$root1,
          [parameter(Mandatory=$false)][String[]]$root2,
          [parameter(Mandatory=$false)][String[]]$sample,
          [parameter(Mandatory=$false)][String[]]$IM3_fd,
          [parameter(Mandatory=$false)][Switch]$Start,
          [parameter(Mandatory=$false)][Switch]$Finish,
          [parameter(Mandatory=$false)][Switch]$s,
          [parameter(Mandatory=$false)][Switch]$a,
          [parameter(Mandatory=$false)][Switch]$d,
          [parameter(Mandatory=$false)][Switch]$xml)
    #
    # if Start switch is active write the start error messaging for shred
    #
    if ($Start) {
        #
        Write-Host '.'
        #
        if ($s) {
            #
            Write-Host 'shredPath' $root1 $root2 $sample
            If (test-path "$root2\$sample\doShred.log") {
                 Remove-Item "$root2\$sample\doShred.log" -Force
                 }
            #
        } else {
            #
            Write-Host 'injectPath' $root1 $root2 $sample
            If (test-path "$root1\$sample\im3\flatw\doInject.log") {
                 Remove-Item "$root1\$sample\im3\flatw\doInject.log" -Force
                 }
            #
        }
        #
        Write-Host " " (get-date).ToString('T')
        #
        if (!$s) {
            Write-Host "  src path $root2\$sample"
            $stats = gci "$root2\$sample\*" '*.fw' | Measure Length -s
            Write-Host '     ' $stats.Count 'File(s)' $stats.Sum 'bytes'
        }
        #
        Write-Host "  im3 path $IM3_fd"
        $stats = gci "$IM3_fd\*" '*.im3' | Measure Length -s
        Write-Host '     ' $stats.Count 'File(s)' $stats.Sum 'bytes'
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
        Write-Host "  dst path $dest"
        #
        if ($s) {
            #
            if($a -or $d) {
                $stats = gci "$dest\*" '*.dat' | Measure Length -s
                Write-Host '     ' $stats.Count 'File(s)' $stats.Sum 'bytes'
            }
            #
            if ($a -or $xml){
                $stats = gci "$dest\*" '*.xml' | Measure Length -s
                Write-Host '     ' $stats.Count 'File(s)' $stats.Sum 'bytes'
            }
            #
        } else {
            $stats = gci "$dest\*" '*.im3' | Measure Length -s
            Write-Host '     ' $stats.Count 'File(s)' $stats.Sum 'bytes'
            #
        }
        #
        Write-Host " " (get-date).ToString('T')
        #
    }
    #
}
#
function Run-IM3Convert {
    <# ----------------------------------------------------- 
    # Part of the shredPath workflow. This function
    # runs the IM3Convert utility for each of the different
    # instances desired
    #
    # ----------------------------------------------------- #>
    param([parameter(Position=0)][String[]]$images,
          [parameter(Position=1)][String[]]$dest,
          [parameter(Mandatory=$false)][Switch]$BIN,
          [parameter(Mandatory=$false)][Switch]$XML,
          [parameter(Mandatory=$false)][Switch]$FULL,
          [parameter(Mandatory=$false)][Switch]$PARMS, 
          [parameter(Mandatory=$false)][Switch]$i,
          [parameter(Mandatory=$false)][String[]]$IM3,
          [parameter(Mandatory=$false)][String[]]$flatw)
    #
    # Set up variables
    #
    $code = "$PSScriptRoot\ConvertIm3.exe"
    $dat = ".//D[@name='Data']/text()"
    $exp = '"' + ".//G[@name='SpectralBasisInfo']//D[@name='Exposure'] | " + 
            "(.//G[@name='Protocol']//G[@name='DarkCurrentSettings'])" + '"'
    $glb_prms =  '"' + "//D[@name='Shape']  | " +
                 "//D[@name='SampleLocation'] | " +
                 "//D[@name='MillimetersPerPixel'] | " +
                 "(.//G[@name='Protocol']//G[@name='CameraState'])[1]" + '"'
    $inject = ".//D[@name='Data']/text()"
    #
    # extracts the binary bit map
    #
    if ($BIN) {
        $images | foreach-object {
            & $code $_ DAT -x $dat -o $dest 2>&1>> "$dest\doShred.log"
        }
    }
    #
    # extracts the xml file for the exposure times
    #
    if ($XML) {
        $images | foreach-object {
            #& $code $_ XML -t 64 -o $dest 2>&1>> "$dest\doShred.log"
            & $code $_ XML -x $exp -o $dest 2>&1>> "$dest\doShred.log"
        } 
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
        $f = (gci "$dest\*].xml")[0].Name
        $f2 = "$dest\$sample.Full.xml"
        if (test-path $f2) {Remove-Item $f2 -Force}
        Rename-Item "$dest\*].xml" $f2 -Force
        Add-Content "$dest\doShred.log" "$f Renamed to $sample.Full.xml"
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
        $f = (gci "$dest\*State.xml")[0].Name
        $f2 = "$dest\$sample.Parameters.xml"
        if (test-path $f2) {Remove-Item $f2 -Force}
        Rename-Item "$dest\*State.xml" $f2 -Force
        Add-Content "$dest\doShred.log" "$f Renamed to $sample.Parameters.xml"
        #
    }
    #
    # for inject switch inject the .dat back into the flatw files
    # 
    if ($i) {
        #
        $images | foreach-object {
            #
            $in = $_.Replace($IM3, $flatw)
            $in = $in.Replace('.im3', '.Data.dat')
            #
            & $code $_ IM3 -x $inject -i $in -o $dest 2>&1>> "$dest\doInject.log"
            # 
            $f2 = $_.replace($IM3, $dest)
            $f = $f2.replace('.im3', '.injected.im3')
            $f2log = $f2.replace("$dest\", '') 
            if (test-path -LiteralPath $f2) {Remove-Item -LiteralPath $f2 -Force}
            Rename-Item -LiteralPath $f $f2 -Force
            Add-Content "$dest\doInject.log" "$f Renamed to $f2log"
            #
        }
        #
    }
    #
}
#
# run the function
#
Im3ConvertPath $root1 $root2 $sample -i:$i -s:$s -a:$a -d:$d -xml:$xml