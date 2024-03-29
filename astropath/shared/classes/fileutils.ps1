﻿<# -------------------------------------------
 fileutils
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to read and write files with 
 error checking and using file locking mutexes
 -------------------------------------------#>
class fileutils : generalutils {
    [INT]$MAX = 500

    <# -----------------------------------------
     OpenCSVFile
     open a csv file with error checking and a 
     file locking mutex into a powershell
     object where each row is a different 
     object and columns are the object fields
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.OpenCSVFile(fpath)
    ----------------------------------------- #>
    [PSCustomObject]OpenCSVFile([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $mxtxid = 'Global\' + $fpath.replace('\', '_').replace('/', '_') + '.LOCK'
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
                $this.createfile($fpath)
                $Q = Import-CSV $fpath -ErrorAction Stop
                $e = 0
            }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $this.MAX) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $this.MAX){
           Throw $cnt.ToString() + ' attempts failed reading ' `
                + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.OpenCSVFile(fpath, headers)
    ----------------------------------------- #>
    [PSCustomObject]OpenCsvFile($fpath, [array]$headers){
        #
        $this.checkexistscreate($fpath, $headers)
        #
        $Q = $this.opencsvfile($fpath)
        #
        return $Q
        #
    }
    <# -----------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.OpenCSVFile(fpath, delim, headers)
    ----------------------------------------- #>
    [PSCustomObject]OpenCSVFile([string] $fpath, [string] $delim, [array] $header){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $mxtxid = 'Global\' + $fpath.replace('\', '_').replace('/', '_') + '.LOCK'
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
                $this.createfile($fpath)
                $Q = Import-CSV $fpath `
                    -Delimiter $delim `
                    -header $header `
                    -ErrorAction Stop
                $e = 0
            }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $this.MAX) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $this.MAX){
           Throw $cnt.ToString() + ' attempts failed reading ' `
                + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
     OpencsvfileConfirm
     Open a csv file with confirmation of data
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.OpencsvfileConfirm(fpath)
    ----------------------------------------- #>
    [PSCustomObject]OpenCSVFileConfirm([string] $fpath){
        #
        $data = $this.OpencsvFile($fpath)
        $e = 1
        #
        while ($e -lt 6 -AND !($data)){
            $data = $this.OpencsvFile($fpath)
            if ($data){
                $e = 6
            }
            $e += 1
            Start-Sleep -s 3
        }
        #
        if (!$data){
            Throw ('File is empty, input expected: ' + $fpath)
        }
        #
        return $data
        #
    }
    #
    [PSCustomObject]OpenCSVFileConfirm([string] $fpath, [array]$headers){
        $this.checkexistscreate($fpath, $headers)
        return $this.opencsvfileconfirm($fpath)
    }
    #
    [void]checkexistscreate($fpath, [array]$headers){
        #
        if (!(test-path $fpath)){
            $this.createcsvfile($fpath, $headers)
        }
        #
    }
    #
    [void]createcsvfile($fpath, [array]$headers){
        #
        $newheaders = ($headers -join ',') + "`r`n"
        $this.setfile($fpath, $newheaders)
        #
    }
    #
    [void]writecsv($fpath, $content){
        #
        $contentout = (($content |
            ConvertTo-Csv -NoTypeInformation) -join "`r`n").Replace('"','') + "`r`n"
        $this.setfile($fpath, $contentout)
        #
    }
    <# -----------------------------------------
     GetContent
     open a file with error checking where each
     row is in a separate line
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.GetContent(fpath)
    ----------------------------------------- #>
    [Array]GetContent([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $mxtxid = 'Global\' + $fpath.replace('\', '_').replace('/', '_') + '.LOCK'
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
                $this.createfile($fpath)
                $Q = Get-Content $fpath -ErrorAction Stop
                $e = 0
            }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
            }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $this.MAX) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $this.MAX){
           Throw $cnt.ToString() + ' attempts failed reading ' `
                + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
        #
    }
    <# -----------------------------------------
     ImportExcel
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
        adopted from: 
        https://www.c-sharpcorner.com/article/read-excel-file-using-psexcel-in-powershell2/
     ------------------------------------------
     Usage: $this.ImportExcel(fpath)
    ----------------------------------------- #>
    [PSCustomObject]ImportExcel($fpath){
        #
        $objExcel = New-Object -ComObject Excel.Application
        $WorkBook = $objExcel.Workbooks.Open($fpath)
        #
        $worksheet = $workbook.WorkSheets(1)
        $columns = $Worksheet.columns.count
        $rows = $Worksheet.Rows.Count
        #
        # get the headers
        #  
        $names = @()
        #
        foreach ($i1 in (1..$columns)){
            $cell = $worksheet.Cells.Item(1, $i1).text.trim()
            if ($cell){
                $names += $cell
            } else {
                break
            }
        }
        #
        $obj = @()
        #
        foreach ($i2 in (2..$rows)){
            $cell = $worksheet.cells.item($i2, 1).text
            if ($cell){
                $obj1 = new-object pscustomobject
                foreach($i3 in (0..($names.count-1))) {
                    $obj1 | Add-Member -NotePropertyName $names[$i3] `
                        -NotePropertyValue $worksheet.cells.item($i2, ($i3 + 1)).text
                }
                #
                $obj += $obj1
            } else { break }
        }
        #
        try{
            $objExcel.Quit()
        } catch {}
        #
        return $obj
        #
    }
    <# -----------------------------------------
     PopFile
     append to the end of a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.PopFile(fpath)
    ----------------------------------------- #>
    [void]PopFile([string] $fpath, [array] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Pop')
        #
    }
    <# -----------------------------------------
     SetFile
     Overwrite an entire file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: $this.SetFile(fpath)
    ----------------------------------------- #>
    [void]SetFile([string] $fpath,[array] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Set')
        #
    }
    <# -----------------------------------------
     HandleWriteFile
     Checks that the file exists and grabs the 
     mutex for a file. When it has the mutex,
     calls the write file method
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
        -fstring[string]: input array
        -opt[sting]: 'Set' or 'Pop'
     ------------------------------------------
     Usage: $this.HandleWriteFile(fpath, fstring, opt)
    ----------------------------------------- #>
    [void]HandleWriteFile([string] $fpath,[array] $fstring, [string] $opt){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $mxtxid = 'Global\' + $fpath.replace('\', '_').replace('/', '_') + '.LOCK'
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
             try{
                $this.createfile($fpath)
                $this.WriteFile($fpath, $fstring, $opt)
                $e = 0
             }catch{
                $err = $_.Exception.Message
                $cnt = $cnt + 1
                Start-Sleep -s 3
                Continue
             }
            $this.ReleaseMxtx($mxtx, $fpath)
            #
        } while(($cnt -lt $this.MAX) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $this.MAX){
            Throw $cnt.ToString() + ' attempts failed writing ' +
                $fstring + ' to ' + $fpath + '. Final message: ' + $err
        }
        #
    }
    <# -----------------------------------------
     WriteFile
     append or overwrite a file depending on 
     input
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
        -fstring[string]:line \ lines to write to the file
        -opt[string]: 'Set' or 'Pop'
     ------------------------------------------
     Usage: $this.WriteFile(fpath, fstring, opt)
    ----------------------------------------- #>
     [void]WriteFile([string]$fpath,[array]$fstring, [string]$opt){
        #
        if ($opt -eq 'Set'){
            Set-Content -LiteralPath $fpath -Value $fstring -NoNewline -EA Stop
        } elseif ($opt -eq 'Pop') {
            Add-Content -LiteralPath $fpath -Value $fstring -NoNewline -EA Stop
        }
     }
    <# -----------------------------------------
     GrabMxtx
     Grab my mutex, from: 
     https://stackoverflow.com/questions/7664490/
        interactively-using-mutexes-et-al-in-powershell
     ------------------------------------------
     Input: 
        -mxtxid: string object for a mutex
     ------------------------------------------
     Usage: $this.GrabMxtx(mxtxid)
    ----------------------------------------- #>
    [System.Threading.Mutex]GrabMxtx([string] $mxtxid){
         try {
            $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
            $count = 0
            while (!$mxtx.WaitOne(1000)) {
                Start-Sleep -m 1000;
                $count ++
                if ($count -eq $this.MAX) {
                    throw "could not grab mutex: $mxtxid"
                }
            }
            return $mxtx
        } catch [System.Threading.AbandonedMutexException] {
            $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
            return $this.GrabMxtx($mxtxid)
        }
    }
    <# -----------------------------------------
     ReleaseMxtx
     release mutex
     ------------------------------------------
     Input: 
        -mxtxid: string object for a mutex
        -fpath[string]: file path
     ------------------------------------------
     Usage: $this.ReleaseMxtx(mxtx, fpath)
    ----------------------------------------- #>
    [void]ReleaseMxtx([System.Threading.Mutex]$mxtx, [string] $fpath){
        try{
            $this.releasemxtx($mxtx)
            $mxtx.Dispose()
        } catch {
            Throw "mutex not released: " + $fpath
        }
    }
    #
    [void]ReleaseMxtx([System.Threading.Mutex]$mxtx){
        $notreleased = $true
        while ($notreleased){
            try {
                $mxtx.ReleaseMutex()
            } catch [System.ApplicationException] {
                $notreleased = $false
                $Error.Clear()
            }
        }
    }
    <# -----------------------------------------
     TaskFileWatcher
     Create a file watcher 
     ------------------------------------------
     Input: 
        -file: full file path
     ------------------------------------------
     Usage: $this.TaskFileWatcher(file, slideid, module)
    ----------------------------------------- #>
    [string]TaskFileWatcher($file, $slideid, $module){
        #
        $fpath = Split-Path $file
        $fname = Split-Path $file -Leaf
        $SI = $module, $slideid -join '-'
        #
        $SI = $this.FileWatcher($fpath, $fname, $SI)
        return $SI
        #
    }
    <# -----------------------------------------
     FileWatcher
     Create a file watcher 
     ------------------------------------------
     Input: 
        -file: full file path
     ------------------------------------------
     Usage: $this.FileWatcher(file)
    ----------------------------------------- #>
    [string]FileWatcher($file){
        #
        $fpath = Split-Path $file
        $fname = Split-Path $file -Leaf
        #
        return($this.FileWatcher($fpath, $fname))
        #
    }
    #
    [string]FileWatcher($fpath, $fname){
        #
        $SI = "$fpath\$fname"
        return (
            $this.filewatcher($fpath, $fname, $this.CrossPlatformPaths($SI))
        )
        #
    }
    #
    [string]FileWatcher($fpath, $fname, $SI){
        #
        $this.createfile($this.CrossPlatformPaths($SI))
        $fpath = $this.CrossPlatformPaths($fpath)
        #
        $testw = $false
        $c = 0
        #
        while (!$testw){
            #
            if ($c -ge 5){
                $this.writeoutput(
                    "WARNING: File watcher was not triggered on test for: $SI")
                break
            } elseif ($c -gt 0){
                $this.writeoutput(
                    "WARNING: File watcher was not triggered on test for: $SI")
                $this.writeoutput(
                    "WARNING: trying file watcher again: $SI")
            }
            #
            $newwatcher = [System.IO.FileSystemWatcher]::new($fpath)
            $newwatcher.Filter = $fname
            $newwatcher.NotifyFilter = [IO.NotifyFilters]'FileName, LastWrite, Attributes'
            #
            Register-ObjectEvent $newwatcher `
                -EventName Changed `
                -SourceIdentifier ($SI + ';changed') | Out-Null
            #
            Register-ObjectEvent $newwatcher `
                -EventName Renamed `
                -SourceIdentifier ($SI + ';renamed')| Out-Null
            #
            $testw = $this.testwatcher($fpath, $fname, $SI)
            #
            $c ++ 
            #
        }
        #
        return $SI
        #
    }
    #
    [switch]testwatcher($fpath, $fname, $SI){
        #
        $file = Get-ChildItem ($this.CrossPlatformPaths($SI))
        try {
            $file.Attributes = 'Archive, Hidden'
            $file.Attributes = 'Archive'
        } catch {
            Write-Host 'Failure to find file:' $this.CrossPlatformPaths($SI)
            throw $_
        }
        #
        start-sleep 1
        #
        $mevents = get-event | 
            Where-Object{$_.sourceidentifier -match [regex]::Escape($SI)}
        #
        if ($mevents){
            #
            $this.clearevents($mevents[0].SourceIdentifier)
            return $true
        } else {
            $this.UnregisterEvent($SI)
        }
        #
        return $false
        #
    }
    #
    [void]clearevents($mevent){
        #
        $mevent = ($mevent -split ';')[0]
        get-event | 
            Where-Object {$_.SourceIdentifier -match [regex]::Escape($mevent)} |
            remove-event
        $mevents = get-event | 
            Where-Object{$_.sourceidentifier -match [regex]::Escape($mevent)}
        if ($mevents){
            $this.writeoutput("WARNING: events not removed correctly: $mevent")
        }
        #
    }
    <# -----------------------------------------
     WaitEvent
     wait for an event to trigger optionally
     remove the event subscriber and the event
     ------------------------------------------
     Input: 
        -SI: the source identifier
     ------------------------------------------
     Usage: $this.WaitEvent(SI)
    ----------------------------------------- #>
    [void]WaitEvent($SI){
        #
        Wait-Event -SourceIdentifier $SI
        Remove-Event -SourceIdentifier $SI
        #
    }
    <# -----------------------------------------
     UnregisterEvent
     wait for an event to trigger optionally
     remove the event subscriber and the event
     ------------------------------------------
     Input: 
        -SI: the source identifier
     ------------------------------------------
     Usage: $this.UnregisterEvent(SI)
    ----------------------------------------- #>
    [void]UnregisterEvent($SI){
        Unregister-Event -SourceIdentifier ($SI+ ';changed') -Force -EA SilentlyContinue
        Unregister-Event -SourceIdentifier ($SI+ ';renamed') -Force -EA SilentlyContinue
    }
}