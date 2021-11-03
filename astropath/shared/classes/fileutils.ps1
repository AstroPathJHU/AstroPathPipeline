<# -------------------------------------------
 fileutils
 created by: Benjamin Green - JHU
 Last Edit: 11.02.2021
 --------------------------------------------
 Description
 methods used to read and write files with 
 error checking and using file locking mutexes
 -------------------------------------------#>
class fileutils : generalutils {
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
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
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
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw $cnt + ' attempts failed reading ' `
                + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
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
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
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
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
           Throw $cnt + ' attempts failed reading ' `
                + $fpath + '. Final message: ' + $err
        }
        #
        return $Q
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
    [void]PopFile([string] $fpath = '',[Object] $fstring){
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
    [void]SetFile([string] $fpath = '',[string] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Set')
        #
    }
    <# -----------------------------------------
     HandleReadFile
     Checks that the file exists and grabs the 
     mutex for a file. When it has the mutex,
     calls the write file method
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
        -fstring[string]: input array
        -opt[sting]: 'Set' or 'Pop'
     ------------------------------------------
     Usage: $this.HandleReadFile(fpath, fstring, opt)
    ----------------------------------------- #>
    [void]HandleWriteFile([string] $fpath = '',[string] $fstring, [string] $opt){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
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
        } while(($cnt -lt $Max) -and ($e -eq 1))
        #
        # if code cannot access the file 
        # after 10 minutes return an error indicator
        #
        if ($cnt -ge $Max){
            Throw $cnt + ' attempts failed writing ' +
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
     [void]WriteFile([string] $fpath = '',[string] $fstring, [string] $opt){
        if ($opt -eq 'Set'){
            Set-Content -Path $fpath -Value $fstring -NoNewline -EA Stop
        } elseif ($opt -eq 'Pop') {
            Add-Content -Path $fpath -Value $fstring -NoNewline -EA Stop
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
            while (-not $mxtx.WaitOne(1000)) {
                Start-Sleep -m 500;
            }
            return $mxtx
        } catch [System.Threading.AbandonedMutexException] {
            $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
            return $this.GrabMutex($mxtxid)
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
            $mxtx.ReleaseMutex()
            #
            # if another process crashes the mutex is never given up,
            # but is passed to the next grabbing process.
            # this attempts to close it again for the off chance there
            # is a duplicate grab
            #
            try { $mxtx.ReleaseMutex() } catch {} 
        } catch {
            Throw "mutex not released: " + $fpath
        }
    }
}