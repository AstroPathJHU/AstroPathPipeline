class fileutils : generalutils {
    #
    [PSCustomObject]OpenCSVFile([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
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
           Throw $cnt + ' attempts failed reading ' + $fpath + '. Final message: ' + $err
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
     Usage: GetContent(fpath)
    ----------------------------------------- #>
    [Array]GetContent([string] $fpath){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        $Q = New-Object -TypeName psobject
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
            try{
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
           Throw $cnt + ' attempts failed reading ' + $fpath + '. Final message: ' + $err
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
     Usage: PopFile(fpath)
    ----------------------------------------- #>
    [void]PopFile([string] $fpath = '',[Object] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Pop')
        #
    }
    <# -----------------------------------------
     SetFile
     Overwrite a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
    ----------------------------------------- #>
    [void]SetFile([string] $fpath = '',[string] $fstring){
        #
        $this.HandleWriteFile($fpath, $fstring, 'Set')
        #
    }
    <# -----------------------------------------
     HandleReadFile
     write to a file with error checking
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
    ----------------------------------------- #>
    [void]HandleWriteFile([string] $fpath = '',[string] $fstring, [string] $opt){
        #
        $cnt = 0
        $e = 1
        $err = ''
        $Max = 120
        $mxtxid = 'Global\' + $fpath.replace('\', '_') + '.LOCK'
        #
        if (!(test-path $fpath)){
            New-Item -path $fpath -itemtype file -Force
        }
        #
        do{
           #
           $mxtx = $this.GrabMxtx($mxtxid)
             try{
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
            Throw $cnt + ' attempts failed writing ' + $fstring + ' to ' + $fpath + '. Final message: ' + $err
        }
        #
    }
    <# -----------------------------------------
     ReadFile
     append or overwrite a file
     ------------------------------------------
     Input: 
        -fpath[string]: file path to read in
     ------------------------------------------
     Usage: SetFile(fpath)
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
     https://stackoverflow.com/questions/7664490/interactively-using-mutexes-et-al-in-powershell
     ------------------------------------------
     Input: 
        -mxtxid: string object for a mutex
     ------------------------------------------
     Usage: GrabMxtx(mxtxid)
    ----------------------------------------- #>
    [System.Threading.Mutex]GrabMxtx([string] $mxtxid){
         try
            {
                $mxtx = New-Object System.Threading.Mutex -ArgumentList 'false', $mxtxid
                while (-not $mxtx.WaitOne(1000))
                {
                    Start-Sleep -m 500;
                }
                return $mxtx
            } 
            catch [System.Threading.AbandonedMutexException] 
            {
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
     ------------------------------------------
     Usage: ReleaseMxtx(mxtx)
    ----------------------------------------- #>
    [void]ReleaseMxtx([System.Threading.Mutex]$mxtx, [string] $fpath){
        try{
            $mxtx.ReleaseMutex()
            try { $mxtx.ReleaseMutex() } catch {} # if another process crashes the mutex is never given up.
        } catch {
            Throw "mutex not released: " + $fpath
        }
    }
}