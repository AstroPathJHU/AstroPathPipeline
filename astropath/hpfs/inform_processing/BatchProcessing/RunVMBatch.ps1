<#
----------------------------------------------------------------------------------------------
 RunVMBatch
 Created by: Benjamin Green
----------------------------------------------------------------------------------------------
Description: 
 This powershell script checks the VM_inForm_queue on a VM and processes the next specimen
 The script will also update the VM_inForm_queue.csv file located in \\Program Files\BatchProcessing
 The VM_inForm_queue.csv should have 5 columns:
 Paths,Specimen,Antibody,Algorithm,Status; for this program to run a batch it must have the
 first four columns of this csv file filled in properly 
 ex: "\\bki04\e$\Clinical_Specimen,M1_1,CD8,CD8_outlier.ifp,"
----------------------------------------------------------------------------------------------
#>

Function Run-VMBatch {
    #
    param($vers, $p1)
    #
    $CI, $CI1 = StartUp-Checks -vers $vers -p1 $p1
    if ($CI -eq 1){
        return
        }
    #
    # change the line of text to add 'Processing Started' tag in the VM inForm queue
    #
    $CI2 = Write-Queue -loc "Started" -p1 $p1 -in $CI -in1 $CI1
    #
    # check the path of the images and the algorithm or project is valid if it is run 
    # InForm
    #
    $pp = $CI -split ','
    $image_path = $pp[0]+'\'+$pp[1]+'\im3\flatw'
    $tQ = test-path $image_path
    $patha = $pp[0]+'\tmp_inform_data\Project_Development\' + $pp[3]
    $aQ = test-path $patha
    # 
    if (!$tQ) {
        Write-Host "  WARNING: flatw path not found for:" $pp -ForegroundColor Magenta
    } elseif (!$aQ){
        Write-Host "  WARNING: algorithm not found for:" $pp -ForegroundColor Magenta
    } else{
        Run-InFormAuto -pp $pp -vers $vers
    }
    #
    # change the line of text to add 'Processing Finished' tag in the VM inForm queue
    #
    $CI3 = Write-Queue -loc "Finished" -p1 $p1 -in $CI2 -in1 $CI1
    #
}

Function StartUp-Checks {
    #
    param($vers, $p1)
    #
    # start up messages
    #
    Write-Host "." -ForegroundColor Yellow
    Write-Host "Starting batch processing for VM ..." -ForegroundColor Yellow
    #
    # which version of inForm to use
    #
    Write-Host "  InForm version:" $vers -ForegroundColor Yellow
    #
    # define path to inForm_queue and check if it exists
    #
    $tQ = test-path $p1
    #
    if ($tQ){
        #
        Write-Host "  VM InForm queue found:" $p1 -ForegroundColor Yellow
        #
        $Q = get-content -Path $p1
        }else{
        #
        Write-Host "  WARNING: VM InForm queue NOT found:" $p1 -ForegroundColor Magenta
        Start-Sleep -s (10 * 60) 
        return 1
        #
        }
    #
    # get the next line of inForm's to run (inForm's that do not have 'Processing Finished' tag
    #
    $Task1 = "Processing Finished"
    $CI1 = $Q | select-string -pattern $Task1 -notmatch | Select-Object -first 1 -skip 1
    $CI = $CI1 -replace ',[^,]*$', ","
    $CI = $CI -replace(' ','')
    #
    # if this string is empty then there are no Batches to be processed
    # rerun the script in 10 mins to check if any new inForms need to be processed
    #
    if (!($CI)){
        #
        Write-Host "No samples to process. Checking again in 10 minutes" -ForegroundColor Yellow
        Start-Sleep -s (10 * 60) 
        return 1
    } else {
        return $CI, $CI1
    }
    #
}

Function Write-Queue {
    #
    param($loc, $p1, $in, $in1)
    #
    # $loc: location at processing, finished or started
    # $p1: path to file
    # $in: string to replace
    #
    $D = Get-Date
    if ($loc -eq "Finished"){
        $str = "Processing Finished on: " + $D
        $out = "$in" + "," + $str
        #
        # add escape to '\'
        #
        $rg = [regex]::escape($in) + "$"
    } else {
        $str = "Processing Started on: " + $D
        $out = "$in" + $str
        #
        # add escape to '\'
        #
        $rg = [regex]::escape($in1) + "$"
    }
    #
    Write-Host " " $str -ForegroundColor Yellow
    #
    # write line to inForm_queue
    #
    $cnt = 0
    $Max = 120
    #
    do{
        try{
            $Q = get-content -Path $p1
            $Q3 = $Q -replace $rg,$out
            Set-Content -Path $p1 -Value $Q3
            break
        }catch{
            $cnt = $cnt + 1
            Start-Sleep -s 3
            Continue
        }
    } while($cnt -lt $Max)
    #
    # if the script could not access the queue file after 10 mins of trying every 2 secs
    # there is an issue and exit the script
    #
    if ($cnt -ge $Max){
        $ErrorMessage = "Could not access inForm_queue.csv"
        Throw $ErrorMessage 
    }
    return $out
}

Function Run-InFormAuto {
    #
    param($pp, $vers)
    #
    # parse input
    #
    $path = $pp[0]
    $sname = $pp[1]
    $ABx = $pp[2]
    $patha = $pp[3]
    $image_list = "a"
    #
    # start the inForm Processing
    #
    $c = "C:\Program Files\BatchProcessing\RunFullBatch.exe"
    $icount = 1
    #
    do {
        #
        if ($icount -eq 1){
            #
            Write-Host "`tStarting InForm for" $pp -ForegroundColor Yellow
            #
        } elseif ($icount -gt 1 -and $icount -lt 6){
            # 
            Write-Host "  WARNING: RunFullBatch returned an Exit Code:" `
                $a.ExitCode -ForegroundColor Magenta
            Write-Host "`tAttempting to Restart InForm for" $pp -ForegroundColor Magenta
            #
        } elseif ($icount -eq 6){
            #
            $ErrorMessage = "  ERROR: RunFullBatch returned Exit Code: " + `
                $a.ExitCode + "`r`n`t" + $pp + "FAILED"
            Write-Host $ErrorMessage -ForegroundColor Red
            Write-Host "ENTER to Exit" -ForegroundColor Red -NoNewline
            Read-Host
            exit
            # 
        }
        #
        #$a = "0" # for editing\testing purposes
        #$a = $a | Add-Member -NotePropertyMembers @{ExitCode=0} -PassThru # for editing\testing purposes
        #
        $a = Start-Process -FilePath $c -ArgumentList `
            $vers,$path,$sname,$ABx,$patha,$image_list -WAIT -NoNewWindow -PassThru
        $icount += 1
        # 
    } while (!($a.ExitCode -eq 0))
    #
 }

while(1){
    #
    $vers = "2.4.8"
    $p1 = "C:\Program Files\BatchProcessing\VM_inForm_queue.csv"
    #
    Run-VMBatch -vers $vers -p1 $p1
    #
}