<#
----------------------------------------------------------------------------------------------
 ProcessInFormFiles
 Created by: Benjamin Green
----------------------------------------------------------------------------------------------
 Description: 
  This powershell script checks the VM_inForm_queue on each VM of BKI05 not in $TLWS
  if all values in the queue are processing it will add another value for that queue
  The script will also update the inForm_queue.csv file located in the main directory.
  The inForm_queue.csv in the main directory should have 5 columns:
  Paths,Specimen,Antibody,Algorithm,Status; for this program to run a batch it must have the
  first four columns of this csv file filled in properly 
  ex: "\\bki04\e$\Clinical_Specimen,M1_1,CD8,CD8_outlier.ifp,"
 
  $main = "\\bki05\c$\Processing_Specimens"
----------------------------------------------------------------------------------------------
#>
param([string] $main)
#
$cred = Get-Credential
$UserName = $cred.UserName
$Password = $cred.GetNetworkCredential().Password
$vers = '2.4.8'
#
Function inform_queue {
    while(1){
        #
        # initialize VM list
        #
        $VMs = Initialize-VMlist
        #
        # Get a new task
        #
        $CI = Extract-Tasks -main $main
        if (!($CI)){
            Write-Host "No samples to process. Checking again in 10 minutes" -ForegroundColor Yellow
            Start-Sleep -s (10*60)
            continue
        }
        #
        # get running jobs 
        #
        $running = @(Get-Job | Where-Object { $_.State -eq 'Running' })
        #
        # put virtual machines already running jobs into a new separate list
        #

        #
        # if there are virtual machines without jobs launch tasks from the queue
        #
        $VMs = Launch-VMTask -VMs $VMs -CI $CI -vers $vers -UserName $UserName -Password $Password
        #
        # if there are still virtual machines without jobs restart the processing otherwise wait for any job to complete then restart
        #

    }
}
Function Get-MyVMJobs {
    #
    $rVM = @(Get-Job | Where-Object { $_.State -eq 'Running' }).Name
    #
    FOREACH ($WS in $rVM){
       $CC = $WS + "$"
       $VMs = $VMs | Select-String $CC -notmatch
    }
    ## should check how long each job has been running and cancel it if it has lasted longer than 48 hours 
    #
    ret

}
#
Function Launch-VMTask {
    param(
        $VMs,
        $CI,
        $vers,
        $UserName,
        $Password
    )
    #
    $sb = {
        param(
            $UserName, 
            $Password,
            $iVM,
            $code,
            $in,
            $vers
        )

        psexec -i -u $UserName -p $Password \\$iVM cmd /c `
            "powershell -noprofile -executionpolicy bypass -command $code -in $in -vers $vers"
    }
    #
    $code = $coderoot+'\inform_worker.ps1'
    $CI1 = $CI -replace ",,",","
    $CI1 = $CI1 -replace ", ,",","
    $CI1 = $CI1 -replace ",  ,",","
    #
    While($VMs -and $CI){
        #
        # select the next usable VM
        #
        $cVM, $VMs = $VMs 
        $iVM = $cVM -replace '_',''
        $iVM = $iVM.ToLower()
        #
        # select the current task
        #
        $CI_check, $CI= $CI
        $in, $CI1 = $CI1
        $in = "'"+$in+"'"
        #
        # launch the task
        #
        start-job -ScriptBlock $sb -ArgumentList $UserName,$Password,$iVM,$code,$in,$vers -Name $cVM
        #
        # Update the queue line
        #
        Update-Queue -CI $CI -cVM $cVM -CI_check $CI_check -p1 $p1
        #
    }
    #
    return $VMs
}
#
Function Initialize-VMlist{
    #
    # get list of Vitual machines that are on
    #
    $VMs = (Get-VM | where {$_.State -eq 'RUNNING'}).Name 
    #
    # remove any of the "Taube Lab Workstations" from usable VMs (VMs 2)
    #
    [System.Collections.ArrayList]$TLWS = "VM_inForm_21","VM_inForm_22"
    FOREACH ($WS in $TLWS){
        $CC = $WS + "$"
       $VMs = $VMs | Select-String $CC -notmatch
    }
    #
    Write-Host "." -ForegroundColor Yellow
    Write-Host "Starting Inform-Task-Distribution" -ForegroundColor Yellow
    write-host " Current Computers for Processing:" -ForegroundColor Yellow
    write-host " " $VMs -ForegroundColor Yellow
    Write-Host "  ." -ForegroundColor Yellow
    #
    return $VMs
    #
}
#
Function Extract-Tasks {
    #
    param($main)
    #
    # get the next inForm image set in the queue that needs to be processed
    #
    $p1 = $main + "\inForm_queue.csv"
    $Q = get-content -Path $p1 
    $CI = @()
    foreach($row in $Q) {
        $array = $row.ToString().Split(",")
        $array = $array -replace '\s',''
        if($array[3]){
            if($array -match "Processing"){ Continue } else { 
                $CI += $row
                }
            } 
    }
    #
    return $CI
    #
}
#
Function Update-Queue {
        param(
            $CI,#current input
            $cVM,#current vm name
            $CI_check,
            $p1 #full path to the queue
        )
        #
        $D = Get-Date
        $CI2 = "$CI" + "Processing sent to: " + $cVM + " on: " + $D
        #
        # add escape to '\'
        #
        $rg = [regex]::escape($CI_check) + "$"
        #
        $cnt = 0
        $Max = 120
        #
        do{
            try{
                $Q = get-content -Path $p1
                $Q2 = $Q -replace $rg,$CI2
                Set-Content -Path $p1 -Value $Q2
                break
            }catch{
                $cnt = $cnt + 1
                Start-Sleep -s 5
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
        #
}