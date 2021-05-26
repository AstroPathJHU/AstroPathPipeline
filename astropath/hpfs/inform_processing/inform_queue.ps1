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
while(1){
#
# get list of VM Names that are off
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
# While a VM is part of the code run 
#
While($VMs){
    #
    # select the next usable VM
    #
    $cVM = $VMs | Select-Object -first 1
    $CC = "$cVM" + "$"
    $VMs = $VMs | Select-String $CC -notmatch
    #
    Write-Host "  Checking:" $cVM -ForegroundColor Yellow
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
                $CI = $row
                break }} 
    }

    if (!($CI)){continue}
    #
    # remove any possible false columns
    #
    $CI_check = $CI
    $CI = $CI -replace ",,",","
    $CI = $CI -replace ", ,",","
    $CI = $CI -replace ",  ,",","
    #
    # get VM credentials
    # 
    $CredU = $cVM -replace "VM_i",".\I"
    $CredPW = ConvertTo-SecureString -String "password" -AsPlainText -Force
    $Cred = New-Object System.Management.Automation.PSCredential -ArgumentList $CredU, $CredPW
    #
    # Enter PSSession 
    #
    $a = invoke-command -VMName $cVM -Credential $Cred -argumentlist $cVM, $CI -scriptblock {
        param ([string] $cVM, [string] $CI)
        #
        # check csv VM_inForm_queue
        #
        $vmn = $cVM -replace "VM_i","I"
        $p3 = Get-ChildItem "C:\Program Files" | Where-Object {$_.PSIsContainer -eq $true -and $_.Name -match "AstroPathPipeline*"}
        $p3 = "C:\Program Files\" + $p3+ "\astropath\hpfs\inform_processing\BatchProcessing"
        $p2 = "$p3\VM_inForm_queue.csv"
        #
        # if file exists get content
        #
        $Check = Test-Path $p2
        #
        if($Check){
            $Q2 = get-content -Path $p2
        } else {
            #  or create file
            $check_path = Test-Path $p3
            if ($check_path){
                $Q2 = ("Path,Specimen,Antibody,Algorithm,Start,Finish","$CI")
                Set-Content -Path "$p2" -Value $Q2
                return $a = 1
            } else {
                $a = 0
                $ErrorMessage = "Worker not initialized properly"
                Throw $ErrorMessage
                #Write-Host $ErrorMessage -ForegroundColor Red
            }
        }
        #
        # if all files have "Processing Specimen" filled in then add a new file to the queue
        #
        #
        $Task2 = "Processing"
        $CI2 = $Q2 | select-string -pattern $Task2 -notmatch | Select-Object -first 1 -skip 1
        #
        if($CI2){
            $a = 0
        }else{
            #
            $cnt = 0
            $Max = 120
            #
            do{
                try{
                    $Q2 = get-content -Path $p2
                    $Q2 = ($Q2,"$CI")
                    Set-Content -Path $p2 -Value $Q2
                    break
                }catch{
                    $cnt = $cnt + 1
                    Start-Sleep -s 4
                    Continue
                }
            } while($cnt -lt $Max)
            #
            # if the script could not access the queue file after 10 mins of trying every 2 secs
            # there is an issue and exit the script
            #
            if ($cnt -ge $Max){
                $ErrorMessage = "Could not access VM_inForm_queue.csv"
                Throw $ErrorMessage 
            }
            #
            $a = 1
            #
        }
        Return $a
      }
    #
    if ($a -eq 1){
        #
        # change the line of text to add processing to it in the inForm queue
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
    }

}
#
Write-Host "  ." -ForegroundColor Yellow
Write-Host "  All Computers Checked. Waiting 10 minutes..." -ForegroundColor Yellow
Write-Host "." -ForegroundColor Yellow
#
# rerun the script in 10 mins to check if any new inForms need to be processed
#
Start-Sleep -s (10 * 60)
# 
}
