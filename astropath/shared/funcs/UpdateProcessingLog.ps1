function UpdateProcessingLog {
    param(
        [Parameter()][string]$logfile,
        [Parameter()]$sample,
        [Parameter()]$erroroutput,
        [Parameter()]$lineoutput
    )
    #
    $infomess = "INFO: "
    $errormess = "ERROR: "
    $mydate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $endline = ";" + $mydate + "`r`n"
    #
    if ($PSBoundParameters.ContainsKey('erroroutput')){
        #
        $count = 1
        #
        if ($erroroutput -ne 0){
            $erroroutput | Foreach-object {
                $sample.popfile($logfile, ($errormess + $count + $endline))
                $sample.popfile($logfile, ($errormess + $_.Exception.Message + $endline))
                $s = $_.ScriptStackTrace.replace("at", "`t at")
                $output.popfile($logfie, ($errormess + $s + $endline))
                $count += 1
            }
        } else {
            $output.popfile($logfile, "INFO: Task completed successfully `r`n")
        }
    } 
    #
    if ($PSBoundParameters.ContainsKey('lineoutput')){
        $msg = $infomess, $lineoutput, $endline -join " "
        $sample.popfile($logfile, $msg)
    }
    #
}