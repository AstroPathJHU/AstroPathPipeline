function UpdateProcessingLog {
    param(
        [Parameter()][string]$logfile,
        [Parameter()]$sample,
        [Parameter()]$erroroutput,
        [Parameter()]$lineoutput
    )
    #
    if ($PSBoundParameters.ContainsKey('erroroutput')){
        #
        $count = 1
        #
        if ($erroroutput -ne 0){
            $erroroutput | Foreach-object {
                $sample.popfile($logfile, ("ERROR: " + $count + "`r`n"))
                $sample.popfile($logfile,("  " + $_.Exception.Message  + "`r`n"))
                $s = $_.ScriptStackTrace.replace("at", "`t at")
                $output.popfile($logfie, ($s + "`r`n"))
                $count += 1
            }
        } else {
            $output.popfile($logfile, "INFO: Task completed successfully `r`n")
        }
    } 
    #
    if ($PSBoundParameters.ContainsKey('lineoutput')){
        $sample.popfile($logfile, ("INFO: " + $lineoutput + "`r`n"))
    }
    #
}