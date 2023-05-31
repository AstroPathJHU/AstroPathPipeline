function UpdateProcessingLog {
    param(
        [Parameter(Mandatory=$true)][string]$logfile,
        [Parameter(Mandatory=$true)][string]$jobname,
        [Parameter(Mandatory=$true)]$sample,
        [Parameter()]$erroroutput,
        [Parameter()]$lineoutput
    )
    #
    $infomess = "INFO: "
    $errormess = "ERROR: "
    $mydate = $sample.getformatdate()
    $startline = ($jobname -replace '\.', ';') + ';'
    $endline = ";" + $mydate + "`r`n"
    #
    if ($PSBoundParameters.ContainsKey('erroroutput')){
        #
        $count = 1
        #
        if ($erroroutput -ne 0){
            $erroroutput | Foreach-object {
                $sample.popfile($logfile, (
                    $startline + $errormess + $count + $endline))
                $sample.popfile($logfile, (
                    $startline + $errormess + $_.Exception.Message + $endline))
                $s = $_.ScriptStackTrace.replace("at", "`t at")
                $sample.popfile($logfie, (
                    $startline + $errormess + $s + $endline))
                $count += 1
            }
        } else {
            $sample.popfile($logfile, (
                $startline + "INFO: Task completed successfully" + $endline)
            )
        }
    } 
    #
    if ($PSBoundParameters.ContainsKey('lineoutput')){
        $msg = $startline + ($infomess, $lineoutput -join " ") + $endline
        $sample.popfile($logfile, $msg)
    }
    #
}