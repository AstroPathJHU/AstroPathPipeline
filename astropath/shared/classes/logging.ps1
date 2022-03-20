﻿<# -------------------------------------------
 write_to_log
 write a message to the log in the specified formating
 projectID;cohortID;slideID;message;timestamp

 Usage: write_to_log -project -cohort -dpath -dname
                     -module -log_string [-log_type -slide_id
                     -message_type -console_output]
 -project: the project ID
 -cohort: the cohort ID
 -dpath: path to the clinical specimen project folder
 -dname: name of the clinical specimen project folder
 -module: the name of the module
 -log_string: string to be output in the message section of the log
 [-log_type]: "master" or "slide" level logging (default is master level)
 [-slide_id]: slide ID for the log must be specified for slide
               level logging (default uses module name)
 [-message_type]: string to be added to log message
                   (ex. "ERROR", "WARNING") (default
                   is "NORMAL" which does not add anything)
 [-console_output]: output log string to the console or not (default is not to)
 ------------------------------------------- #>
 [Flags()] Enum LogLevels {
    IMAGE = 1
    SLIDE = 2
    MAIN = 4
    CONSOLE = 8
    INFO = 16
    WARNING = 32
    ERROR = 64
    STARTED = 128
    FINISHED = 256
 }
#
class mylogger : samplereqs {
    [int]$level = 2
    [string]$message
    [string]$messageappend = ''
    [string]$vers
    [array]$val
    #
    # constructors
    #
    mylogger() : base(){}
    #
    mylogger($mpath, $module) : base($mpath, $module){}
    #
    mylogger($mpath, $module, $slideid) : base($mpath, $module, $slideid){
        $this.getlogger()
    }
    #
    mylogger($mpath, $module, $batchid, $project) : base($mpath, $module, $batchid, $project){
        $this.level = 4
        $this.getlogger()
    }
    #
    getlogger(){
        $this.vers = $this.GetVersion($this.mpath, $this.module, $this.project)
    }
    #
    # change level default 
    #
    [void]deflevels([int]$ilevel){
        $this.level = $ilevel
    }
    #
    [string]formatter(
        ){
           $mydate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
           $this.message = $this.message -replace ';', '-'
           $msg = @($this.Project, $this.Cohort, $this.slideid, `
                ($this.message+$this.messageappend), $mydate) -join ';'
           return  @($msg,"`r`n") -join ''
        }
    [string]formatternonewline(
    ){
        $mydate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $this.message = $this.message -replace ';', '-'
        $msg = @($this.Project, $this.Cohort, $this.slideid, `
            ($this.message+$this.messageappend), $mydate) -join ';'
        return  $msg
    }     
    #
    [void]info($msg){
        $this.info($msg, $this.level)
    }
    #
    [void]info($msg, $ilevel){
        $ilevel += [LogLevels]::INFO
        $this.parsemultiplelines($msg, $ilevel)
    }
    #
    [void]warning($msg){
         $ilevel = $this.level
        if (!(($this.level -band [LogLevels]::MAIN) -eq [LogLevels]::MAIN)){
            $ilevel += [LogLevels]::MAIN
        }
       $this.warning($msg, $this.level)
    }
    #
    [void]warning($msg, $ilevel){
       $ilevel += [LogLevels]::WARNING
       $this.parsemultiplelines($msg, $ilevel)
    }
    #
    [void]error($msg){
        $ilevel = $this.level
        if (!(($this.level -band [LogLevels]::MAIN) -eq [LogLevels]::MAIN)){
            $ilevel += [LogLevels]::MAIN
        }
       $this.error($msg, $ilevel)
    }
    #
    [void]error($msg, $ilevel){
       $ilevel += [LogLevels]::ERROR
       $this.parsemultiplelines($msg, $ilevel)
    }
    #
    [void]parsemultiplelines($msg, $ilevel){
        if (($ilevel -band [LogLevels]::INFO) -eq [LogLevels]::INFO){
            $tag = 'INFO'
        } elseif (($ilevel -band [LogLevels]::WARNING) -eq [LogLevels]::WARNING){
            $tag = 'WARNING'
        } elseif (($ilevel -band [LogLevels]::ERROR) -eq [LogLevels]::ERROR){
            $tag = 'ERROR'
        } else {
            $tag = ''
        }
        #
        $msg | ForEach-Object {
            $this.message = $tag + ": " + $_
            $this.Writelog($ilevel)
        }
    }
    #
    [void]start($msg){
        $this.buildappend()
        $this.message = "START: "+$msg+' '+$this.vers
        $this.defmsgcaps()
    }
    #
    [void]finish($msg){
        $this.message =  "FINISH: "+$msg+' '+$this.vers
        $this.defmsgcaps()
    }
    #
    [void]defmsgcaps(){
        $ilevel = $this.level
        if (!(($this.level -band [LogLevels]::MAIN) -eq [LogLevels]::MAIN)){
            $ilevel += [LogLevels]::MAIN
        }
        $this.Writelog($ilevel)
    }
    #
    [void]writelog(){
        $this.Writelog($this.level)
    }
    #
    [void]writelog($ilevel){
        if (($ilevel -band [LogLevels]::SLIDE) -eq [LogLevels]::SLIDE){
            $this.PopFile($this.slidelog, $this.formatter())
        }
        #
        if (($ilevel -band [LogLevels]::MAIN) -eq [LogLevels]::MAIN){
            $this.PopFile( $this.mainlog, $this.formatter())
        }
        #
        if (($ilevel -band [LogLevels]::CONSOLE) -eq [LogLevels]::CONSOLE){
            Write-Host $this.formatternonewline()
        }
    }
    #
    [void]buildappend(){
        if ($this.module -eq 'vminform'){
            $this.messageappend = ": Antibody: " + $this.val[2] + 
                " - Algorithm: " + $this.val[3] + 
                " - inForm version: " + $this.val[4]
        }
    }
}
