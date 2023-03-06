function LaunchModule{
    [CmdletBinding(DefaultParameterSetName = 'slide')]
    param(
        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [string]$mpath='\\bki04\astropath_processing',

        [Parameter(Mandatory=$true, ParameterSetName = 'slide')]
        [Parameter(Mandatory=$true, ParameterSetName = 'batch')]
        [Parameter(Mandatory=$true, ParameterSetName = 'inform')]
        [string]$module = '',

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(Mandatory=$true, ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [string]$project='',

        [Parameter(Mandatory=$true, ParameterSetName = 'slide')]
        [Parameter(Mandatory=$true, ParameterSetName = 'inform')]
        [string]$slideid='',

        [Parameter(Mandatory=$true, ParameterSetName = 'batch')]
        [string]$batchid='',

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [string]$processloc='',

        [Parameter(ParameterSetName = 'inform')]
        [string]$taskid='',

        [Parameter(Mandatory=$true, ParameterSetName = 'inform')]
        [string]$antibody='',

        [Parameter(Mandatory=$true, ParameterSetName = 'inform')]
        [string]$algorithm='',

        [Parameter(Mandatory=$true, ParameterSetName = 'inform')]
        [string]$informvers='',

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [string]$tasklogfile='',

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [string]$jobname='',

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [switch]$test,

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [switch]$nolog,

        [Parameter(ParameterSetName = 'slide')]
        [Parameter(ParameterSetName = 'batch')]
        [Parameter(ParameterSetName = 'inform')]
        [switch]$interactive,

        [Parameter(ValueFromRemainingArguments)]$overloadargs

    )
    #
    switch ($true) {
        ($PSBoundParameters.test -and $PSBoundParameters.interactive) {
            throw 'Cannot use both test and interactive switch at the same time'
        }
        $PSBoundParameters.test {
            $inp = initmodule -task $PSBoundParameters -module $module -test
            return $inp
        }
        $PSBoundParameters.interactive {
            $inp = initmodule -task $PSBoundParameters -module $module -interactive
            return $inp
        }
    }
    #
    if ($module -match 'batch'){
        $m = [launchmodule]::new($mpath, $module,
            $batchid, $project, $PSBoundParameters)
        #
    } else {
        $m = [launchmodule]::new($mpath, $module,
            $slideid, $PSBoundParameters)
    }        
    #
    return $m
    
<#
.SYNOPSIS

run a specified astropath [module] class with 
astropath log formatting

.DESCRIPTION
 
Function to run a module. creates a launchmodule
class for the specified module. The module class 
is called by the module parameter. By default will
run the 'run[module]' method of the module class. 
Use this function with the -test flag to return the module 
object for testing.

.PARAMETER mpath
main path to the astropath processing directory

.PARAMETER module
astropath module name to run

.PARAMETER project
astropath project number 

.PARAMETER slideid
astropath slideid

.PARAMETER batchid
astropath batchid

.PARAMETER processloc
The processing location of the task. This usually
indicates a unc path to a drive or folder for the 
task to use to save temporary data

.PARAMETER taskid
optional argument indicating the inform queue taskid

.PARAMETER antibody
The antibody that will be run through inForm which
will be indicated by the output folders

.PARAMETER algorithm
The inForm algorithm or project to be run by the task

.PARAMETER tasklogfile
An optional arguement for the task log file. The 
task will print the processid and any error information 
to this log and is used to provide easy reporting 
back to the main running function

.PARAMETER jobname
The jobname for the task. Should be a server.location.module
combination which will be prepended to the log messages.

.PARAMETER test 
The optional test switch. If supplied rather than running 
the module the code will return the module object. module
methods can then be run using $varname.sample.[method]

.PARAMETER nolog
Optional argument not to write to the logs

.Example

launchmodule -mpath $mpath -module $module -slideid $slideid

.Example 

launchmodule -mpath [mpath] -module [module] -batchid [batchid] -project [project]
#>
}
#