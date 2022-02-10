    function sampledef {
        param(
            [parameter()][string]$mpath,
            [parameter()][string]$module,
            [parameter()][string]$slideid,
            [parameter()][string]$batchid,
            [parameter()][string]$project
        )
        #
        if (!($PSBoundParameters.ContainsKey('mpath'))){
            return [sampledef]::new()
        }
        #
        if (!($PSBoundParameters.ContainsKey('module'))){
            return [sampledef]::new()
        }
        #
        if (!($PSBoundParameters.ContainsKey('slideid')) -AND 
            !($PSBoundParameters.ContainsKey('batchid'))
        ){
            return [sampledef]::new($mpath, $module)
        }
        #
        if ($PSBoundParameters.ContainsKey('slideid')){
            return [sampledef]::new($mpath, $module, $slideid)
        }
        #
        if (($PSBoundParameters.ContainsKey('project')) -AND 
            ($PSBoundParameters.ContainsKey('batchid'))
            ){
            return [sampledef]::new($mpath, $module, $batchid, $project)
        }
        #
        Throw 'usage: logger [mpath [module [slideid] [project batchid]]]'
        #
    }