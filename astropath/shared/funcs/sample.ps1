    function sample {
        param(
            [parameter()][string]$mpath,
            [parameter()][string]$module,
            [parameter()][string]$slideid,
            [parameter()][string]$batchid,
            [parameter()][string]$project
        )
        #
        if (!($PSBoundParameters.ContainsKey('mpath'))){
            return [samplefiles]::new()
        }
        #
        if (!($PSBoundParameters.ContainsKey('module'))){
            return [samplefiles]::new()
        }
        #
        if (!($PSBoundParameters.ContainsKey('slideid')) -AND 
            !($PSBoundParameters.ContainsKey('batchid'))
        ){
            return [samplefiles]::new($mpath, $module)
        }
        #
        if ($PSBoundParameters.ContainsKey('slideid')){
            return [samplefiles]::new($mpath, $module, $slideid)
        }
        #
        if (($PSBoundParameters.ContainsKey('project')) -AND 
            ($PSBoundParameters.ContainsKey('batchid'))
            ){
            return [samplefiles]::new($mpath, $module, $batchid, $project)
        }
        #
        Throw 'usage: sample [mpath [module [slideid] [project batchid]]]'
        #
    }