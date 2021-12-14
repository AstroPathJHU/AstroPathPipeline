﻿#
# Module manifest for module 'astropath.psd1'
#
# Generated by: Benjamin Green
#
# Generated on: 10/22/2020
#

@{

# Script module or binary module file associated with this manifest.
RootModule = 'astropath.psm1'

# Version number of this module.
ModuleVersion = '0.0.2'

# Supported PSEditions
# CompatiblePSEditions = @()

# ID used to uniquely identify this module
GUID = '7d4d3c4d-8f89-4c5c-9101-4da34459c398'

# Author of this module
Author = 'Benjamin Green'

# Company or vendor of this module
CompanyName = 'AstroPath-Johns Hopkins'

# Copyright statement for this module
Copyright = 'Copyright 2021 AstroPath    
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.'

# Description of the functionality provided by this module
Description = 'Used to launch and maintain the HPFs and Scans modules of the AstroPath Pipeline.'

# Minimum version of the Windows PowerShell engine required by this module
PowerShellVersion = '3.0'

# Name of the Windows PowerShell host required by this module
# PowerShellHostName = 'Windows PowerShell ISE Host'

# Modules that must be imported into the global environment prior to importing this module
# RequiredModules = @()

# Assemblies that must be loaded prior to importing this module
# RequiredAssemblies = @()

# Script files (.ps1) that are run in the caller's environment prior to importing this module.
 ScriptsToProcess = @('shared\classes\copyutils.ps1',
                      'shared\classes\generalutils.ps1',
                      'shared\classes\fileutils.ps1',
                      'shared\classes\aptabletools.ps1',
                      'shared\classes\sharedtools.ps1',
		              'shared\classes\sampledef.ps1',
                      'shared\classes\logging.ps1',
		              'shared\classes\launchmodule.ps1',
		              'shared\classes\moduletools.ps1',
                      'shared\classes\queue.ps1',
                      'shared\classes\dispatcher.ps1',
                      'hpfs\vminform\classes\vminform.ps1',
                      'hpfs\imagecorrection\classes\imagecorrection.ps1',
                      'hpfs\flatfield\classes\meanimage.ps1',
                      'hpfs\flatfield\classes\batchmeanimagecomparison.ps1',
                      'hpfs\flatfield\classes\batchflatfield.ps1',
                      'hpfs\segmaps\classes\segmaps.ps1',
                      'hpfs\shredxml\classes\shredxml.ps1',
					  'hpfs\merge\classes\merge.ps1',
		      'hpfs\warping\classes\warpoctets.ps1'
                      )

# Modules to import as nested modules of the module specified in RootModule/ModuleToProcess
# NestedModules = @()

# Functions to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no functions to export.
FunctionsToExport = '*'

# Cmdlets to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no cmdlets to export.
CmdletsToExport = '*'

# Variables to export from this module
VariablesToExport = '*'

# Aliases to export from this module, for best performance, do not use wildcards and do not delete the entry, use an empty array if there are no aliases to export.
AliasesToExport = '*'

# DSC resources to export from this module
# DscResourcesToExport = @()

# List of all modules packaged with this module
# ModuleList = @()

# List of all files packaged with this module
# FileList = @()

# Private data to pass to the module specified in RootModule/ModuleToProcess. This may also contain a PSData hashtable with additional module metadata used by PowerShell.
PrivateData = @{

    PSData = @{

        # Tags applied to this module. These help with module discovery in online galleries.
        # Tags = @()

        # A URL to the license for this module.
        LicenseUri = 'https://github.com/AstroPathJHU/AstroPathPipeline/LICENSE'

        # A URL to the main website for this project.
        ProjectUri = 'https://github.com/AstroPathJHU/AstroPathPipeline'

        # A URL to an icon representing this module.
        IconUri = 'https://github.com/AstroPathJHU/AstroPathPipeline/astropath/lib'

        # ReleaseNotes of this module
        ReleaseNotes = 'Still in beta'

    } # End of PSData hashtable

} # End of PrivateData hashtable

# HelpInfo URI of this module
HelpInfoURI = 'https://github.com/AstroPathJHU/AstroPathPipeline'

# Default prefix for commands exported from this module. Override the default prefix using Import-Module -Prefix.
# DefaultCommandPrefix = 'astropath'

}

