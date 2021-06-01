#cs ----------------------------------------------------------------------------

 AutoIt Version: 3.3.14.5
 Author:         Benjamin Green

 Script Function:
	Run a Batch of im3 images through inForm GUI using the algorithm given. Input should be 2 strings
	input 1: Clinical_Specimen_path, specimen,antibody,algorithm
	ex> '\\bki04\e$\Clinical_Specimen,M1_1,CD8,CD8_outlier.ifp,'
	input 2: the version number for inForm ex>'2.4.3'

#ce ----------------------------------------------------------------------------

#include <MsgBoxConstants.au3> ;required for msgbox constants
#include <WinAPI.au3>
#include <Constants.au3>
#include <Array.au3> ; required for _ArrayDisplay and for read Batch to array
#include <File.au3> ; requirerd to read Batch log
#include <String.au3> ; required for string between
#include <GUITreeView.au3>
#cs
Local $path = '\\bki04\Clinical_Specimen'
Local $sname = 'M1_1'
Local $ABx = 'CD8'
Local $patha = 'CD8_outlier.ifp'
Local $image_list = 'a'
Local $vers = '2.4.8'
#ce
Local $vers = $CmdLine[1]
Local $path = $CmdLine[2]
Local $sname = $CmdLine[3]
Local $ABx = $CmdLine[4]
Local $patha = $CmdLine[5]
Local $image_list = $CmdLine[6]
;
RunFullBatch($vers, $path, $sname, $ABx, $patha, $image_list)
;
Func RunFullBatch($vers, $path, $sname, $ABx, $patha, $image_list)
		;
		; cut input
		;
		Global $icount = 1
		Local $pathim
		;
		sleep(1 * 1000)
		_ParseInput($path, $sname, $ABx, $patha, $pathim)
		sleep(1 * 1000)
		;
		; turn on inForm
		;
		_StartinForm($vers)
		;
		; switch to Batch tab
		;
		Local $iFtit
		_GetBatchTab($iFtit, $vers)
		;
		; add the algorithm
		;
		Local $iFhnd
		_GetAlgor($iFtit, $iFhnd, $patha)
		;
		; create export directory
		;
		Local $pathf = 'Z:\tmp_inform_data\CD8\InForm_2_2'
		_CreateExportDir($path, $ABx, $pathf)
		;
		; export directory to inform
		;
		_GetExportdir($iFhnd, $pathf, $path)
		;
		; add all images
		;
		If ($image_list = "a") Then
			_AddAllImages($iFhnd, $pathim)
		Else
			_AddSomeImages($iFhnd, $pathim, $image_list)
		EndIf
		;
		; now run inForm Batch
		;
		_RuninForm($iFhnd)
		;
		; get possible fixable errors images
		;
		Local $tmpar = ''
		_GetErrors($pathf, $tmpar, $sname)
		;
		; run an inForm session till all errors are gone
		;
		_RemoveErrs($iFhnd, $pathim, $tmpar, $sname, $path, $ABx)
		;
		WinClose($iFhnd)
		;
		EXIT
EndFunc

Func _ParseInput(ByRef $path, ByRef $sname, ByRef $ABx, ByRef $patha, ByRef $pathim)
	;
	; parse the input string and map the drive
	;
	Local $iFile = False, $iFile2 = False
	While ($iFile = False) OR ($iFile2 = False)
		;
		; exit if files are not found after 5 tries
		;
		IF $icount = 5 Then
			IF ($iFile = False) Then
				Exit(1)
			ElseIf ($iFile2 = False) Then
				Exit(2)
			EndIf
		EndIf
		;
		; map the root directory
		;
		Local $ss = DriveMapGet("Z:")
		If ($ss <> 'INVALID') Then
			DriveMapDel("Z:")
		EndIf
		DriveMapAdd("Z:", $path, $DMA_PERSISTENT, "win\username","password")
		;
		; check if the specimen\ flatw folder exists
		;
		$pathim = "Z:\" & $sname & "\im3\flatw"
		$iFile = FileExists($pathim)
		;
		; check that the algorithm exists, if the input path
		;
		IF NOT FileExists($patha) Then
			Local $pathb = "Z:\tmp_inform_data\Project_Development\" & $patha
		Else
			Local $pathb = $patha
		EndIf
		$iFile2 = FileExists($pathb)
		;
		$icount += 1
		;
	WEnd
	;
	$patha = $pathb
	;
	$icount = 1
	;
EndFunc

Func _StartinForm(ByRef $vers)
	;
	; turn on inForm
	;
	Local $inform = 'C:\Program Files\Akoya\inForm\' & $vers & '\inForm.exe'
	Run($inform)
	;
	; wait for the window to respond. If it doesn't try again.
	;
	Local $iFtit_a = ""
	While ($iFtit_a = "")
		;
		IF ($icount = 4) Then
			EXIT(3)
		EndIf
		;
		Local  $w_time = 10
		_Check_tmp_dialog($w_time)
		WinWait("inForm", "", $w_time)
		$iFtit_a = WinGetTitle("inForm")
		While ($iFtit_a = "inForm Not Launched")
			WinClose($iFtit_a)
			$iFtit_a = WinGetTitle("inForm")
		WEnd
		;
		$icount += 1
		;
	WEnd
	;
	$vers = StringSplit($vers, ".")
	$vers = "inForm " & $vers[1] & "." & $vers[2] & " - Batch  Analysis"
	$icount = 1
	;
EndFunc

Func _Check_tmp_dialog($tt)
	;
	; if it is a tmp license then close the tmp_license window
	;
	WinWait("License Information","",$tt)
	Local $tlcs = WinGetTitle("License Information")
	If Not $tlcs = "" Then
		WinClose($tlcs)
	EndIf
	;
EndFunc

Func _GetBatchTab(ByRef $iFtit, $vers)
	;
	; get the handle and title of the main inform window and activate it
	;
	WinWait("inForm", "", 10)
	$iFtit = WinGetTitle("inForm")
	Local $iFhnd = WinGetHandle($iFtit)
	WinActivate($iFhnd)
	WinWaitActive($iFhnd)
	;
	; get to the Batch Analysis tab (has two spaces between Batch & Analysis
	;
	Local $pos = ControlGetPos($iFhnd, "", "Manual  Analysis")
	Local $X = $pos[0]
	Local $pos2 = ControlGetPos($iFhnd, "", "Image Format:")
	Local $Y = $pos2[1] + 150
	Opt("MouseCoordMode", 0)
	MouseClick("left",$X, $Y)
	sleep(1 * 1000)
	;
	; handle and name of page changes when switching tabs
	;
	$iFtit = WinGetTitle("inForm")
	;
	; error check which attempts to select batch tab up to 5 times if the
	; correct handle is not returned
	;
	IF NOT ($iFtit = $vers) Then
		IF $icount = 5 Then
			Exit (4)
		ENDIF
		$icount += 1
		_GetBatchTab($iFtit, $iFhnd)
	ENDIF
	;
	$icount = 1
	;
EndFunc

Func _GetAlgor($iFtit, ByRef $iFhnd, $patha)
	;
	; get the algorithm
	;
	;
	$iFhnd = WinGetHandle($iFtit)
	Local $Algtit = ""
	;
	While ($Algtit = "")
		;
		; after the fifth try exit
		;
		if ($icount = 5) Then
			Exit (5)
		EndIf
		WinActivate($iFhnd)
		WinWaitActive($iFhnd)
		sleep(2 * 1000)
		;
		; Open the Algorithm 'Browse...' file explorer. I use the
		; handle for Export Directory 'Browse...'; to select the
		; Algorithm 'Browse...'. A work around b/c I could not
		; find Algorithm 'Browse'... Handle
		;
		Local $expbut = ControlGetHandle($iFhnd, "", "Browse...")
		Local $pos = ControlGetPos($iFhnd, "", $expbut)
		Opt("MouseCoordMode", 0)
		MouseClick("left",$pos[0] + $pos[3] + 45, $pos[1] - 50)
		;
		; allow the window 5 secs to initialize then activate it,
		; if it fails to load try to select the window again
		;
		sleep(2 * 1000)
		WinWait("Open","",10)
		Local $Algtit = WinGetTitle("Open")
		$icount += 1
	WEnd
	;
	$icount = 1
	Local $Alghnd = WinGetHandle($Algtit)
	WinActivate($Alghnd)
	WinWaitActive($Alghnd)
	;
	; click into the file location and enter string
	;
	Opt("MouseCoordMode", 0)
	$pos = ControlGetPos($Alghnd, "", "&Open")
	Opt("MouseCoordMode", 0)
	MouseClick("left", $pos[0] - 78, $pos[1] + 15)
	sleep(2 * 1000)
	Send($patha)
	;
	; check for an error
	;
	Local $E_DIA = WinGetTitle("Open")
	While ($E_DIA = "Open Project or Algorithm")
		;
		; exit after try 5
		;
		IF ($icount = 5) Then
			Exit (5)
		EndIf
		;
		sleep(2 * 1000)
		Send('{Enter}') ; Exit the error dialog
		;
		; click and enter string again
		;
		Opt("MouseCoordMode", 0)
		$pos = ControlGetPos($Alghnd, "", "&Open")
		Opt("MouseCoordMode", 0)
		MouseClick("left", $pos[0] - 78, $pos[1] + 15)
		MouseClick("left")
		sleep(2 * 1000)
		Send($patha)
		;
		Local $E_DIA = WinGetTitle("Open")
		$icount += 1
		;
	WEnd
	;
	$icount = 1
	sleep(2 * 1000)
	;
EndFunc

Func _CreateExportDir($path, $ABx, ByRef $pathf)
	;
	; create a new export directory for inform
	;
	Local $expath = 'Z:\tmp_inform_data\' & $ABx & '\'
	Local $FileList = _FileListToArray($expath)
	Local $tmp = _FileListToArrayRec('C:\Users', "InForm_*", $FLTAR_FOLDERS + $FLTAR_NOHIDDEN + $FLTAR_NOSYSTEM)
	;
	IF IsArray($FileList) Then
		Local $FileNum = string($FileList[0] + 1)
	ELSE
		Local $FileNum = 1
	EndIf
	;
	$pathf = $expath & $tmp[1] & "_" & $FileNum
	DirCreate($pathf)
	If Not FileExists($pathf) Then
		Exit(6)
	EndIf
EndFunc

Func _GetExportDir($iFhnd, $pathf, $path)
	;
	; select the export folder
	;
	Local $str = StringSplit($path, "\")
	_ArrayDelete($str, "1;2")
	;
	Local $str2 = $str[2] & "(\\" & $str[1] & ")"
	$str = StringSplit($pathf, "\")
	_ArrayDelete($str, "1")
	;
	; scroll to the top of the inform screen
	;
	Local $val1 = ControlGetPos($iFhnd, "", "Images to export:")
	Opt("MouseCoordMode", 0)
	MouseClick("left",$val1[0], $val1[1])
	MouseWheel("up", 20)
	;
	; Open Browse Box
	;
	Local $expbut = ControlGetHandle($iFhnd, "", "Browse...")
	WinActivate($iFhnd)
	WinWaitActive($iFhnd)
	Local $Brtit = ""
	while $Brtit = ""
		if $icount = 5 Then
			Exit(6)
		EndIf
		ControlClick($iFhnd, "",$expbut)
		WinWait("Browse For Folder","",10)
		$Brtit = WinGetTitle("Browse")
		$icount += 1
	WEnd
	;
	$icount = 1
	Local $Brhnd = WinGetHandle($Brtit)
	WinActivate($Brhnd)
	WinWaitActive($Brhnd)
	sleep(1000)
	;
	; select the Desktop folder
	;
	Opt("MouseCoordMode", 0)
	MouseClick("left", 290, 100, 1, 5)
	sleep(1000)
	MouseWheel("up", 600)
	;
	MouseClick("left", 40, 95, 1, 5)
	sleep(1000)
	;
	; close the window to preserve the path to the desktop (used as a work around for mystery "ghost" paths filled in by the OS)
	;
	Send("{ENTER}")
	sleep(1000)
	;
	; Open the window again
	;
	WinActivate($iFhnd)
	WinWaitActive($iFhnd)
	$Brtit = ""
	while $Brtit = ""
		if $icount = 5 Then
			Exit(6)
		EndIf
		ControlClick($iFhnd, "",$expbut)
		WinWait("Browse For Folder","",10)
		$Brtit = WinGetTitle("Browse")
		$icount += 1
	WEnd
	;
	$icount = 1
	$Brhnd = WinGetHandle($Brtit)
	WinActivate($Brhnd)
	WinWaitActive($Brhnd)
	sleep(1000)
	;
	Opt("MouseCoordMode", 0)
    $hnd = ControlGetHandle($Brhnd, "", "[INSTANCE: 5]")
	MouseClick("left", 40, 95, 1, 5)
	Send("{RIGHT}")
	sleep(1000)
	;
	; The desktop folder will be selected but the
	;
	ControlSend($Brhnd, "", $hnd, "This PC")
	Send("{RIGHT}")
	;
	sleep(1000)
	Send($str2)
	;
	For $i1 = 1 To ($str[0] - 1)
		Send("{RIGHT}")
		Sleep(1000)
		Send($str[$i1])
	Next
	;
	WinActivate($Brhnd)
	WinWaitActive($Brhnd)
	;
	Send("{ENTER}")
	sleep(1000)
	;
EndFunc

Func _AddAllImages($iFhnd, $pathim)
	;
	; add all images
	;
	WinActivate($iFhnd)
	WinWaitActive($iFhnd)
	ControlClick($iFhnd, "", "Add Images...")
	WinWait("Open", "", 10)
	Local $Imtit = WinGetTitle("Open")
	Local $Imhnd = WinGetHandle($Imtit)
	WinActivate($Imhnd)
	Local $pos = ControlGetPos($Imhnd, "", "&Open")
	Opt("MouseCoordMode", 0)
	MouseClick("left", $pos[0] - 80, $pos[1] + 15)
	Send($pathim & '{Enter}')
	Sleep(2 * 1000)
	MouseClick("left", $pos[0], $pos[1] - 50)
	send("^a")
	sleep(2 * 1000)
	WinWaitActive("Open", "", 10)
	ControlClick($Imhnd, "", "&Open")
	sleep(3 * 1000)
	;
	; close files ignored dialog if it opens
	;
	Local $str2 = WinGetTitle("[ACTIVE]")
	If ($str2 = "Files Ignored") Then
		WinClose($str2)
	EndIf
	;
EndFunc

Func _RuninForm($iFhnd)
	;
	; click the run inForm Batch process button
	;
	WinWait($iFhnd,"",5)
	WinActivate($iFhnd)
	WinWaitActive($iFhnd,"",5)
	;
	Local $val1 = ControlGetPos($iFhnd, "", "Images to export:")
	Opt("MouseCoordMode", 0)
	MouseClick("left",$val1[0], $val1[1])
	MouseWheel("down", 20)
	sleep(500)
	$val1 = ControlGetHandle($iFhnd, "", "Run")
	ControlClick($iFhnd, "", $val1)
	;
	; check for inform error
	;
	WinWait("Batch", "", 20)
	Local $Run = WinGetHandle("Batch")
	If $Run = "" Then
		WinClose($iFhnd)
		EXIT(7)
	EndIf
	;
	_WaitForinForm($Run)
	ControlClick($Run, "", "Done")
	;
	WinWait("Problems in Batch Processing","",10)
	Local $err2 = WinGetTitle("Problems in Batch Processing")
	If Not $err2 = "" Then
		WinClose($err2)
	EndIf
	;
EndFunc

Func _WaitForinForm($Run)
;
Local $hndl = ControlGetText($Run, "", "[INSTANCE: 4]")
;
While $hndl <> "Done"
	Sleep(120 * 1000)
	$hndl = ControlGetText($Run, "", "[INSTANCE: 4]")
	;Local $hnd2 = ControlGetText($Run, "", "[INSTANCE: 3]")
	;MsgBox($MB_SYSTEMMODAL,"",$hndl & @CRLF & $hnd2)
WEND
;
EndFunc

Func _GetErrors($pathf, ByRef $tmpar, $sname)
	;
	; check for possible fixedable errors  in Batch.log
	;
	Local $pathb = $pathf & '\Batch.log'
	Local $errs
	_FileReadToArray($pathb, $errs)
	Local $val
	$tmpar = ""
	for $i1 = 1 To UBound($errs,1)-1 Step 1
		$val = $errs[$i1]
		if (StringInStr($val,"semaphore")) Or (StringInStr($val,"TIFF")) _
			Or (StringInStr($val,"Unable to find the specified file")) _
			Or (StringInStr($val,"Export problem")) Then
			$val = _StringBetween($val, "[", "]")
			$tmpar &= '"' & $sname & '_[' & $val[0] & '].im3" '
		EndIf
	Next
EndFunc

Func _RemoveErrs($iFhnd, $pathim, ByRef $tmpar, $sname, $path, $ABx)
	;
	; get a new folder
	;
	Local $pathfb
	While $tmpar
		_CreateExportDir($path, $ABx, $pathfb)
		_GetExportDir($iFhnd, $pathfb, $path)
		_AddSomeImages($iFhnd, $pathim, $tmpar)
		_RuninForm($iFhnd)
		_GetErrors($pathfb, $tmpar, $sname)
	WEnd
EndFunc

Func _AddSomeImages($iFhnd, $pathim, $tmpar)
	WinActivate($iFhnd)
	ControlClick($iFhnd, "", "Remove All")
	Sleep(1 * 1000)
	ControlClick($iFhnd, "", "Add Images...")
	WinWait("Open", "", 10)
	Local $Imtit = WinGetTitle("Open")
	Local $Imhnd = WinGetHandle($Imtit)
	WinActivate($Imhnd)
	Local $pos = ControlGetPos($Imhnd, "", "&Open")
	Opt("MouseCoordMode", 0)
	MouseClick("left", $pos[0] - 80, $pos[1] + 15)
	Send($pathim & '{Enter}')
	Sleep(2 * 1000)
	MouseClick("left", $pos[0] - 80, $pos[1] + 15)
	send($tmpar)
	sleep(2 * 1000)
	ControlClick($Imhnd, "", "&Open")
	sleep(3 * 1000)
	;
	; close files ignored dialog if it opens
	;
	Local $str2 = WinGetTitle("[ACTIVE]")
	If ($str2 = "Files Ignored") Then
		WinClose($str2)
	EndIf
	;
EndFunc
