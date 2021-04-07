::-----------------------------------------------------------------
::  TransferOneAnno.bat
::
:: script to transfer one annotations in one directory to the specimen
:: folders within another directory
::
:: Benjamin Green, JHU 01/31/2020
::
:: Usage: TransferOneAnno \\bki04\Clinical_Specimen_2\Upkeep and Progress\Halo annotations\Images\L1_1_Scan1_45540.annotations 
:: \\bki04\Clinical_Specimen_2
::-----------------------------------------------------------------
@ECHO OFF
SETLOCAL
::
IF "|%2|"=="||" ECHO Usage: TransferOneAnno root dest && ENDLOCAL && EXIT /B;
::
SET root=%1
SET dest=%2

::
ECHO 	TransferOneAnno %root% %dest%
::
IF NOT EXIST %root% ECHO directory not valid %root% && ENDLOCAL && EXIT /B;
IF NOT EXIST "%dest%" ECHO file not valid "%dest%" && ENDLOCAL && EXIT /B;
::
::
:: parse the filename to get the full desintation path
::
SET ff=%~n1
CALL SET after=%%ff:*Scan=%%
CALL SET specimen=%%ff:_Scan%after%=%%
SET scan=Scan%after:~0,1%
SET fulldestpath=%dest%\%specimen%\im3\%scan%\%specimen%_%scan%.annotations.polygons.xml
::
:: do the actual copying
::
echo f | XCOPY /s /y /z %root% %fulldestpath% > nul
::
ENDLOCAL