::-----------------------------------------------------------------
:: script to run the matlab flatfielding and warping task on
:: a single sample directory,fully specified on the command line
::
:: Alex Szalay, Baltimore, 2018-07-14
:: Usage: doOneSampleCore F:\Clinical_Specimen F:\new F:\flatw M27_1
::-----------------------------------------------------------------
::
@ECHO OFF
SETLOCAL 
::
SET root=%1
SET flatw=%2
SET sample=%3
SET mcode=%~dp0
::
ECHO . 
ECHO doOneSample %root% %flatw% %samp%
ECHO   %date% %time%
::
CALL %mcode%fixM2 %root% %sample%
PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& '%mcode%ConvertIm3Path.ps1' '%root%' '%flatw%' '%sample%'" -s
CALL %mcode%flatwPath  %root% %flatw% %sample%
PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& '%mcode%ConvertIm3Path.ps1' '%root%' '%flatw%' '%sample%'" -i
CALL %mcode%extractLayer %flatw% %sample% 1
CALL %mcode%cleanPath %flatw% %sample% %root%
:: 
ENDLOCAL
::