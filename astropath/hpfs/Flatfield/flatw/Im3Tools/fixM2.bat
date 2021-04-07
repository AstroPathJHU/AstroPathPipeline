::-----------------------------------------------------------------
:: fixM2.bat
::
:: Fix all filenames that were created due to an error.
:: In these cases the original .im3 file has been truncated,
:: it exists but cannot be used. The Vectra system then
:: re-wrote the file, but padded the filename with _M2.
:: Here we do two things: if there is an _M2 file, we first
:: delete the file with the short length, then rename the file.
::
:: Alex Szalay, Baltimore 2018-07-13
::
:: Usage: fixM2('F:\Clinical_Specimen','M2_1')
::-----------------------------------------------------------------
::
@ECHO OFF
::
SETLOCAL enabledelayedexpansion
::
IF "|%2|"=="||" ECHO Usage: fixM2 root sample && ENDLOCAL && EXIT /B;
::
SET root=%1
SET sample=%2
SET log=%root%\%sample%\fixM2.log
SET mcode=%3
::
ECHO .
ECHO fixM2 %root% %sample%
ECHO   %time%
::
matlab -nosplash -nodesktop -minimize -sd %mcode%\mtools -r "; fixM2('%root%','%sample%');exit(0);" -wait -logfile %log%
:: CALL %mcode%\Im3Tools\matlabEcho %log%
::
IF EXIST "%log%" DEL /Q %log%
ECHO   %time%
ENDLOCAL
::