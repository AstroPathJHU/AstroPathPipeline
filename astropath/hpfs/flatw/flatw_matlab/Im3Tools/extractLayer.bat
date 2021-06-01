::-----------------------------------------------------------------
:: extractLayer.bat
::
:: script to run the matlab flatfielding and warping task on a
:: single sample directory, fully specified on the command line
::
:: Alex Szalay, Baltimore, 2018-07-14
:: Usage: extractLayer F:\dapi * 1
::    or  extractLayer F:\dapi M41_1 1
::-----------------------------------------------------------------
::
@ECHO OFF
SETLOCAL enabledelayedexpansion
::
IF "|%3|"=="||" ECHO Usage: extractLayer flatw sample layer && ENDLOCAL && EXIT /B;
::
SET flatw=%1
SET sample=%2
SET layer=%3
SET mcode=%~dp0..\mtools
::
SET log=%flatw%\%sample%\layer%layer%.log
::
ECHO .
ECHO extractLayer %flatw% %sample% %layer%
ECHO   %time% 
::
matlab -minimize -nosplash -nodesktop -sd %mcode% -r "extractLayer('%flatw%','%sample%','%layer%');quit(0);" -wait -logfile %log%
:: CALL %mcode%\Im3Tools\matlabEcho %log%
::
IF EXIST "%log%" DEL /Q %log% 
ECHO   %time%
ENDLOCAL
::