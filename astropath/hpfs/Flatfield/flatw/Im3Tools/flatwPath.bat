::-----------------------------------------------------------------
:: flatwPath.bat
::
:: script to run the matlab flatfielding and warping task on a single 
:: sample directory, cantaining the .raw files, fully specified on the 
:: command line. The result will be a set of files in the same folder,
:: with the extension .fw
::
:: Alex Szalay, Baltimore, 2018-07-14
:: Usage: doFlatwAll F:\new3 M41_1
::-----------------------------------------------------------------
::
@ECHO OFF
::
SETLOCAL enabledelayedexpansion
::
IF "|%2|"=="||" ECHO Usage: flatwPath flatw sample && ENDLOCAL && EXIT /B;
::
SET flatw=%2
SET sample=%3
SET PP=%flatw%\%sample%
SET log=%flatw%\%sample%\flatw.log
SET mcode=%4
::
ECHO .
ECHO flatwPath %flatw% %sample%
ECHO   %time%
ECHO   fw path "%flatw%\%sample%"

:: CALL %mcode%\flatw\runflatw\application\runFlatw.exe %1 %2 %3

matlab -nosplash -nodesktop -minimize -sd %mcode%\mtools -r ";runFlatw('%1','%2','%3');exit(0);" -wait -logfile %log%
:: CALL %mcode%\Im3Tools\matlabEcho %log%
::
IF EXIST "%log%" DEL /Q %log%
ECHO   %time%
ENDLOCAL
::