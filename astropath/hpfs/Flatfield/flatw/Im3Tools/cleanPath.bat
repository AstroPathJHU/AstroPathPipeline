::-----------------------------------------------------------------
::  cleanPath.bat
::
:: script to remove the unncesseray images from the staging 
:: directory of a single sample
::
:: Alex Szalay, Baltimore, 2018-07-14
:: Usage: cleanPath Z:\test M41_1
::-----------------------------------------------------------------
::
@ECHO OFF
SETLOCAL enabledelayedexpansion
::
IF "|%2|"=="||" ECHO Usage: cleanPath flatw sample && ENDLOCAL && EXIT /B;
::
SET flatw=%1
SET sample=%2
SET root=%3\%sample%\im3\xml\
SET dest=%flatw%\%sample%
::
ECHO .
ECHO cleanPath %flatw% %sample%
ECHO   %time% 
ECHO   dest %dest%
::
:: clean up the .imm and .raw files
::
DEL /Q %dest%\*.dat
XCOPY /Q /Y /Z %dest%\*.xml %root% 
DEL /Q %dest%\*.xml
::
ECHO   %time% 
::
ENDLOCAL
::