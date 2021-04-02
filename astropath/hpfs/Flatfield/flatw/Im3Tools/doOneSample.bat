::-----------------------------------------------------------------
::  doOneSample.bat
::
:: script to run the matlab flatfielding and warping task on
:: a single sample directory,fully specified on the command line.
:: It calls the function doOneSampleCore to be able to cleanly
:: redirect all the log outputs into one file.
::
:: Alex Szalay, Baltimore, 2018-07-14
::
:: Usage: doOneSample F:\Clinical_Specimen F:\flatw M27_1
::-----------------------------------------------------------------
@ECHO OFF
SETLOCAL 
::
IF "|%3|"=="||" ECHO Usage: doOneSample root flatw sample && ENDLOCAL && EXIT /B;
::
SET root=%1
SET flatw=%2
SET sample=%3
SET mcode=%4
::
ECHO doOneSample %root% %flatw% %sample%
::
IF NOT EXIST "%flatw%" mkdir "%flatw%"
IF NOT EXIST "%flatw%\%sample%" mkdir "%flatw%\%sample%"
::
SET log=%flatw%\%sample%\flatwlog.log
::SET mcode=\\BKI05\E$\Processing_Specimens\AS_code
::
CALL %mcode%\Im3Tools\doOneSampleCore %root% %flatw% %sample% %mcode% > %log%
:: 
ENDLOCAL
