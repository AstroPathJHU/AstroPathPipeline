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
::
SET mcode=%4
::
ECHO . 
ECHO doOneSample %root% %flatw% %samp%
ECHO   %date% %time%
::
CALL %mcode%\Im3Tools\fixM2 %root% %sample% %mcode%
CALL %mcode%\Im3Tools\shredPath  %root% %flatw% %sample% %mcode%
CALL %mcode%\Im3Tools\flatwPath  %root% %flatw% %sample% %mcode%
CALL %mcode%\Im3Tools\injectPath %root% %flatw% %sample% %mcode%
CALL %mcode%\Im3Tools\extractLayer %flatw% %sample% 1 %mcode%
CALL %mcode%\Im3Tools\cleanPath %flatw% %sample% %mcode%
:: 
ENDLOCAL
::