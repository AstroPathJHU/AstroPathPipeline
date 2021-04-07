::-----------------------------------------------------------------
::  TransferAnno.bat
::
:: script to transfer all annotations in one directory to the specimen
:: folders within another directory
::
:: Benjamin Green, JHU 01/31/2020
::
:: Usage: TransferAnno F:\Clinical_Specimen F:\flatw M27_1
::-----------------------------------------------------------------
@ECHO OFF
SETLOCAL
::
IF "|%2|"=="||" ECHO Usage: TransferAnno root dest && ENDLOCAL && EXIT /B;
::
SET root=%1
SET dest=%2
SET sdirs=\\bki08\astropath_code\TransferAnno
SET log=%sdirs%\TransferAnno.log
::
ECHO TransferAnno %root% %dest%
::
IF NOT EXIST %root% ECHO directory not valid %root% && ENDLOCAL && EXIT /B;
IF NOT EXIST "%dest%" ECHO directory not valid %dest% && ENDLOCAL && EXIT /B;
IF NOT EXIST "%sdirs%" ECHO source code directory not found %sdirs% && ENDLOCAL && EXIT /B; 
::
SET code=%sdirs%\TransferOneAnno.bat
::
::loop through each annotation file, copying and changing the name of each
::
FOR %%f in (%root%\*.annotations) do call "%code%" "%%f" %dest%
::
ENDLOCAL