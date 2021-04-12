::----------------------------------------------------------------------
:: shredPath.bat
::
:: shred the im3 files for a whole sample
:: Alex Szalay, Baltimore, 2018-06-12
::
:: usage:
::		shredPath E:\Clinical_Specimen F:\new M1_1
:: To extract the already flattened:
::		shredPath E:\Clinical_Specimen F:\DAPI M1_1
::----------------------------------------------------------------------
@ECHO OFF
SETLOCAL
::
IF "|%3|"=="||" ECHO Usage: shredPath dataroot dest sample && ENDLOCAL && EXIT /B;
::
SET mcode=%4
SET code=%mcode%\Im3Tools\ShredIm3.exe
SET root1=%1
SET root2=%2
SET sample=%3
::
:: find the highest scan number, exit if im3 directory not found
::
set test=%root1%\%sample%\im3
IF NOT EXIST "%test%" ECHO IM3 root path "%test%" not found. && ENDLOCAL && EXIT /B;
FOR /F "tokens=*" %%g IN ('DIR %test%\Scan* /A:D /B') do (SET scanpath=%%g)
::
:: build the full im3 path, exit if not found
::
SET im3=%test%\%scanpath%\MSI
IF NOT EXIST "%im3%" ECHO IM3 subpath "%im3%" not found. && ENDLOCAL && EXIT /B;
::
:: build the destination path, and create folders if they do not exist
::
SET dst=%root2%\%sample%
IF NOT EXIST "%root2%" mkdir "%root2%"
IF NOT EXIST "%root2%\%sample%" mkdir "%root2%\%sample%"
::
ECHO .
ECHO shredPath %root1% %root2% %sample%
ECHO   %time%
ECHO   im3 path "%im3%"
DIR %im3%\*.im3 | findstr "File"
::
IF EXIST "%dst%\doShred.log" del "%dst%\doShred.log"
::
FOR %%f in (%im3%\*.im3) do "%code%" %%f -o %dst% >> "%dst%\doShred.log"
::
ECHO   dst path "%dst%"
DIR %dst%\*.raw | findstr "File"
::
ECHO   %time%
::
ENDLOCAL

