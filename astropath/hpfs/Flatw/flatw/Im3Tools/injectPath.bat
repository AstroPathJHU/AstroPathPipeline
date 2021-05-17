::---------------------------------------------------------------------
:: injectPath.bat
::
:: inject the flatwarped images into the im3 files for a whole sample
:: Alex Szalay, Baltimore, 2018-06-12
::
:: Usage  : injectPath im3root flatw sample
:: example: injectPath F:\Clinical_Specimen F:\new M2_1
::---------------------------------------------------------------------
@ECHO OFF
::
SETLOCAL
::
IF "|%3|"=="||" ECHO Usage: injectPath dataroot flatw sample && ENDLOCAL && EXIT /B;
::
SET root1=%1
SET root2=%2
SET sample=%3
SET mcode=%4
::
SET code=%mcode%\Im3tools\InjectIm3.exe
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
:: build and test the src path to the fw images
::
SET src=%root2%\%sample%
IF NOT EXIST "%src%" ECHO flatw path "%src%" not found. && ENDLOCAL && EXIT /B;
::
ECHO .
ECHO injectPath %root1% %root2% %sample%
ECHO   %time%
::
SET dst=%root1%\%sample%\im3\flatw
IF NOT EXIST "%dst%" mkdir "%dst%"
::
SET log=%dst%\doInject.log
IF EXIST %log% DEL /Q %log%
::
ECHO   src path "%src%"
DIR %src%\*.fw | findstr "File"
::
ECHO   im3 path "%im3%"
DIR %im3%\*.im3 | findstr "File"
::
ECHO   dst path "%dst%"
::
::FOR %%f IN (%src%\*.fw) DO call echo %code%  "%im3%\%%~nf.im3" "%%f" -o "%dst%" 
FOR %%f IN (%src%\*.fw) DO call %code%  "%im3%\%%~nf.im3" "%%f" -o "%dst%" >> %log%
::
DIR %dst%\*.im3 | findstr "File"
ECHO   %time%
::
ENDLOCAL
::