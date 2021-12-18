@ECHO OFF
::
SETLOCAL
::
SET code=%~dp0..
:: for /f "delims=" %%a in ('dir  /b %code%\astropath*') do set "name=%%a"
:: SET code=%code%\%name%
::
SET module=batchmicomp
::
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command^
 "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -noexit -command ""&{Import-Module %code%; DispatchTasks %module%}""' -Verb RunAs}"
::
ENDLOCAL&
