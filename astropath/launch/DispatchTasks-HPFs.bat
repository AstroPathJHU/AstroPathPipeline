@ECHO OFF
::
SETLOCAL
::
SET code=%~dp0
SET code1=%code:~0,-8%
::
SET module=hpfs
::
PING -n 5 127.0.0.1>nul
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command^
 "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -command ""&{Import-Module %code1%; DispatchTasks %module%}""' -Verb RunAs}"
::
ENDLOCAL&
