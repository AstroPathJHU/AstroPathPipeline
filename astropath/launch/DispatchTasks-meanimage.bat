@ECHO OFF
::
SETLOCAL
::
SET code=%~dp0
SET code1=%code:~0,-8%
::
SET module=meanimage
::
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command^
 "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -noexit -command ""&{Import-Module %code1%; DispatchTasks %module%}""' -Verb RunAs}"
::
ENDLOCAL&
