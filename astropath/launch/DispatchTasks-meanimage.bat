@ECHO OFF
::
SETLOCAL
::
SET code=%~dp0..
::
SET module=meanimage
::
sleep 5
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command^
 "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -noexit -command ""&{Import-Module %code%; DispatchTasks %module%}""' -Verb RunAs}"
::
ENDLOCAL&
