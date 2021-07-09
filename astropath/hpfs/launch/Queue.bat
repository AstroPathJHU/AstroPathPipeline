@ECHO OFF
::
SETLOCAL
::
SET code=%~dp0..\..\..
SET module=inform
::
sleep 5
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -noexit -command ""&{Import-Module %code%; Queue %module%}""' -Verb RunAs}"
:: START powershell -noprofile -executionpolicy bypass -noexit -command " &{Import-Module %code%; Queue %module%}"
::
ENDLOCAL&