@ECHO OFF
SETLOCAL
::
SET main=\\bki05\c$\Processing_Specimens
SET code=\\bki05\g$\PowerShell\
::
sleep 5
START PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& '%code%ProcessInFormFiles.ps1' '%main%'"
::
ENDLOCAL