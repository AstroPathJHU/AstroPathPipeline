@ECHO OFF

SETLOCAL

::

SET main=\\bki04\astropath_processing\across_project_queues

SET code=%~dp0\..\inform_processing\

::

sleep 5

START PowerShell -NoProfile -ExecutionPolicy Bypass -Command "& '%code%inform_queue.ps1' '%main%'"

::

ENDLOCAL