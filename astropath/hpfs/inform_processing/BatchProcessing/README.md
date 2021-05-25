RunFullBatch.au3:
#
from cmd
START /WAIT \\halo1\Backup\Software\Auotit\RunFullBatch.exe "2.4.3" "\\bki04\e$\Clinical_Specimen,M1_1,CD8,CD8_outlier.ifp,"
#
from powershell
Start-Process -FilePath "C:\Users\InForm_13\Desktop\RunFullBatch.exe" -ArgumentList "2.4.3","\\bki04\g$\Clinical_Specimen_3,MA1,CD8,CD8_pheno_Lib_CD20_01172019.ifp," -WAIT
#
inform_worker:
#
PowerShell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command "& {Start-Process PowerShell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ""C:\Program Files\BatchProcessing\RunVMBatch.ps1""' -Verb RunAs}"