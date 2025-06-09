@echo off
setlocal enabledelayedexpansion

REM Directories
set ctu_dir=datasets\CTU\
set preproc_dir=preprocessed_data\

REM Process CTU attacks
for %%A in (neris) do (
    echo -------------------------------------
    echo ---------BOTNET: %%A-----------
    echo -------------------------------------
    python train_gat.py --train %preproc_dir%CTU\%%A_train.csv --test %preproc_dir%CTU\%%A_test.csv
    
    echo -------Testing GAT-----------
    @REM python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv
    @REM python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack benign_from_C
)

echo ==== ALL TASKS COMPLETED ====
pause
