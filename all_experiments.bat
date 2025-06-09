@echo off
setlocal enabledelayedexpansion

REM Directories
set ctu_dir=datasets\CTU\
set ton_dir=datasets\ToN-IoT\
set preproc_dir=preprocessed_data\

REM Process CTU attacks
for %%A in (neris rbot virut menti murlo) do (
    echo -------------------------------------
    echo ---------BOTNET: %%A-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    @REM python preprocessing.py --mal %ctu_dir%%%A.csv --ben %ctu_dir%benign_tot.csv 
    
    echo -------Training E-GRAPHSAGE----------
    python train_egraphsage.py --train %preproc_dir%CTU\%%A_train.csv --test %preproc_dir%CTU\%%A_test.csv
    
    echo -------Testing E-GRAPHSAGE-----------
    python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv
    python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv --feature_attack
    python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack benign_from_C
    python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack malicious_from_C
    python baseline_test_egraphsage.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack add_node

    echo -------Training LINE-GRAPHSAGE-------
    python batch_train_linegraphsage.py --train %preproc_dir%CTU\%%A_train.csv --test %preproc_dir%CTU\%%A_test.csv
    
    echo -------Testing LINE-GRAPHSAGE--------
    python baseline_test_linegraph.py --test %preproc_dir%CTU\%%A_test.csv
    python baseline_test_linegraph.py --test %preproc_dir%CTU\%%A_test.csv --feature_attack
    python baseline_test_linegraph.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack benign_from_C
    python baseline_test_linegraph.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack malicious_from_C
    python baseline_test_linegraph.py --test %preproc_dir%CTU\%%A_test.csv --structure_attack add_node
)

REM Process ToN-IoT attacks
for %%B in (backdoor ddos dos injection password ransomware scanning xss) do (
    echo -------------------------------------
    echo ---------ATTACK: %%B-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    @REM python preprocessing_ToN.py --mal %ton_dir%%%B.csv --ben %ton_dir%benign_tot.csv 
    
    echo -------Training E-GRAPHSAGE----------
    python train_egraphsage.py --train %preproc_dir%ToN_IoT\%%B_train.csv --test %preproc_dir%ToN_IoT\%%B_test.csv
    
    echo -------Testing E-GRAPHSAGE-----------
    python baseline_test_egraphsage.py --test %preproc_dir%ToN_IoT\%%B_test.csv
    python baseline_test_egraphsage.py --test %preproc_dir%ToN_IoT\%%B_test.csv --feature_attack
    python baseline_test_egraphsage.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack benign_from_C
    python baseline_test_egraphsage.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack malicious_from_C
    python baseline_test_egraphsage.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack add_node

    echo -------Training LINE-GRAPHSAGE-------
    python batch_train_linegraphsage.py --train %preproc_dir%ToN_IoT\%%B_train.csv --test %preproc_dir%ToN_IoT\%%B_test.csv
    
    echo -------Testing LINE-GRAPHSAGE--------
    python baseline_test_linegraph.py --test %preproc_dir%ToN_IoT\%%B_test.csv
    python baseline_test_linegraph.py --test %preproc_dir%ToN_IoT\%%B_test.csv --feature_attack
    python baseline_test_linegraph.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack benign_from_C
    python baseline_test_linegraph.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack malicious_from_C
    python baseline_test_linegraph.py --test %preproc_dir%ToN_IoT\%%B_test.csv --structure_attack add_node
)

echo ==== ALL TASKS COMPLETED ====
pause
