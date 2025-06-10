@echo off
set "ctu_dir=datasets\IDS2017\"
set "preproc_dir=preprocessed_data\"
set "model_dir=preprocess\IDS2017\"

for %%A in (heartbleed) do (
    echo -------------------------------------
    echo ---------ATTACK: %%A-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    python %model_dir%preprocessing_ids2017.py --mal "%ctu_dir%%%A.csv" --ben "%ctu_dir%benign_tot.csv"
)
