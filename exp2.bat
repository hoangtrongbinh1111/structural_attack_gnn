@echo off
set "ctu_dir=datasets\ToN-IoT\"
set "preproc_dir=preprocessed_data\"

for %%A in (neris rbot virut menti murlo sogou nsisay) do (
    echo -------------------------------------
    echo ---------BOTNET: %%A-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    python preprocessing.py --mal "%ctu_dir%%%A.csv" --ben "%ctu_dir%benign_tot.csv"
)
