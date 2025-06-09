@echo off
set "ctu_dir=datasets\ToN-IoT\"
set "preproc_dir=preprocessed_data\"

for %%A in (backdoor ddos dos injection password ransomware scanning xss) do (
    echo -------------------------------------
    echo ---------ATTACK: %%A-----------
    echo -------------------------------------

    echo -----------Preprocessing-------------
    python preprocessing_ToN.py --mal "%ctu_dir%%%A.csv" --ben "%ctu_dir%benign_tot.csv"
)
