import pandas as pd
import glob
import os
import shutil
import tqdm

# === BƯỚC 1: TÌM TẤT CẢ FILE .csv ===
file_paths = glob.glob("./datasets/IDS2017_raw/*.csv", recursive=True)

df_all = pd.DataFrame()

label_list = {
    "neris": 1,
    "rbot": 2,
    "virut": 3,
    "menti": 4,
    "sogou": 5,
    "murlo": 6,
    "nsisay": 7
}

# if os.path.exists("./datasets/IDS2017"):
#     shutil.rmtree("./datasets/IDS2017")

# labels:
#['BENIGN', 'Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'PortScan', 'SSH-Patator', 'Web Attack � Brute Force', 'Web Attack � Sql Injection', 'Web Attack � XSS']

label_mapping = {
    'BENIGN': 'benign',
    'Bot': 'bot', 
    'DDoS': 'ddos',
    'DoS GoldenEye': 'dos_goldeneye',
    'DoS Hulk': 'dos_hulk',
    'DoS Slowhttptest': 'dos_slowhttptest',
    'DoS slowloris': 'dos_slowloris',
    'FTP-Patator': 'ftp_patator',
    'Heartbleed': 'heartbleed',
    'Infiltration': 'infiltration',
    'PortScan': 'port_scan',
    'SSH-Patator': 'ssh_patator',
    'Brute Force': 'web_attack_bruteforce',
    'Sql Injection': 'web_attack_sql_injection',
    'XSS': 'web_attack_xss'
}

for file in tqdm.tqdm(file_paths, desc="Processing files"):
    try:
        df = pd.read_csv(file, sep=',', engine='python', encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        # Map labels to standardized values
        df.loc[:,'Label'] = df['Label'].map(label_mapping)

        # Split into attack and benign dataframes
        df_bot = df[df['Label'] == 'bot'].copy()
        df_ddos = df[df['Label'] == 'ddos'].copy()
        df_dos_goldeneye = df[df['Label'] == 'dos_goldeneye'].copy()
        df_dos_hulk = df[df['Label'] == 'dos_hulk'].copy()
        df_dos_slowhttptest = df[df['Label'] == 'dos_slowhttptest'].copy()
        df_dos_slowloris = df[df['Label'] == 'dos_slowloris'].copy()
        df_ftp_patator = df[df['Label'] == 'ftp_patator'].copy()
        df_heartbleed = df[df['Label'] == 'heartbleed'].copy()
        df_infiltration = df[df['Label'] == 'infiltration'].copy()
        df_port_scan = df[df['Label'] == 'port_scan'].copy()
        df_ssh_patator = df[df['Label'] == 'ssh_patator'].copy()
        df_web_attack_bruteforce = df[df['Label'] == 'web_attack_bruteforce'].copy()
        df_web_attack_sql_injection = df[df['Label'] == 'web_attack_sql_injection'].copy()
        df_web_attack_xss = df[df['Label'] == 'web_attack_xss'].copy()
        df_benign = df[df['Label'] == 'benign'].copy()

        # Set binary labels - 0 for benign, 1 for attacks
        df_benign.loc[:,'Label'] = 0
        for df_attack in [df_bot, df_ddos, df_dos_goldeneye, df_dos_hulk, df_dos_slowhttptest,
                         df_dos_slowloris, df_ftp_patator, df_heartbleed, df_infiltration,
                         df_port_scan, df_ssh_patator, df_web_attack_bruteforce,
                         df_web_attack_sql_injection, df_web_attack_xss]:
            df_attack.loc[:,'Label'] = 1

        os.makedirs("./datasets/IDS2017", exist_ok=True)
        
        # List of tuples mapping dataframes to their output filenames
        df_mapping = [
            (df_bot, 'bot'),
            (df_ddos, 'ddos'),
            (df_dos_goldeneye, 'dos_goldeneye'),
            (df_dos_hulk, 'dos_hulk'),
            (df_dos_slowhttptest, 'dos_slowhttptest'),
            (df_dos_slowloris, 'dos_slowloris'),
            (df_ftp_patator, 'ftp_patator'),
            (df_heartbleed, 'heartbleed'),
            (df_infiltration, 'infiltration'),
            (df_port_scan, 'port_scan'),
            (df_ssh_patator, 'ssh_patator'),
            (df_web_attack_bruteforce, 'web_attack_bruteforce'),
            (df_web_attack_sql_injection, 'web_attack_sql_injection'),
            (df_web_attack_xss, 'web_attack_xss'),
            (df_benign, 'benign_tot')
        ]

        # Save flows if they have data
        for df, name in df_mapping:
            if len(df) > 0:
                output_file = f'./datasets/IDS2017/{name}.csv'
                if os.path.exists(output_file):
                    df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"❌ Lỗi đọc {file}: {e}")