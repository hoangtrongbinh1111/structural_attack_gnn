import pandas as pd
import glob
import os

file_path = "./datasets/ToN-IoT_raw/train_test_network.csv"

try:
    print(f"Reading {file_path}")
    df = pd.read_csv(file_path, sep=',', engine='python', encoding='utf-8', on_bad_lines='skip')

    # print("\nDistinct Labels:")
    # print(df['label'].unique())
    # print("\nLabel counts:")
    # print(df['label'].value_counts())
    
    # print("\nDistinct Types:")
    # print(df['type'].unique()) 
    # print("\nType counts:")
    # print(df['type'].value_counts())

    # Split into botnet and benign dataframes
    df_backdoor = df[df['type'] == 'backdoor']
    df_ddos = df[df['type'] == 'ddos'] 
    df_dos = df[df['type'] == 'dos']
    df_injection = df[df['type'] == 'injection']
    df_mitm = df[df['type'] == 'mitm']
    df_normal = df[df['type'] == 'normal']
    df_password = df[df['type'] == 'password']
    df_ransomware = df[df['type'] == 'ransomware']
    df_scanning = df[df['type'] == 'scanning']
    df_xss = df[df['type'] == 'xss']

    # Save attack flows
    df_backdoor.to_csv('./datasets/ToN-IoT/backdoor.csv', index=False)
    df_ddos.to_csv('./datasets/ToN-IoT/ddos.csv', index=False)
    df_dos.to_csv('./datasets/ToN-IoT/dos.csv', index=False)
    df_injection.to_csv('./datasets/ToN-IoT/injection.csv', index=False)
    df_mitm.to_csv('./datasets/ToN-IoT/mitm.csv', index=False)
    df_password.to_csv('./datasets/ToN-IoT/password.csv', index=False)
    df_ransomware.to_csv('./datasets/ToN-IoT/ransomware.csv', index=False)
    df_scanning.to_csv('./datasets/ToN-IoT/scanning.csv', index=False)
    df_xss.to_csv('./datasets/ToN-IoT/xss.csv', index=False)
    
    # Save benign flows
    df_normal.to_csv('./datasets/ToN-IoT/benign_tot.csv', index=False)
except Exception as e:
    print(f"❌ Lỗi đọc file: {e}")
