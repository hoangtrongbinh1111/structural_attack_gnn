import pandas as pd
import glob
import os

# === BƯỚC 1: TÌM TẤT CẢ FILE .binetflow ===
file_paths = glob.glob("./datasets/CTU_raw/**/*.binetflow", recursive=True)

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

for file in file_paths:
    print(file)
    label_name = os.path.basename(file).replace('.binetflow','')
    try:
        df = pd.read_csv(file, sep=',', engine='python', encoding='utf-8', on_bad_lines='skip')

        df['Label'] = df['Label'].apply(lambda x: 1 if 'Botnet' in str(x) else 0)
        df_all = pd.concat([df_all, df], ignore_index=True)

        # Split into botnet and benign dataframes
        df_botnet = df[df['Label'] == 1]
        df_benign = df[df['Label'] == 0]
        
        # Save botnet flows
        botnet_file = './datasets/CTU/' + label_name + '.csv'
        if os.path.exists(botnet_file):
            # Append to existing file
            df_botnet.to_csv(botnet_file, mode='a', header=False, index=False)
        else:
            # Create new file
            df_botnet.to_csv(botnet_file, index=False)
            
        # Save benign flows 
        benign_file = './datasets/CTU/benign_tot.csv'
        if os.path.exists(benign_file):
            # Append to existing file
            df_benign.to_csv(benign_file, mode='a', header=False, index=False)
        else:
            # Create new file
            df_benign.to_csv(benign_file, index=False)
    except Exception as e:
        print(f"❌ Lỗi đọc {file}: {e}")

# print(df_all.head())

# # Get distinct labels
# print("\nDistinct Labels:")
# print(df_all['Label'].unique())
# print("\nLabel counts:")
# print(df_all['Label'].value_counts())



# === BƯỚC 2: LỌC LABEL "Botnet" và "Normal" ===
# df_all = df_all[df_all['Label'].isin(['Botnet', 'Normal'])]

# # === BƯỚC 3: GÁN TÊN BOTNET DỰA TRÊN FILE ===
# botnet_map = {
#     "capture20110810": "Neris",
#     "capture20110811": "Neris",
#     "capture20110812": "Neris",
#     "capture20110901": "Rbot",
#     "capture20110902": "Rbot",
#     "capture20110914": "Virut",
#     "capture20110915": "Murlo",
#     "capture20110916": "Menti"
# }

# def detect_botnet(filename):
#     for key, val in botnet_map.items():
#         if key in filename:
#             return val
#     return "Other"

# df_all["BotnetFamily"] = df_all["filename"].apply(detect_botnet)

# # === BƯỚC 4: TÁCH CÁC TẬP RIÊNG ===
# df_neris = df_all[(df_all["Label"] == "Botnet") & (df_all["BotnetFamily"] == "Neris")]
# df_rbot = df_all[(df_all["Label"] == "Botnet") & (df_all["BotnetFamily"] == "Rbot")]
# df_virut = df_all[(df_all["Label"] == "Botnet") & (df_all["BotnetFamily"] == "Virut")]
# df_murlo = df_all[(df_all["Label"] == "Botnet") & (df_all["BotnetFamily"] == "Murlo")]
# df_menti = df_all[(df_all["Label"] == "Botnet") & (df_all["BotnetFamily"] == "Menti")]
# df_benign = df_all[df_all["Label"] == "Normal"]

# # === BƯỚC 5: GHI RA FILE CSV ===
# output_dir = "processed_ctu13_flows"
# os.makedirs(output_dir, exist_ok=True)

# df_neris.to_csv(f"{output_dir}/flows_neris.csv", index=False)
# df_rbot.to_csv(f"{output_dir}/flows_rbot.csv", index=False)
# df_virut.to_csv(f"{output_dir}/flows_virut.csv", index=False)
# df_murlo.to_csv(f"{output_dir}/flows_murlo.csv", index=False)
# df_menti.to_csv(f"{output_dir}/flows_menti.csv", index=False)
# df_benign.to_csv(f"{output_dir}/flows_benign.csv", index=False)

# print("✅ Xuất dữ liệu thành công tại thư mục:", output_dir)
