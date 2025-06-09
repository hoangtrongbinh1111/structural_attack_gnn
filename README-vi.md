# Hướng dẫn chuẩn bị dữ liệu và chạy code

1. Chuẩn bị dữ liệu
    - Tải 2 bộ dữ liệu CTU-13 và ToN-IoT datasets từ link gg drive
    - Giải nén và đưa 2 thư mục CTU_raw và ToN-IoT_raw vào folder datasets

2. Tiền xử lý dữ liệu
    - Lần lượt chạy 2 file ctu_insight.py và ton_iot_insight.py
    - Dữ liệu được xử lý xong sẽ lưu vào 2 folder là /datasets/CTU và /dataset/ToN-IoT

3. Chạy các thử nghiệm
    - Xem toàn bộ thí nghiệm chạy ở file all_experiments.sh
    - Với Windows, chạy file all_experiments.bat

4. Cài đặt môi trường
    - Anaconda
    - Lệnh chạy:
        - `conda create -n gnn_env python=3.10 cudatoolkit`
        - `pip install torch==2.1.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
        - `conda install -c dglteam/label/cu118 dgl`
        - `pip install numpy==1.24.4 scipy==1.10.1`
        - `pip install -r requirements.txt`
        - Nếu còn thiếu môi trường gì mọi người cài thêm, vì thực sự cũng chưa đầy đủ các thư viện
