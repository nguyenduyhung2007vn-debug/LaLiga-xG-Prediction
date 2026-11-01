# LaLiga xG Prediction System

## 1. Giới thiệu
Hệ thống dự đoán Expected Goals (xG) của các đội bóng LaLiga
bằng thuật toán XGBoost Regressor.

## 2. Công nghệ
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly

## 3. Cấu trúc dự án
- app.py: Giao diện dự đoán.
- data/: Dataset đã Feature Engineering.
- models/: Mô hình và thông tin đánh giá.
- screenshots/: Ảnh minh họa dashboard.

## 4. Cài đặt
Cài Python 3.10 hoặc phiên bản tương thích với các thư viện.

Mở Terminal tại thư mục dự án và chạy:

    python -m venv venv

Windows:

    venv\Scripts\activate

Cài thư viện:

    pip install -r requirements.txt

## 5. Khởi chạy

    streamlit run app.py

Mở trình duyệt tại:

    http://localhost:8501

## 6. Chỉ số đánh giá trên tập Test
- MAE: 0.582392
- RMSE: 0.737804
- R2: 0.199157

## 7. Lưu ý
xG là giá trị bàn thắng kỳ vọng do mô hình dự đoán,
không phải số bàn thắng thực tế hoặc xác suất thắng trận.
