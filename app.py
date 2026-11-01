
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import date
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# =====================================================
# 1. CẤU HÌNH
# =====================================================

st.set_page_config(
    page_title="LaLiga xG Prediction Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xg_xgboost_model.joblib")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "models", "xg_xgboost_features.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "laliga_feature_engineered.csv")

# Chỉ số đánh giá mô hình trên tập Test
MODEL_METRICS = {
    "MAE": 0.582392,
    "RMSE": 0.737804,
    "R2": 0.199157
}

# =====================================================
# 2. GIAO DIỆN CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .sub-title {
        color: #64748B;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #172554;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #94A3B8;
        font-size: 12px;
    }

    .info-box {
        background: #EAF2FF;
        padding: 14px 18px;
        border-radius: 10px;
        color: #1D4ED8;
        margin: 10px 0 18px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button {
        background: #2563EB;
        color: white;
        border-radius: 9px;
        border: none;
        height: 45px;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover {
        background: #1D4ED8;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #172554;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    hr {
        border-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. TẢI DỮ LIỆU VÀ MÔ HÌNH
# =====================================================

@st.cache_resource
def load_resources():

    # Tải danh sách 29 features
    features = joblib.load(FEATURE_PATH)

    # Tải mô hình XGBoost JSON
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(
       os.path.join(BASE_DIR, "xgboost_model.json")
    )

        # Khôi phục Imputer
    imputer_path = os.path.join(BASE_DIR, "imputer_statistics.json")

    with open(imputer_path, "r", encoding="utf-8") as f:
        statistics = json.load(f)

    imputer = SimpleImputer(strategy="median")

    # Khởi tạo trạng thái đã fit
    imputer.fit(
        np.array(statistics, dtype=float).reshape(1, -1)
    )

    # Khôi phục median từ mô hình gốc
    imputer.statistics_ = np.array(statistics, dtype=float)

    # Ghép thành Pipeline
    model = Pipeline([
        ("imputer", imputer),
        ("model", xgb_model)
    ])

    # Tải dữ liệu
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])

    return model, features, data
try:
    model, xgb_features, data = load_resources()

except Exception as e:
    st.error(f"Không thể tải dữ liệu hoặc mô hình: {e}")
    st.stop()
# =====================================================
# 4. HÀM TẠO ĐẶC TRƯNG
# Giữ logic tương ứng với Notebook
# =====================================================

def get_team_features(team_name, opponent_name,
                      match_date, venue, data):

    match_date = pd.to_datetime(match_date)

    history = data[
        data["date"] < match_date
    ].copy()

    team_history = history[
        history["team"] == team_name
    ].sort_values("date")

    opponent_history = history[
        history["team"] == opponent_name
    ].sort_values("date")

    recent = team_history.tail(5)
    opponent_recent = opponent_history.tail(5)

    h2h_history = history[
        (history["team"] == team_name) &
        (history["opponent"] == opponent_name)
    ].sort_values("date")

    h2h_recent = h2h_history.tail(5)

    def avg(series):
        return series.mean() if len(series) > 0 else np.nan

    # Phong độ 5 trận gần nhất
    xg_avg5 = avg(recent["xg"])
    xga_avg5 = avg(recent["xga"])
    goals_avg5 = avg(recent["gf"])
    goals_against_avg5 = avg(recent["ga"])
    shots_avg5 = avg(recent["sh"])
    sot_avg5 = avg(recent["sot"])

    previous_matches = len(team_history)

    is_home = 1 if venue == "Home" else 0

    # Thống kê theo sân nhà/sân khách
    venue_history = team_history[
        team_history["venue"] == venue
    ].tail(5)

    venue_xg_avg5 = avg(venue_history["xg"])
    venue_xga_avg5 = avg(venue_history["xga"])

    # Số ngày nghỉ
    if len(team_history) > 0:
        rest_days = (
            match_date - team_history.iloc[-1]["date"]
        ).days
    else:
        rest_days = np.nan

    # Thống kê đối thủ
    opponent_xg_avg5 = avg(opponent_recent["xg"])
    opponent_xga_avg5 = avg(opponent_recent["xga"])
    opponent_goals_avg5 = avg(opponent_recent["gf"])
    opponent_goals_against_avg5 = avg(opponent_recent["ga"])
    opponent_sot_avg5 = avg(opponent_recent["sot"])

    # Lịch sử đối đầu
    h2h_xg_avg5 = avg(h2h_recent["xg"])
    h2h_xga_avg5 = avg(h2h_recent["xga"])
    h2h_goals_avg5 = avg(h2h_recent["gf"])
    h2h_goals_against_avg5 = avg(h2h_recent["ga"])
    h2h_shots_avg5 = avg(h2h_recent["sh"])
    h2h_sot_avg5 = avg(h2h_recent["sot"])

    # Đếm toàn bộ số trận đối đầu trước ngày dự đoán
    h2h_matches_count = len(h2h_history)

    has_h2h_history = int(h2h_matches_count > 0)

    # Đặc trưng nâng cao
    net_xg_avg5 = xg_avg5 - xga_avg5

    opponent_net_xg_avg5 = (
        opponent_xg_avg5 - opponent_xga_avg5
    )

    attack_defense_index = (
        xg_avg5 + opponent_xga_avg5
    ) / 2

    xg_avg3 = avg(team_history.tail(3)["xg"])

    form_trend = xg_avg3 - xg_avg5

    features = {
        "xg_avg5": xg_avg5,
        "xga_avg5": xga_avg5,
        "goals_avg5": goals_avg5,
        "goals_against_avg5": goals_against_avg5,
        "shots_avg5": shots_avg5,
        "sot_avg5": sot_avg5,
        "previous_matches": previous_matches,
        "is_home": is_home,
        "venue_xg_avg5": venue_xg_avg5,
        "venue_xga_avg5": venue_xga_avg5,
        "rest_days": rest_days,
        "opponent_xg_avg5": opponent_xg_avg5,
        "opponent_xga_avg5": opponent_xga_avg5,
        "opponent_goals_avg5": opponent_goals_avg5,
        "opponent_goals_against_avg5": opponent_goals_against_avg5,
        "opponent_sot_avg5": opponent_sot_avg5,
        "h2h_xg_avg5": h2h_xg_avg5,
        "h2h_xga_avg5": h2h_xga_avg5,
        "h2h_goals_avg5": h2h_goals_avg5,
        "h2h_goals_against_avg5": h2h_goals_against_avg5,
        "h2h_shots_avg5": h2h_shots_avg5,
        "h2h_sot_avg5": h2h_sot_avg5,
        "h2h_matches_count": h2h_matches_count,
        "net_xg_avg5": net_xg_avg5,
        "opponent_net_xg_avg5": opponent_net_xg_avg5,
        "attack_defense_index": attack_defense_index,
        "xg_avg3": xg_avg3,
        "form_trend": form_trend,
        "has_h2h_history": has_h2h_history
    }

    return pd.DataFrame(
        [[features[f] for f in xgb_features]],
        columns=xgb_features
    )


# =====================================================
# 5. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">⚽ LaLiga xG Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Hệ thống dự đoán bàn thắng kỳ vọng bằng Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

# =====================================================
# 6. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## ⚽ LaLiga xG")

    st.caption("MACHINE LEARNING PROJECT")

    st.divider()

    st.markdown("### Thông tin hệ thống")

    st.write("**Thuật toán:** XGBoost Regressor")
    st.write("**Số đặc trưng:** 29")
    st.write("**Dữ liệu:** LaLiga 2019–2025")
    st.write("**Số trận:** 2.350")

    st.divider()

    st.markdown("### Chức năng")

    st.write("⚽ Dự đoán xG")
    st.write("📊 Phân tích kết quả")
    st.write("🎯 Feature Importance")
    st.write("📈 Đánh giá mô hình")


# =====================================================
# 7. TỔNG QUAN MÔ HÌNH
# =====================================================

st.markdown(
    '<div class="section-title">Tổng quan mô hình</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Số trận dữ liệu", "2.350")
c2.metric("Số đặc trưng", "29")
c3.metric("Thuật toán", "XGBoost")
c4.metric("Giai đoạn dữ liệu", "2019–2025")

st.divider()


# =====================================================
# 8. DỰ ĐOÁN TRẬN ĐẤU
# =====================================================

st.markdown(
    '<div class="section-title">1. Thông tin trận đấu</div>',
    unsafe_allow_html=True
)

teams = sorted(data["team"].dropna().unique().tolist())

col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox(
        "Đội chủ nhà",
        teams,
        index=teams.index("Barcelona")
        if "Barcelona" in teams else 0
    )

with col2:
    away_default = (
        teams.index("Real Madrid")
        if "Real Madrid" in teams else 1
    )

    away_team = st.selectbox(
        "Đội khách",
        teams,
        index=away_default
    )

with col3:
    match_date = st.date_input(
        "Ngày thi đấu",
        value=date(2025, 10, 1)
    )

st.markdown(
    '<div class="info-box">'
    'ℹ️ Hệ thống chỉ sử dụng dữ liệu lịch sử trước ngày thi đấu '
    'được chọn để tạo đặc trưng dự đoán.'
    '</div>',
    unsafe_allow_html=True
)

predict_button = st.button(
    "⚽ DỰ ĐOÁN xG",
    use_container_width=True
)


# =====================================================
# 9. THỰC HIỆN DỰ ĐOÁN
# =====================================================

if predict_button:

    if home_team == away_team:
        st.error("Vui lòng chọn hai đội bóng khác nhau.")

    else:

        try:
            home_input = get_team_features(
                home_team,
                away_team,
                match_date,
                "Home",
                data
            )

            away_input = get_team_features(
                away_team,
                home_team,
                match_date,
                "Away",
                data
            )

            home_xg = float(model.predict(home_input)[0])
            away_xg = float(model.predict(away_input)[0])

            st.session_state["prediction"] = {
                "home": home_team,
                "away": away_team,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "date": str(match_date),
                "home_input": home_input,
                "away_input": away_input
            }

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")


# =====================================================
# 10. HIỂN THỊ KẾT QUẢ
# =====================================================

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    home_xg = result["home_xg"]
    away_xg = result["away_xg"]
    total_xg = home_xg + away_xg

    st.divider()

    st.markdown(
        '<div class="section-title">2. Kết quả dự đoán</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f'{result["home"]} vs {result["away"]} | '
        f'Ngày {result["date"]}'
    )

    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{result["home"]}</div>
                <div class="metric-value">{home_xg:.3f}</div>
                <div class="metric-note">Expected Goals - xG</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{result["away"]}</div>
                <div class="metric-value">{away_xg:.3f}</div>
                <div class="metric-note">Expected Goals - xG</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Tổng xG</div>
                <div class="metric-value">{total_xg:.3f}</div>
                <div class="metric-note">Tổng xG của hai đội</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Biểu đồ so sánh
    st.markdown(
        '<div class="section-title">3. So sánh xG giữa hai đội</div>',
        unsafe_allow_html=True
    )

    chart_df = pd.DataFrame({
        "Đội bóng": [result["home"], result["away"]],
        "xG dự đoán": [home_xg, away_xg]
    })

    fig = px.bar(
        chart_df,
        x="Đội bóng",
        y="xG dự đoán",
        text="xG dự đoán",
        color="Đội bóng",
        color_discrete_sequence=["#2563EB", "#F97316"]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        yaxis_title="Expected Goals (xG)",
        xaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # So sánh
    if home_xg > away_xg:
        st.info(
            f"📌 {result['home']} có xG dự đoán cao hơn "
            f"{result['away']}."
        )
    elif away_xg > home_xg:
        st.info(
            f"📌 {result['away']} có xG dự đoán cao hơn "
            f"{result['home']}."
        )
    else:
        st.info("Hai đội có xG dự đoán bằng nhau.")

    # Bảng đặc trưng
    st.markdown(
        '<div class="section-title">4. Đặc trưng đầu vào</div>',
        unsafe_allow_html=True
    )

    feature_view = pd.DataFrame({
        "Đặc trưng": xgb_features,
        result["home"]: result["home_input"].iloc[0].values,
        result["away"]: result["away_input"].iloc[0].values
    })

    st.dataframe(
        feature_view.round(4),
        use_container_width=True,
        hide_index=True
    )

    csv = feature_view.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Tải bảng đặc trưng CSV",
        data=csv,
        file_name="xg_prediction_features.csv",
        mime="text/csv"
    )


# =====================================================
# 11. FEATURE IMPORTANCE
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">5. Phân tích mức độ quan trọng của đặc trưng</div>',
    unsafe_allow_html=True
)

try:

    xgb_model = model.named_steps["model"]

    importance_df = pd.DataFrame({
        "Đặc trưng": xgb_features,
        "Mức độ quan trọng": xgb_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Mức độ quan trọng",
        ascending=False
    ).head(10)

    fig_importance = px.bar(
        importance_df.sort_values(
            "Mức độ quan trọng",
            ascending=True
        ),
        x="Mức độ quan trọng",
        y="Đặc trưng",
        orientation="h",
        text="Mức độ quan trọng",
        color="Mức độ quan trọng",
        color_continuous_scale="Blues"
    )

    fig_importance.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig_importance.update_layout(
        height=480,
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=20, r=50, t=20, b=20)
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

    st.caption(
        "Feature Importance thể hiện mức độ mô hình sử dụng "
        "mỗi đặc trưng trong quá trình dự đoán, không chứng minh quan hệ nhân quả."
    )

except Exception as e:
    st.warning(f"Không thể hiển thị Feature Importance: {e}")


# =====================================================
# 12. ĐÁNH GIÁ MÔ HÌNH
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">6. Đánh giá hiệu suất mô hình</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

m1.metric(
    "MAE",
    f'{MODEL_METRICS["MAE"]:.4f}',
    help="Sai số tuyệt đối trung bình trên tập Test."
)

m2.metric(
    "RMSE",
    f'{MODEL_METRICS["RMSE"]:.4f}',
    help="Căn bậc hai sai số bình phương trung bình trên tập Test."
)

m3.metric(
    "R²",
    f'{MODEL_METRICS["R2"]:.4f}',
    help="Hệ số xác định trên tập Test."
)

st.markdown("""
**Diễn giải:**

- **MAE:** Sai số tuyệt đối trung bình khoảng 0.582 xG.
- **RMSE:** Sai số bình phương trung bình căn bậc hai khoảng 0.738 xG.
- **R²:** Mô hình giải thích được khoảng 19.92% biến thiên của xG trong tập Test.
""")

# =====================================================
# 13. GIẢI THÍCH xG
# =====================================================

st.divider()

st.markdown(
    '<div class="section-title">7. Tìm hiểu về Expected Goals (xG)</div>',
    unsafe_allow_html=True
)

with st.expander("xG là gì? Nhấn để xem giải thích"):

    st.write("""
    **Expected Goals (xG)** là chỉ số thể hiện xác suất một cú sút
    trở thành bàn thắng dựa trên những đặc điểm của tình huống dứt điểm.

    Trong dự án này, hệ thống sử dụng dữ liệu lịch sử của các đội bóng
    LaLiga để dự đoán giá trị xG cho từng đội trước trận đấu.

    **Lưu ý:**

    - xG không phải số bàn thắng thực tế.
    - xG cao không đảm bảo đội bóng sẽ ghi được nhiều bàn thắng.
    - Kết quả là dự đoán của mô hình Machine Learning, không phải kết quả trận đấu.
    """)

# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#94A3B8;padding:15px;">
        ⚽ LaLiga xG Prediction System<br>
        Machine Learning Project | XGBoost Regressor
    </div>
    """,
    unsafe_allow_html=True
)
