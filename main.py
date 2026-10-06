import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="서울 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300

TEST_START = 2006
TEST_END = 2025


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    # 평균기온 결측치 제거
    annual = annual.dropna(subset=["연평균기온"])

    # 회귀용 연수
    annual["1908년부터_지난_연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# --------------------------------------------------
# 선형회귀 함수
# --------------------------------------------------
def make_regression(data):
    x = data["1908년부터_지난_연수"].to_numpy()
    y = data["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    return slope, intercept


def predict_temperature(years, slope, intercept):
    x = np.asarray(years) - BASE_YEAR
    return slope * x + intercept


# --------------------------------------------------
# 전체 데이터 회귀
# --------------------------------------------------
all_slope, all_intercept = make_regression(annual)

annual["전체데이터_회귀예측"] = predict_temperature(
    annual["연도"],
    all_slope,
    all_intercept,
)

correlation = annual["연도"].corr(
    annual["연평균기온"]
)

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# --------------------------------------------------
# 학습 / 테스트 데이터 분리
# --------------------------------------------------

# 최근 50년: 1956~2005 학습
train_50 = annual[
    (annual["연도"] >= 1956)
    & (annual["연도"] <= 2005)
].copy()

# 최근 100년: 1906~2005 학습
train_100 = annual[
    (annual["연도"] >= 1906)
    & (annual["연도"] <= 2005)
].copy()

# 공통 테스트 데이터: 2006~2025
test = annual[
    (annual["연도"] >= TEST_START)
    & (annual["연도"] <= TEST_END)
].copy()


# --------------------------------------------------
# 50년 모델
# --------------------------------------------------
slope_50, intercept_50 = make_regression(train_50)

test["50년_예측"] = predict_temperature(
    test["연도"],
    slope_50,
    intercept_50,
)

mae_50 = mean_absolute_error(
    test["연평균기온"],
    test["50년_예측"],
)

mse_50 = mean_squared_error(
    test["연평균기온"],
    test["50년_예측"],
)

r2_50 = r2_score(
    test["연평균기온"],
    test["50년_예측"],
)


# --------------------------------------------------
# 100년 모델
# --------------------------------------------------
slope_100, intercept_100 = make_regression(train_100)

test["100년_예측"] = predict_temperature(
    test["연도"],
    slope_100,
    intercept_100,
)

mae_100 = mean_absolute_error(
    test["연평균기온"],
    test["100년_예측"],
)

mse_100 = mean_squared_error(
    test["연평균기온"],
    test["100년_예측"],
)

r2_100 = r2_score(
    test["연평균기온"],
    test["100년_예측"],
)


# --------------------------------------------------
# 화면 제목
# --------------------------------------------------
st.title("🌡️ 서울 연평균 기온 선형회귀 분석")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 선형회귀 모델을 만들고, "
    "1956~2005년과 1906~2005년을 각각 학습한 모델이 "
    "공통 테스트 기간인 2006~2025년의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# --------------------------------------------------
# 데이터 구성
# --------------------------------------------------
st.subheader("📊 데이터 구성")

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 데이터",
    f"{year_count}개 연도"
)

col2.metric(
    "학습 데이터",
    "1956~2005 / 1906~2005"
)

col3.metric(
    "공통 테스트 데이터",
    "2006~2025"
)

st.info(
    "1956~2005년 또는 1906~2005년을 학습 데이터로 사용하고, "
    "두 모델 모두 동일한 2006~2025년을 테스트 데이터로 사용합니다."
)


# --------------------------------------------------
# 전체 데이터 요약통계
# --------------------------------------------------
st.subheader("📋 원본 연평균 기온 데이터 요약통계")

summary = pd.DataFrame({
    "항목": [
        "개수",
        "평균",
        "최소",
        "최대",
    ],
    "값": [
        len(annual),
        annual["연평균기온"].mean(),
        annual["연평균기온"].min(),
        annual["연평균기온"].max(),
    ]
})

summary["값"] = [
    f"{summary.iloc[0, 1]:.0f}개",
    f"{summary.iloc[1, 1]:.2f} ℃",
    f"{summary.iloc[2, 1]:.2f} ℃",
    f"{summary.iloc[3, 1]:.2f} ℃",
]

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 전체 데이터 회귀선
# --------------------------------------------------
st.subheader("📈 전체 기간의 연평균 기온과 회귀선")

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

line_years = np.arange(
    start_year,
    end_year + 1
)

line_temperature = predict_temperature(
    line_years,
    all_slope,
    all_intercept,
)

fig_all.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperature,
        mode="lines",
        name="전체 데이터 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
)

fig_all.update_xaxes(
    tickformat="d",
    range=[start_year - 2, end_year + 2],
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)

st.write(
    f"전체 데이터 회귀선의 기울기: "
    f"**{all_slope:.5f} ℃/년**"
)

st.write(
    f"100년 기준 변화량: "
    f"**{all_slope * 100:.2f} ℃/100년**"
)

st.write(
    f"전체 데이터의 상관계수: "
    f"**{correlation:.4f}**"
)


# --------------------------------------------------
# 50년 / 100년 학습 데이터 비교
# --------------------------------------------------
st.subheader("📚 학습 데이터별 회귀선 비교")

fig_train = go.Figure()

# 실제 전체 데이터
fig_train.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5,
            opacity=0.55,
        ),
    )
)

# 100년 회귀선
years_100 = np.arange(
    1906,
    2006
)

fig_train.add_trace(
    go.Scatter(
        x=years_100,
        y=predict_temperature(
            years_100,
            slope_100,
            intercept_100,
        ),
        mode="lines",
        name="1906~2005 학습 회귀선",
    )
)

# 50년 회귀선
years_50 = np.arange(
    1956,
    2006
)

fig_train.add_trace(
    go.Scatter(
        x=years_50,
        y=predict_temperature(
            years_50,
            slope_50,
            intercept_50,
        ),
        mode="lines",
        name="1956~2005 학습 회귀선",
    )
)

fig_train.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
)

st.plotly_chart(
    fig_train,
    use_container_width=True
)


# --------------------------------------------------
# 기울기 비교
# --------------------------------------------------
st.subheader("📐 회귀선 기울기 비교")

slope_comparison = pd.DataFrame({
    "모델": [
        "최근 50년 모델",
        "최근 100년 모델",
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005",
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100,
    ],
    "100년당 변화량 (℃/100년)": [
        slope_50 * 100,
        slope_100 * 100,
    ],
})

slope_comparison["기울기 (℃/년)"] = (
    slope_comparison["기울기 (℃/년)"]
    .round(5)
)

slope_comparison["100년당 변화량 (℃/100년)"] = (
    slope_comparison["100년당 변화량 (℃/100년)"]
    .round(2)
)

st.dataframe(
    slope_comparison,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 테스트 데이터 예측 그래프
# --------------------------------------------------
st.subheader("🎯 2006~2025년 테스트 데이터 예측")

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 연평균기온",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_예측"],
        mode="lines",
        name="50년 학습 모델 예측",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델 예측: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_예측"],
        mode="lines",
        name="100년 학습 모델 예측",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델 예측: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
)

fig_test.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# --------------------------------------------------
# 예측 성능 평가
# --------------------------------------------------
st.subheader("📊 테스트 데이터 예측 성능")

st.write(
    "두 모델 모두 학습에 사용하지 않은 **2006~2025년 데이터**로 평가했습니다."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown("### 최근 50년 모델")
    st.write("학습: **1956~2005년**")
    st.metric("MAE", f"{mae_50:.3f} ℃")
    st.metric("MSE", f"{mse_50:.3f}")
    st.metric("R²", f"{r2_50:.3f}")

with metric_col2:
    st.markdown("### 최근 100년 모델")
    st.write("학습: **1906~2005년**")
    st.metric("MAE", f"{mae_100:.3f} ℃")
    st.metric("MSE", f"{mse_100:.3f}")
    st.metric("R²", f"{r2_100:.3f}")

with metric_col3:
    st.markdown("### 평가 기준")
    st.write("테스트: **2006~2025년**")
    st.write("MAE: 낮을수록 좋음")
    st.write("MSE: 낮을수록 좋음")
    st.write("R²: 높을수록 좋음")


# --------------------------------------------------
# 성능 비교표
# --------------------------------------------------
st.subheader("🔍 50년 모델과 100년 모델 성능 비교")

performance = pd.DataFrame({
    "모델": [
        "최근 50년",
        "최근 100년",
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005",
    ],
    "테스트기간": [
        "2006~2025",
        "2006~2025",
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100,
    ],
    "100년당 변화량 (℃)": [
        slope_50 * 100,
        slope_100 * 100,
    ],
    "MAE (℃)": [
        mae_50,
        mae_100,
    ],
    "MSE": [
        mse_50,
        mse_100,
    ],
    "R²": [
        r2_50,
        r2_100,
    ],
})

performance["기울기 (℃/년)"] = (
    performance["기울기 (℃/년)"].round(5)
)

performance["100년당 변화량 (℃)"] = (
    performance["100년당 변화량 (℃)"].round(2)
)

performance["MAE (℃)"] = (
    performance["MAE (℃)"].round(3)
)

performance["MSE"] = (
    performance["MSE"].round(3)
)

performance["R²"] = (
    performance["R²"].round(3)
)

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 어떤 모델이 더 좋은지 자동 판단
# --------------------------------------------------
st.subheader("📝 결과 해석")

if mae_50 < mae_100:
    mae_result = "최근 50년 모델의 MAE가 더 작아 테스트 데이터의 평균적인 예측 오차가 더 작았습니다."
else:
    mae_result = "최근 100년 모델의 MAE가 더 작아 테스트 데이터의 평균적인 예측 오차가 더 작았습니다."

if mse_50 < mse_100:
    mse_result = "최근 50년 모델의 MSE가 더 작아 큰 예측 오차까지 고려했을 때 더 좋은 성능을 보였습니다."
else:
    mse_result = "최근 100년 모델의 MSE가 더 작아 큰 예측 오차까지 고려했을 때 더 좋은 성능을 보였습니다."

if r2_50 > r2_100:
    r2_result = "최근 50년 모델의 R²가 더 높아 실제 기온 변동을 더 잘 설명했습니다."
else:
    r2_result = "최근 100년 모델의 R²가 더 높아 실제 기온 변동을 더 잘 설명했습니다."

st.write(f"**MAE 비교:** {mae_result}")
st.write(f"**MSE 비교:** {mse_result}")
st.write(f"**R² 비교:** {r2_result}")

slope_difference = slope_50 - slope_100

st.write(
    f"**기울기 차이:** 최근 50년 모델과 최근 100년 모델의 "
    f"기울기 차이는 **{slope_difference:.5f} ℃/년**입니다."
)

st.info(
    "해석할 때는 기울기와 예측 성능을 함께 보는 것이 중요합니다. "
    "기울기가 크다는 것은 장기적인 온도 상승 추세를 더 크게 추정한다는 뜻이고, "
    "MAE·MSE·R²는 실제로 미래 구간인 2006~2025년을 얼마나 잘 예측했는지를 보여줍니다."
)


# --------------------------------------------------
# 연도 선택 예측
# --------------------------------------------------
st.subheader("🔮 회귀선을 이용한 미래 연도 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

pred_50 = predict_temperature(
    selected_year,
    slope_50,
    intercept_50,
)

pred_100 = predict_temperature(
    selected_year,
    slope_100,
    intercept_100,
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 최근 50년 모델")
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:25px;
            border-radius:15px;
            background-color:rgba(128,128,128,0.12);
        ">
            <div style="font-size:22px;">
                {selected_year}년 예상 연평균기온
            </div>
            <div style="font-size:50px;font-weight:bold;">
                {pred_50:.2f} ℃
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown("### 최근 100년 모델")
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:25px;
            border-radius:15px;
            background-color:rgba(128,128,128,0.12);
        ">
            <div style="font-size:22px;">
                {selected_year}년 예상 연평균기온
            </div>
            <div style="font-size:50px;font-weight:bold;">
                {pred_100:.2f} ℃
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(
    "주의: 회귀선을 2025년 이후로 연장한 값은 단순한 선형 추세 추정치이며, "
    "실제 미래 기후를 정밀하게 예측하는 기후모형의 결과는 아닙니다."
)
