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

# 학습 / 테스트 기간
TRAIN_50_START = 1956
TRAIN_50_END = 2005

TRAIN_100_START = 1906
TRAIN_100_END = 2005

TEST_START = 2006
TEST_END = 2025


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜와 평균기온 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜"]
    )

    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= LAST_CLASS_YEAR
    ]

    # --------------------------------------------------
    # 연도별 평균기온 계산
    # --------------------------------------------------
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수가 300일 이상인 연도만 사용
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    # 결측치 제거
    annual = annual.dropna(
        subset=["연평균기온"]
    )

    # 회귀분석용 X
    annual["1908년부터_지난_연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# --------------------------------------------------
# 선형회귀 함수
# --------------------------------------------------
def make_regression(data):

    x = data[
        "1908년부터_지난_연수"
    ].to_numpy()

    y = data[
        "연평균기온"
    ].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


def predict_temperature(
    years,
    slope,
    intercept
):

    x = np.asarray(years) - BASE_YEAR

    return (
        slope * x
        + intercept
    )


# --------------------------------------------------
# 전체 데이터 회귀
# --------------------------------------------------
all_slope, all_intercept = make_regression(
    annual
)

annual["전체회귀_예측"] = predict_temperature(
    annual["연도"],
    all_slope,
    all_intercept
)

correlation = annual["연도"].corr(
    annual["연평균기온"]
)


# --------------------------------------------------
# 데이터 분리
# --------------------------------------------------

# 50년 학습 데이터
train_50 = annual[
    (annual["연도"] >= TRAIN_50_START)
    &
    (annual["연도"] <= TRAIN_50_END)
].copy()

# 100년 학습 데이터
train_100 = annual[
    (annual["연도"] >= TRAIN_100_START)
    &
    (annual["연도"] <= TRAIN_100_END)
].copy()

# 공통 테스트 데이터
test = annual[
    (annual["연도"] >= TEST_START)
    &
    (annual["연도"] <= TEST_END)
].copy()


# --------------------------------------------------
# 50년 모델 학습
# --------------------------------------------------
slope_50, intercept_50 = make_regression(
    train_50
)

test["50년_예측"] = predict_temperature(
    test["연도"],
    slope_50,
    intercept_50
)


# --------------------------------------------------
# 100년 모델 학습
# --------------------------------------------------
slope_100, intercept_100 = make_regression(
    train_100
)

test["100년_예측"] = predict_temperature(
    test["연도"],
    slope_100,
    intercept_100
)


# --------------------------------------------------
# 테스트 성능 평가
# --------------------------------------------------

# 실제값
y_test = test["연평균기온"]

# 50년 모델
mae_50 = mean_absolute_error(
    y_test,
    test["50년_예측"]
)

mse_50 = mean_squared_error(
    y_test,
    test["50년_예측"]
)

r2_50 = r2_score(
    y_test,
    test["50년_예측"]
)

# 100년 모델
mae_100 = mean_absolute_error(
    y_test,
    test["100년_예측"]
)

mse_100 = mean_squared_error(
    y_test,
    test["100년_예측"]
)

r2_100 = r2_score(
    y_test,
    test["100년_예측"]
)


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title(
    "🌡️ 서울 연평균 기온 선형회귀 분석"
)

st.write(
    "서울의 연평균 기온 데이터를 이용하여 "
    "서로 다른 기간으로 선형회귀 모델을 학습하고, "
    "공통 테스트 데이터에서 예측 성능을 비교합니다."
)


# --------------------------------------------------
# 분석 방법
# --------------------------------------------------
st.subheader("📌 분석 방법")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        "1956~2005"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        "1906~2005"
    )

with col3:
    st.metric(
        "공통 테스트",
        "2006~2025"
    )

st.info(
    "두 모델 모두 2006~2025년이라는 동일한 테스트 데이터를 "
    "사용하여 예측 성능을 공정하게 비교합니다."
)


# --------------------------------------------------
# 전체 데이터 요약통계
# --------------------------------------------------
st.subheader(
    "📊 전체 연평균 기온 데이터 요약통계"
)

summary = pd.DataFrame({
    "통계량": [
        "개수",
        "평균",
        "최소",
        "최대"
    ],
    "값": [
        f"{len(annual)}개",
        f"{annual['연평균기온'].mean():.2f} ℃",
        f"{annual['연평균기온'].min():.2f} ℃",
        f"{annual['연평균기온'].max():.2f} ℃"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"연평균 기온은 관측일수가 "
    f"{MIN_OBSERVATION_DAYS}일 이상인 연도만 사용했습니다."
)


# --------------------------------------------------
# 전체 데이터 회귀선
# --------------------------------------------------
st.subheader(
    "📈 전체 데이터의 선형회귀"
)

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온"
    )
)

line_years = np.arange(
    annual["연도"].min(),
    annual["연도"].max() + 1
)

line_temp = predict_temperature(
    line_years,
    all_slope,
    all_intercept
)

fig_all.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temp,
        mode="lines",
        name="전체 데이터 회귀선"
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_all,
    use_container_width=True
)

st.write(
    f"전체 데이터 회귀선 기울기: "
    f"**{all_slope:.5f} ℃/년**"
)

st.write(
    f"100년당 기온 변화: "
    f"**{all_slope * 100:.2f} ℃/100년**"
)

st.write(
    f"전체 데이터 상관계수: "
    f"**{correlation:.4f}**"
)


# --------------------------------------------------
# 50년 / 100년 회귀선 비교
# --------------------------------------------------
st.subheader(
    "📐 최근 50년과 최근 100년 회귀선 비교"
)

fig_train = go.Figure()

# 실제 데이터
fig_train.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=5,
            opacity=0.5
        )
    )
)

# 100년 회귀선
years_100 = np.arange(
    TRAIN_100_START,
    TRAIN_100_END + 1
)

fig_train.add_trace(
    go.Scatter(
        x=years_100,
        y=predict_temperature(
            years_100,
            slope_100,
            intercept_100
        ),
        mode="lines",
        name="1906~2005 회귀선"
    )
)

# 50년 회귀선
years_50 = np.arange(
    TRAIN_50_START,
    TRAIN_50_END + 1
)

fig_train.add_trace(
    go.Scatter(
        x=years_50,
        y=predict_temperature(
            years_50,
            slope_50,
            intercept_50
        ),
        mode="lines",
        name="1956~2005 회귀선"
    )
)

fig_train.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_train,
    use_container_width=True
)


# --------------------------------------------------
# 기울기 비교
# --------------------------------------------------
st.subheader(
    "📈 회귀선 기울기 비교"
)

slope_table = pd.DataFrame({
    "모델": [
        "최근 50년",
        "최근 100년"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 변화량 (℃)": [
        slope_50 * 100,
        slope_100 * 100
    ]
})

slope_table[
    "기울기 (℃/년)"
] = slope_table[
    "기울기 (℃/년)"
].round(5)

slope_table[
    "100년당 변화량 (℃)"
] = slope_table[
    "100년당 변화량 (℃)"
].round(2)

st.dataframe(
    slope_table,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 테스트 데이터 실제값 vs 예측값
# --------------------------------------------------
st.subheader(
    "🎯 2006~2025년 실제값과 예측값"
)

fig_test = go.Figure()

# 실제값
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)

# 50년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["50년_예측"],
        mode="lines",
        name="50년 모델 예측"
    )
)

# 100년 모델
fig_test.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["100년_예측"],
        mode="lines",
        name="100년 모델 예측"
    )
)

fig_test.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_test,
    use_container_width=True
)


# --------------------------------------------------
# 예측 성능 평가
# --------------------------------------------------
st.subheader(
    "📊 테스트 데이터 예측 성능 평가"
)

st.write(
    "테스트 데이터는 학습에 사용하지 않은 "
    "**2006~2025년**입니다."
)

col50, col100 = st.columns(2)

with col50:

    st.markdown(
        "### 🔵 최근 50년 모델"
    )

    st.write(
        "학습 데이터: **1956~2005년**"
    )

    st.metric(
        "MAE",
        f"{mae_50:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_50:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_50:.3f}"
    )


with col100:

    st.markdown(
        "### 🟠 최근 100년 모델"
    )

    st.write(
        "학습 데이터: **1906~2005년**"
    )

    st.metric(
        "MAE",
        f"{mae_100:.3f} ℃"
    )

    st.metric(
        "MSE",
        f"{mse_100:.3f}"
    )

    st.metric(
        "R²",
        f"{r2_100:.3f}"
    )


# --------------------------------------------------
# 최종 비교표
# --------------------------------------------------
st.subheader(
    "🔎 최근 50년 vs 최근 100년 최종 비교"
)

comparison = pd.DataFrame({
    "모델": [
        "최근 50년",
        "최근 100년"
    ],
    "학습기간": [
        "1956~2005",
        "1906~2005"
    ],
    "테스트기간": [
        "2006~2025",
        "2006~2025"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "100년당 상승량 (℃)": [
        slope_50 * 100,
        slope_100 * 100
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

comparison[
    "기울기 (℃/년)"
] = comparison[
    "기울기 (℃/년)"
].round(5)

comparison[
    "100년당 상승량 (℃)"
] = comparison[
    "100년당 상승량 (℃)"
].round(2)

comparison[
    "MAE (℃)"
] = comparison[
    "MAE (℃)"
].round(3)

comparison[
    "MSE"
] = comparison[
    "MSE"
].round(3)

comparison[
    "R²"
] = comparison[
    "R²"
].round(3)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 자동 결과 해석
# --------------------------------------------------
st.subheader(
    "📝 결과 해석"
)

if mae_50 < mae_100:
    mae_best = "최근 50년 모델"
else:
    mae_best = "최근 100년 모델"

if mse_50 < mse_100:
    mse_best = "최근 50년 모델"
else:
    mse_best = "최근 100년 모델"

if r2_50 > r2_100:
    r2_best = "최근 50년 모델"
else:
    r2_best = "최근 100년 모델"

st.write(
    f"• **MAE:** {mae_best}의 평균적인 예측 오차가 더 작습니다."
)

st.write(
    f"• **MSE:** {mse_best}의 큰 오차를 포함한 전체적인 예측 성능이 더 좋습니다."
)

st.write(
    f"• **R²:** {r2_best}이 테스트 데이터의 기온 변동을 더 잘 설명합니다."
)

st.write(
    f"• **기울기:** 최근 50년 모델은 "
    f"**{slope_50:.5f} ℃/년**, "
    f"최근 100년 모델은 "
    f"**{slope_100:.5f} ℃/년**입니다."
)

st.write(
    f"즉, 100년 기준으로 보면 최근 50년 모델은 "
    f"**{slope_50 * 100:.2f} ℃**, "
    f"최근 100년 모델은 "
    f"**{slope_100 * 100:.2f} ℃**의 변화 추세를 나타냅니다."
)


# --------------------------------------------------
# 미래 연도 예측
# --------------------------------------------------
st.subheader(
    "🔮 회귀선을 이용한 연도별 예상 기온"
)

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

prediction_50 = predict_temperature(
    selected_year,
    slope_50,
    intercept_50
)

prediction_100 = predict_temperature(
    selected_year,
    slope_100,
    intercept_100
)

col_a, col_b = st.columns(2)

with col_a:

    st.markdown(
        "### 🔵 최근 50년 모델"
    )

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
            <div style="
                font-size:50px;
                font-weight:bold;
            ">
                {prediction_50:.2f} ℃
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col_b:

    st.markdown(
        "### 🟠 최근 100년 모델"
    )

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
            <div style="
                font-size:50px;
                font-weight:bold;
            ">
                {prediction_100:.2f} ℃
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.caption(
    "회귀선을 미래로 연장한 값은 단순한 선형 추세 추정값이며, "
    "실제 미래 기후를 정밀하게 예측하는 기후모형의 결과는 아닙니다."
)
