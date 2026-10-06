
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
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

RECENT_START = 2006
RECENT_END = 2025


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

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

    # 연도별 평균기온 계산
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 해 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    # 평균기온 결측치 제거
    annual = annual.dropna(
        subset=["연평균기온"]
    )

    # 1908년부터 지난 연수
    annual["1908년부터_지난_연수"] = (
        annual["연도"] - BASE_YEAR
    )

    return annual


annual = load_data()


# --------------------------------------------------
# 회귀 함수
# --------------------------------------------------
def calculate_regression(data):

    x = data["1908년부터_지난_연수"].to_numpy()
    y = data["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    return slope, intercept


# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
all_slope, all_intercept = calculate_regression(
    annual
)

annual["전체회귀_예측"] = (
    all_slope
    * annual["1908년부터_지난_연수"]
    + all_intercept
)


# --------------------------------------------------
# 최근 20년 데이터
# 2006~2025
# --------------------------------------------------
recent_20 = annual[
    (annual["연도"] >= RECENT_START)
    & (annual["연도"] <= RECENT_END)
].copy()


recent_slope, recent_intercept = calculate_regression(
    recent_20
)


# --------------------------------------------------
# 상관계수
# --------------------------------------------------
correlation = annual["연도"].corr(
    annual["연평균기온"]
)


# --------------------------------------------------
# 데이터 기간 정보
# --------------------------------------------------
start_year = int(
    annual["연도"].min()
)

end_year = int(
    annual["연도"].max()
)

year_count = len(annual)


# --------------------------------------------------
# 100년당 상승량 계산
# --------------------------------------------------
all_rise_100 = all_slope * 100
recent_rise_100 = recent_slope * 100


# --------------------------------------------------
# 화면 제목
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴보고, "
    "선형 회귀를 이용해 연도별 예상 평균기온을 계산합니다."
)


# --------------------------------------------------
# 데이터 정보
# --------------------------------------------------
st.subheader("📌 회귀 직선에 사용한 데이터")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "사용한 연도 수",
    f"{year_count}개"
)

col2.metric(
    "시작 연도",
    f"{start_year}년"
)

col3.metric(
    "끝 연도",
    f"{end_year}년"
)

col4.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.caption(
    f"2025년 이후 데이터와 "
    f"관측일이 {MIN_OBSERVATION_DAYS}일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 산점도 + 전체 회귀선
# --------------------------------------------------
st.subheader(
    "📈 연도별 평균기온과 회귀 직선"
)

fig = go.Figure()

# 전체 연평균기온
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀선
line_years = np.arange(
    start_year,
    end_year + 1
)

line_x = (
    line_years - BASE_YEAR
)

line_temperature = (
    all_slope * line_x
    + all_intercept
)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperature,
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선 기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="데이터",
)

fig.update_xaxes(
    tickformat="d",
    range=[
        start_year - 2,
        end_year + 2
    ]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 100년에 몇 도 오르는가?
# --------------------------------------------------
st.subheader("🌡️ 기온 상승 속도")

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:35px;
        border-radius:18px;
        background-color:rgba(128,128,128,0.12);
        margin-top:15px;
        margin-bottom:25px;
    ">
        <div style="font-size:24px;">
            전체 기간의 기온 상승 속도
        </div>

        <div style="
            font-size:60px;
            font-weight:bold;
            margin-top:10px;
        ">
            {all_rise_100:+.2f} ℃
        </div>

        <div style="
            font-size:20px;
            margin-top:5px;
        ">
            100년에
            {"상승" if all_rise_100 >= 0 else "하락"}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "전체 기간의 회귀선 기울기를 100배하여 "
    "**100년에 몇 ℃ 변화하는지**로 환산했습니다."
)


# --------------------------------------------------
# 전체 기간 vs 최근 20년 비교
# --------------------------------------------------
st.subheader(
    "🔎 전체 기간 vs 최근 20년 기온 상승 속도"
)

col_all, col_recent = st.columns(2)

with col_all:

    st.markdown(
        "### 🌏 전체 기간"
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:30px;
            border-radius:18px;
            background-color:rgba(128,128,128,0.12);
        ">
            <div style="font-size:20px;">
                {start_year}~{end_year}
            </div>

            <div style="
                font-size:52px;
                font-weight:bold;
                margin-top:10px;
            ">
                {all_rise_100:+.2f} ℃
            </div>

            <div style="font-size:18px;">
                100년에 {"상승" if all_rise_100 >= 0 else "하락"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_recent:

    st.markdown(
        "### 📅 최근 20년"
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:30px;
            border-radius:18px;
            background-color:rgba(128,128,128,0.12);
        ">
            <div style="font-size:20px;">
                {RECENT_START}~{RECENT_END}
            </div>

            <div style="
                font-size:52px;
                font-weight:bold;
                margin-top:10px;
            ">
                {recent_rise_100:+.2f} ℃
            </div>

            <div style="font-size:18px;">
                100년에 {"상승" if recent_rise_100 >= 0 else "하락"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 최근 20년 회귀선과 전체 회귀선 비교 그래프
# --------------------------------------------------
st.subheader(
    "📊 전체 기간과 최근 20년 회귀선 비교"
)

fig_compare = go.Figure()

# 실제 데이터
fig_compare.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=5,
            opacity=0.45
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig_compare.add_trace(
    go.Scatter(
        x=line_years,
        y=(
            all_slope
            * (line_years - BASE_YEAR)
            + all_intercept
        ),
        mode="lines",
        name="전체 기간 회귀선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
recent_years = np.arange(
    RECENT_START,
    RECENT_END + 1
)

fig_compare.add_trace(
    go.Scatter(
        x=recent_years,
        y=(
            recent_slope
            * (recent_years - BASE_YEAR)
            + recent_intercept
        ),
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=4
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig_compare.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
)

fig_compare.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# --------------------------------------------------
# 비교표
# --------------------------------------------------
st.subheader("📋 상승 속도 비교")

comparison = pd.DataFrame({
    "구분": [
        "전체 기간",
        "최근 20년"
    ],
    "기간": [
        f"{start_year}~{end_year}",
        "2006~2025"
    ],
    "기울기 (℃/년)": [
        all_slope,
        recent_slope
    ],
    "100년당 변화량 (℃)": [
        all_rise_100,
        recent_rise_100
    ]
})

comparison["기울기 (℃/년)"] = (
    comparison["기울기 (℃/년)"].round(5)
)

comparison["100년당 변화량 (℃)"] = (
    comparison["100년당 변화량 (℃)"].round(2)
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("🧮 회귀 분석")

st.write(
    f"전체 기간 회귀식: "
    f"**예상 기온 = "
    f"{all_slope:.5f} × (연도 - {BASE_YEAR}) "
    f"{all_intercept:+.3f}**"
)

st.write(
    f"최근 20년 회귀식: "
    f"**예상 기온 = "
    f"{recent_slope:.5f} × (연도 - {BASE_YEAR}) "
    f"{recent_intercept:+.3f}**"
)


# --------------------------------------------------
# 연도 선택 및 예측
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = (
    selected_year - BASE_YEAR
)

predicted_temperature = (
    all_slope * selected_x
    + all_intercept
)


st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:30px;
        border-radius:15px;
        background-color:rgba(128,128,128,0.12);
        margin-top:15px;
        margin-bottom:20px;
    ">
        <div style="font-size:24px;">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size:60px;
            font-weight:bold;
            margin-top:5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# 관측 기간 밖이면 안내
if (
    selected_year < start_year
    or selected_year > end_year
):
    st.warning(
        f"{selected_year}년은 회귀 직선에 사용한 "
        f"관측 기간 ({start_year}~{end_year}년) 밖의 값이므로 "
        "회귀 직선을 연장한 추정값입니다."
    )


st.caption(
    "이 값은 과거 서울 기온의 선형 추세를 단순히 연장한 값이며, "
    "실제 미래 기후를 정밀하게 예측하는 기후모형의 결과는 아닙니다."
)

