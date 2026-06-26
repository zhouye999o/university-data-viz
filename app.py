import streamlit as st
st.set_page_config(page_title="数说高校：中国本科院校数据可视化分析", page_icon="🎓", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score

# ==================== 配色方案 ====================

COLOR_PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
                 '#6A994E', '#BC4B51', '#5D737E', '#F4A261', '#9B5DE5', '#00B4D8']

PIE_PALETTE = ['#89CFF0', '#F4A261', '#6A994E', '#E76F51', '#A23B72',
               '#2E86AB', '#BC4B51', '#9B5DE5', '#3B1F2B', '#5D737E', '#F18F01']

BLUES_SCALE = 'Blues'
RDBU_SCALE = 'RdBu_r'

# ==================== 自定义CSS样式 ====================

st.markdown("""
<style>
    /* 全局背景 - 浅灰蓝色调，温和不刺眼 */
    .stApp {
        background: linear-gradient(160deg, #e8ecf1 0%, #dfe4ed 50%, #e2e6ee 100%);
        background-attachment: fixed;
    }
    
    /* 主内容区域 - 白色卡片，清晰突出 */
    .main .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* 侧边栏 - 柔和深灰蓝 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    
    /* Streamlit默认文字颜色确保可读 */
    .main .block-container p,
    .main .block-container span,
    .main .block-container label {
        color: #2c3e50;
    }
    
    /* 页面标题 */
    h1 {
        color: #1a252f !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }
    h2 {
        color: #2E86AB !important;
        font-weight: 600 !important;
    }
    h3 {
        color: #34495e !important;
        font-weight: 600 !important;
    }
    
    /* 信息卡片 */
    .insight-card {
        background: linear-gradient(135deg, #f0f8ff 0%, #e8f4fd 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(46, 134, 171, 0.08);
        color: #2c3e50;
    }
    .insight-card-warning {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #F18F01;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(241, 143, 1, 0.08);
        color: #2c3e50;
    }
    .insight-card-success {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #6A994E;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(106, 153, 78, 0.08);
        color: #2c3e50;
    }
    .insight-card-pink {
        background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #A23B72;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(162, 59, 114, 0.08);
        color: #2c3e50;
    }
    .cluster-desc {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 12px 15px;
        border-radius: 8px;
        border-left: 4px solid #9B5DE5;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(155, 93, 229, 0.08);
        color: #2c3e50;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #dde3ec 100%);
        padding: 10px 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px 0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    }
    
    /* 图表容器 */
    .js-plotly-plot, .plot-container {
        border-radius: 10px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* 数据表格颜色 */
    .stDataFrame {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 数据加载 ====================

@st.cache_data
def load_data():
    uni = pd.read_csv("universities.csv")
    prov = pd.read_csv("province_data.csv")
    merged = uni.merge(prov, on="省份", how="left", suffixes=("", "_prov"))
    return uni, prov, merged

uni_df, prov_df, merged_df = load_data()

# ==================== 中国地图GeoJSON ====================

PROVINCE_ADCODE = {
    "北京": 110000, "天津": 120000, "河北": 130000, "山西": 140000, "内蒙古": 150000,
    "辽宁": 210000, "吉林": 220000, "黑龙江": 230000, "上海": 310000, "江苏": 320000,
    "浙江": 330000, "安徽": 340000, "福建": 350000, "江西": 360000, "山东": 370000,
    "河南": 410000, "湖北": 420000, "湖南": 430000, "广东": 440000, "广西": 450000,
    "海南": 460000, "重庆": 500000, "四川": 510000, "贵州": 520000, "云南": 530000,
    "西藏": 540000, "陕西": 610000, "甘肃": 620000, "青海": 630000, "宁夏": 640000,
    "新疆": 650000,
}

@st.cache_data
def load_china_geojson():
    """加载本地中国省份GeoJSON数据"""
    import json
    import os
    geo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "china_geojson.json")
    with open(geo_path, "r", encoding="utf-8") as f:
        return json.load(f)

china_geojson = load_china_geojson()

# ==================== 洞察辅助函数 ====================

def get_top_province_info():
    """获取高校数量最多的省份信息"""
    top = uni_df["省份"].value_counts()
    top_name = top.index[0]
    top_count = top.iloc[0]
    second_name = top.index[1] if len(top) > 1 else ""
    second_count = top.iloc[1] if len(top) > 1 else 0
    return top_name, top_count, second_name, second_count

def get_type_stats():
    """获取院校类型统计"""
    type_counts = uni_df["院校类型"].value_counts()
    total = len(uni_df)
    top_type = type_counts.index[0]
    top_pct = type_counts.iloc[0] / total * 100
    second_type = type_counts.index[1] if len(type_counts) > 1 else ""
    second_pct = type_counts.iloc[1] / total * 100 if len(type_counts) > 1 else 0
    return top_type, top_pct, second_type, second_pct, type_counts

def get_level_stats():
    """获取学校层次统计"""
    total_985 = uni_df["是否985"].sum()
    total_211 = uni_df["是否211"].sum()
    total_sl = uni_df["是否双一流"].sum()
    return int(total_985), int(total_211), int(total_sl)

def get_region_stats():
    """获取区域高校占比"""
    region_counts = uni_df["所属区域"].value_counts()
    total = len(uni_df)
    top_region = region_counts.index[0]
    top_region_pct = region_counts.iloc[0] / total * 100
    return top_region, top_region_pct, region_counts

def get_city_top10():
    """获取Top10城市高校数量"""
    city_counts = uni_df["城市"].value_counts().head(10)
    return city_counts

def get_decade_stats():
    """获取按年代分组的办学历史统计"""
    uni_decade = uni_df.copy()
    uni_decade["年代"] = (uni_decade["创办年份"] // 10) * 10
    uni_decade = uni_decade[uni_decade["创办年份"] > 0]
    decade_counts = uni_decade["年代"].value_counts().sort_index()
    return decade_counts

def get_correlation_text():
    """获取相关性分析文本"""
    corr_gdp_vs_uni = prov_df["GDP(亿元)"].corr(prov_df["高校数量"])
    corr_pop_vs_uni = prov_df["人口(万人)"].corr(prov_df["高校数量"])
    corr_pgdp_vs_sl = prov_df["人均GDP(元)"].corr(prov_df["双一流数量"])
    return corr_gdp_vs_uni, corr_pop_vs_uni, corr_pgdp_vs_sl

def get_gdp_rank_comparison():
    """获取GDP排名与高校数量排名对比"""
    df = prov_df.copy()
    df["GDP排名"] = df["GDP(亿元)"].rank(ascending=False)
    df["高校排名"] = df["高校数量"].rank(ascending=False)
    df["排名差"] = df["GDP排名"] - df["高校排名"]
    df = df[["省份", "GDP(亿元)", "GDP排名", "高校数量", "高校排名", "排名差"]].sort_values("排名差")
    return df

# 预计算常用统计
top_province, top_count, second_province, second_count = get_top_province_info()
top_type, top_pct, second_type, second_pct, type_counts = get_type_stats()
total_985, total_211, total_sl = get_level_stats()
top_region, top_region_pct, region_counts = get_region_stats()
city_top10 = get_city_top10()
decade_counts = get_decade_stats()
corr_gdp_uni, corr_pop_uni, corr_pgdp_sl = get_correlation_text()
total_uni = len(uni_df)

# ==================== 侧边栏导航 ====================

st.sidebar.title("🎓 数说高校")
st.sidebar.markdown("中国本科院校数据可视化分析平台")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航菜单",
    ["📋 数据浏览", "📊 描述性统计分析", "🔗 关联分析", "🧩 聚类分析", "🔮 预测分析"]
)

# ==================== 页面1：数据浏览 ====================

if page == "📋 数据浏览":
    st.title("📋 数据浏览")
    st.markdown("浏览和筛选中国本科院校数据，支持多维度筛选与关键词搜索。")

    # 顶部指标卡片
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><h3 style="color:#2E86AB;margin:0;">{total_uni}</h3><small>本科院校总数</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3 style="color:#A23B72;margin:0;">{total_985}</h3><small>985工程院校</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3 style="color:#F18F01;margin:0;">{total_211}</h3><small>211工程院校</small></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3 style="color:#6A994E;margin:0;">{total_sl}</h3><small>双一流院校</small></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><h3 style="color:#9B5DE5;margin:0;">{uni_df["省份"].nunique()}</h3><small>覆盖省份</small></div>', unsafe_allow_html=True)

    st.markdown("---")

    # 侧边栏筛选器
    st.sidebar.markdown("### 🔍 筛选条件")

    provinces = sorted(uni_df["省份"].unique().tolist())
    selected_provinces = st.sidebar.multiselect("省份", provinces, default=[])

    types = sorted(uni_df["院校类型"].unique().tolist())
    selected_types = st.sidebar.multiselect("院校类型", types, default=[])

    level_options = ["985工程", "211工程", "双一流", "普通本科"]
    selected_levels = st.sidebar.multiselect("学校层次", level_options, default=[])

    regions = sorted(uni_df["所属区域"].unique().tolist())
    selected_regions = st.sidebar.multiselect("所属区域", regions, default=[])

    keyword = st.sidebar.text_input("🔎 关键词搜索", placeholder="输入院校名称...")

    # 筛选数据
    filtered = uni_df.copy()

    if selected_provinces:
        filtered = filtered[filtered["省份"].isin(selected_provinces)]
    if selected_types:
        filtered = filtered[filtered["院校类型"].isin(selected_types)]
    if selected_levels:
        filtered = filtered[filtered["学校层次标签"].isin(selected_levels)]
    if selected_regions:
        filtered = filtered[filtered["所属区域"].isin(selected_regions)]
    if keyword:
        filtered = filtered[filtered["院校名称"].str.contains(keyword, case=False)]

    # 显示筛选统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("筛选结果", f"{len(filtered)} 所高校")
    with col2:
        st.metric("覆盖省份", f"{filtered['省份'].nunique()} 个")
    with col3:
        st.metric("覆盖城市", f"{filtered['城市'].nunique()} 个")

    st.markdown("---")
    st.dataframe(
        filtered,
        column_config={
            "院校名称": "院校名称",
            "省份": "省份",
            "城市": "城市",
            "院校类型": "院校类型",
            "办学层次": "办学层次",
            "主管部门": "主管部门",
            "创办年份": "创办年份",
            "是否双一流": "双一流",
            "是否985": "985",
            "是否211": "211",
            "所属区域": "所属区域",
            "办学年限": "办学年限",
            "学校层次标签": "学校层次",
        },
        use_container_width=True,
        hide_index=True,
    )

# ==================== 页面2：描述性统计分析 ====================

elif page == "📊 描述性统计分析":
    st.title("📊 描述性统计分析")
    st.markdown("从多个维度展示中国本科院校的分布特征。")

    # ---- 顶部指标卡片 ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🏫 总高校数", f"{total_uni} 所", delta=None)
    with c2:
        st.metric("🏅 985院校", f"{total_985} 所", delta=None)
    with c3:
        st.metric("🎖️ 211院校", f"{total_211} 所", delta=None)
    with c4:
        st.metric("⭐ 双一流院校", f"{total_sl} 所", delta=None)

    st.markdown("---")

    # 2.1 各省高校数量柱状图
    st.subheader("2.1 各省高校数量分布")
    province_count = uni_df["省份"].value_counts().reset_index()
    province_count.columns = ["省份", "高校数量"]
    province_count = province_count.merge(
        uni_df[["省份", "所属区域"]].drop_duplicates("省份"), on="省份", how="left"
    )
    province_count = province_count.sort_values("高校数量", ascending=True)

    fig1 = px.bar(
        province_count,
        x="高校数量",
        y="省份",
        color="所属区域",
        orientation="h",
        title="各省本科院校数量（按数量降序）",
        color_discrete_sequence=COLOR_PALETTE,
        text="高校数量",
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(height=700, xaxis_title="高校数量", yaxis_title="省份")

    col_chart, col_insight = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(fig1, use_container_width=True)
    with col_insight:
        # 计算华东区域占比
        east_provinces = province_count[province_count["所属区域"] == "华东"]
        east_total = east_provinces["高校数量"].sum() if len(east_provinces) > 0 else 0
        east_pct = east_total / total_uni * 100
        # 西部省份
        west_regions = ["西北", "西南"]
        west_total = province_count[province_count["所属区域"].isin(west_regions)]["高校数量"].sum()
        west_pct = west_total / total_uni * 100
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：各省高校分布</strong><br>
        <strong>{top_province}</strong>以 <strong>{top_count}</strong> 所本科院校位居全国第一，
        <strong>{second_province}</strong>（{second_count} 所）紧随其后。
        华东地区高校资源最为丰富，占全国总量的 <strong>{east_pct:.1f}%</strong>。
        西藏、青海、宁夏等西部省份高校资源相对匮乏，西北和西南地区合计仅占 <strong>{west_pct:.1f}%</strong>。
        </div>
        """, unsafe_allow_html=True)

    # 2.2 Top10城市高校数量
    st.subheader("2.2 Top10城市高校数量")
    city_top10_df = city_top10.reset_index()
    city_top10_df.columns = ["城市", "高校数量"]

    col_chart2, col_insight2 = st.columns([3, 1])
    with col_chart2:
        fig_city = px.bar(
            city_top10_df.sort_values("高校数量", ascending=True),
            x="高校数量",
            y="城市",
            orientation="h",
            title="Top10 城市本科院校数量",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            text="高校数量",
        )
        fig_city.update_traces(textposition="outside")
        fig_city.update_layout(height=450, xaxis_title="高校数量", yaxis_title="城市")
        st.plotly_chart(fig_city, use_container_width=True)
    with col_insight2:
        top_city = city_top10_df.iloc[0]
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：城市高校分布</strong><br>
        <strong>{top_city['城市']}</strong>以 <strong>{top_city['高校数量']}</strong> 所高校位居首位，
        远超其他城市。Top10城市合计拥有 <strong>{city_top10_df['高校数量'].sum()}</strong> 所高校，
        占全国总量的 <strong>{city_top10_df['高校数量'].sum()/total_uni*100:.1f}%</strong>，
        高校资源高度集中于一二线城市。
        </div>
        """, unsafe_allow_html=True)

    # 2.3 院校类型分布饼图
    st.subheader("2.3 院校类型分布")
    col_left, col_right = st.columns(2)

    with col_left:
        type_count = uni_df["院校类型"].value_counts().reset_index()
        type_count.columns = ["院校类型", "数量"]
        fig2 = px.pie(
            type_count,
            values="数量",
            names="院校类型",
            title="院校类型分布",
            color_discrete_sequence=PIE_PALETTE,
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        st.markdown("### ")
        st.markdown("### ")
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：院校类型</strong><br>
        <strong>{top_type}</strong>类院校占比最高（<strong>{top_pct:.1f}%</strong>），
        体现了我国工业化发展对工程技术人才的大量需求。
        <strong>{second_type}</strong>类大学占比第二（<strong>{second_pct:.1f}%</strong>），
        反映了高等教育向综合化方向发展的趋势。师范类院校也占有一定比例，体现了国家对基础教育的重视。
        </div>
        """, unsafe_allow_html=True)

    # 2.4 学校层次分布饼图
    st.subheader("2.4 学校层次分布")
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        level_count = uni_df["学校层次标签"].value_counts().reset_index()
        level_count.columns = ["学校层次", "数量"]
        level_order = ["985工程", "211工程", "双一流", "普通本科"]
        level_count["学校层次"] = pd.Categorical(level_count["学校层次"], categories=level_order, ordered=True)
        level_count = level_count.sort_values("学校层次")
        fig3 = px.pie(
            level_count,
            values="数量",
            names="学校层次",
            title="学校层次分布",
            color_discrete_sequence=PIE_PALETTE[:4],
        )
        fig3.update_traces(textposition="inside", textinfo="percent+label")
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        st.markdown("### ")
        st.markdown("### ")
        normal_count = len(uni_df[uni_df["学校层次标签"] == "普通本科"])
        normal_pct = normal_count / total_uni * 100
        st.markdown(f"""
        <div class="insight-card-warning">
        <strong>📌 数据洞察：学校层次</strong><br>
        985工程院校仅 <strong>{total_985}</strong> 所，211工程院校 <strong>{total_211}</strong> 所，
        双一流院校 <strong>{total_sl}</strong> 所。其中985院校全部属于211和双一流，
        优质高等教育资源高度集中。普通本科院校占 <strong>{normal_pct:.1f}%</strong>，
        是高等教育的主体力量。
        </div>
        """, unsafe_allow_html=True)

    # 2.5 高校办学历史时间线
    st.subheader("2.5 高校办学历史时间线")
    decade_df = decade_counts.reset_index()
    decade_df.columns = ["年代", "高校数量"]
    decade_df = decade_df.sort_values("年代")

    col_chart3, col_insight3 = st.columns([3, 1])
    with col_chart3:
        fig_decade = px.bar(
            decade_df,
            x="年代",
            y="高校数量",
            title="各年代创办高校数量",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            text="高校数量",
        )
        fig_decade.update_traces(textposition="outside")
        fig_decade.update_layout(height=450, xaxis_title="创办年代", yaxis_title="高校数量")
        fig_decade.update_xaxes(type='category')
        st.plotly_chart(fig_decade, use_container_width=True)
    with col_insight3:
        max_decade = decade_df.loc[decade_df["高校数量"].idxmax()]
        oldest = uni_df[uni_df["创办年份"] > 0]["创办年份"].min()
        oldest_name = uni_df[uni_df["创办年份"] == oldest]["院校名称"].iloc[0] if oldest > 0 else "未知"
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：办学历史</strong><br>
        高校创办高峰出现在 <strong>{int(max_decade['年代'])}</strong> 年代（{int(max_decade['高校数量'])} 所），
        反映了新中国高等教育的大规模建设时期。最早的高校可追溯至 <strong>{int(oldest)}</strong> 年
        （{oldest_name}），办学历史逾百年。
        </div>
        """, unsafe_allow_html=True)

    # 2.6 各区域高校数量与类型堆叠柱状图
    st.subheader("2.6 各区域高校数量与类型分布")
    region_type = uni_df.groupby(["所属区域", "院校类型"]).size().reset_index(name="数量")
    region_order = region_type.groupby("所属区域")["数量"].sum().sort_values(ascending=False).index.tolist()

    col_chart5, col_insight5 = st.columns([3, 1])
    with col_chart5:
        fig5 = px.bar(
            region_type,
            x="所属区域",
            y="数量",
            color="院校类型",
            title="各区域高校数量与类型分布",
            color_discrete_sequence=COLOR_PALETTE,
            category_orders={"所属区域": region_order},
        )
        fig5.update_layout(height=450, xaxis_title="区域", yaxis_title="高校数量")
        st.plotly_chart(fig5, use_container_width=True)
    with col_insight5:
        region_total = region_type.groupby("所属区域")["数量"].sum().sort_values(ascending=False)
        top_region_name = region_total.index[0]
        top_region_total = region_total.iloc[0]
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：区域分布</strong><br>
        <strong>{top_region_name}</strong>地区高校数量最多（<strong>{top_region_total}</strong> 所），
        占全国总量的 <strong>{top_region_total/total_uni*100:.1f}%</strong>。
        各区域高校类型分布差异明显，理工类院校在各区域均占主导地位。
        区域间高校资源的差距反映了我国经济发展的空间不平衡性。
        </div>
        """, unsafe_allow_html=True)

# ==================== 页面3：关联分析 ====================

elif page == "🔗 关联分析":
    st.title("🔗 关联分析")
    st.markdown("探索各省经济指标与高等教育资源之间的关联关系。")

    # 3.1 各省GDP与高校数量散点图
    st.subheader("3.1 各省GDP与高校数量")
    col_chart_a, col_insight_a = st.columns([3, 1])
    with col_chart_a:
        fig6 = px.scatter(
            prov_df,
            x="GDP(亿元)",
            y="高校数量",
            size="人口(万人)",
            color="区域",
            hover_name="省份",
            title="各省GDP与高校数量关系（气泡大小=人口）",
            color_discrete_sequence=COLOR_PALETTE,
            size_max=45,
        )
        fig6.update_layout(height=500, xaxis_title="GDP（亿元）", yaxis_title="高校数量")
        fig6.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        st.plotly_chart(fig6, use_container_width=True)
    with col_insight_a:
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：GDP vs 高校数量</strong><br>
        GDP与高校数量的<strong>Pearson相关系数为 {corr_gdp_uni:.2f}</strong>，呈强正相关。
        这表明经济发展水平是影响高校资源分布的重要因素。
        值得注意的是，北京、上海等直辖市虽然GDP总量不是最高，
        但高校密度远超全国平均水平。
        </div>
        """, unsafe_allow_html=True)

    # 3.2 各省人均GDP与双一流数量散点图
    st.subheader("3.2 各省人均GDP与双一流数量")
    col_chart_b, col_insight_b = st.columns([3, 1])
    with col_chart_b:
        fig7 = px.scatter(
            prov_df,
            x="人均GDP(元)",
            y="双一流数量",
            color="区域",
            hover_name="省份",
            title="各省人均GDP与双一流高校数量（含回归趋势线）",
            color_discrete_sequence=COLOR_PALETTE,
            trendline="ols",
            trendline_scope="overall",
        )
        fig7.update_layout(height=500, xaxis_title="人均GDP（元）", yaxis_title="双一流高校数量")
        st.plotly_chart(fig7, use_container_width=True)
    with col_insight_b:
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：人均GDP vs 双一流</strong><br>
        人均GDP与双一流数量的相关系数为 <strong>{corr_pgdp_sl:.2f}</strong>，
        经济发达地区更倾向于拥有高质量高校。
        北京的人均GDP（{prov_df[prov_df['省份']=='北京']['人均GDP(元)'].values[0]:.0f}元）和
        双一流数量（{prov_df[prov_df['省份']=='北京']['双一流数量'].values[0]}所）均远超其他省份，
        在图中呈现明显的离群特征。
        </div>
        """, unsafe_allow_html=True)

    # 3.3 各省高校数量排名与GDP排名对比
    st.subheader("3.3 各省高校数量排名与GDP排名对比")
    gdp_rank_df = get_gdp_rank_comparison()

    col_chart_c, col_insight_c = st.columns([3, 1])
    with col_chart_c:
        fig_rank = go.Figure()
        fig_rank.add_trace(go.Bar(
            y=gdp_rank_df["省份"],
            x=gdp_rank_df["GDP排名"],
            name="GDP排名",
            orientation="h",
            marker_color=COLOR_PALETTE[0],
            text=gdp_rank_df["GDP排名"].astype(int),
            textposition="outside",
        ))
        fig_rank.add_trace(go.Bar(
            y=gdp_rank_df["省份"],
            x=gdp_rank_df["高校排名"],
            name="高校排名",
            orientation="h",
            marker_color=COLOR_PALETTE[1],
            text=gdp_rank_df["高校排名"].astype(int),
            textposition="outside",
        ))
        fig_rank.update_layout(
            title="各省GDP排名 vs 高校数量排名（排名越小越靠前）",
            height=700,
            barmode="group",
            xaxis_title="排名",
            yaxis_title="省份",
            xaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_rank, use_container_width=True)
    with col_insight_c:
        over_ranked = gdp_rank_df.nsmallest(3, "排名差")
        under_ranked = gdp_rank_df.nlargest(3, "排名差")
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：排名对比</strong><br>
        高校排名远超GDP排名的省份（教育强省）：<br>
        {over_ranked[['省份', '排名差']].to_string(index=False, header=False)}<br><br>
        高校排名远低于GDP排名的省份（经济强省）：<br>
        {under_ranked[['省份', '排名差']].to_string(index=False, header=False)}<br><br>
        排名差为正表示GDP排名落后于高校排名（教育领先经济），
        为负表示GDP排名领先于高校排名（经济领先教育）。
        </div>
        """, unsafe_allow_html=True)

    # 3.4 相关性热力图
    st.subheader("3.4 相关性热力图")
    corr_cols = ["GDP(亿元)", "人口(万人)", "人均GDP(元)", "高校数量", "双一流数量"]
    corr_df = prov_df[corr_cols].corr()

    col_chart_d, col_insight_d = st.columns([3, 1])
    with col_chart_d:
        fig8 = go.Figure(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale=RDBU_SCALE,
                zmin=-1,
                zmax=1,
                text=np.round(corr_df.values, 2),
                texttemplate="%{text}",
                textfont={"size": 14},
            )
        )
        fig8.update_layout(
            title="经济指标与高校数量相关系数矩阵",
            height=450,
            width=600,
        )
        st.plotly_chart(fig8, use_container_width=True)
    with col_insight_d:
        max_corr_pair = ""
        max_corr_val = 0
        for i in range(len(corr_cols)):
            for j in range(i+1, len(corr_cols)):
                if abs(corr_df.iloc[i, j]) > max_corr_val:
                    max_corr_val = abs(corr_df.iloc[i, j])
                    max_corr_pair = f"{corr_cols[i]} ↔ {corr_cols[j]}"
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：相关矩阵</strong><br>
        GDP与高校数量的相关系数最高（<strong>{corr_gdp_uni:.2f}</strong>），
        其次是人口与高校数量（<strong>{corr_pop_uni:.2f}</strong>）。
        人均GDP与双一流数量的相关性（<strong>{corr_pgdp_sl:.2f}</strong>）
        表明经济发达地区更倾向于拥有高质量高校。
        经济指标与高校指标之间存在普遍的强正相关关系。
        </div>
        """, unsafe_allow_html=True)

    # 3.5 高校办学年限分布直方图
    st.subheader("3.5 高校办学年限分布")
    col_chart_e, col_insight_e = st.columns([3, 1])
    with col_chart_e:
        fig9 = px.histogram(
            uni_df,
            x="办学年限",
            nbins=40,
            title="中国本科院校办学年限分布",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            marginal="box",
        )
        fig9.update_layout(height=450, xaxis_title="办学年限（年）", yaxis_title="高校数量")
        st.plotly_chart(fig9, use_container_width=True)
    with col_insight_e:
        avg_age = uni_df["办学年限"].mean()
        median_age = uni_df["办学年限"].median()
        max_age = uni_df["办学年限"].max()
        max_age_name = uni_df.loc[uni_df["办学年限"].idxmax(), "院校名称"]
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 数据洞察：办学年限</strong><br>
        全国本科院校平均办学年限 <strong>{avg_age:.0f}</strong> 年，
        中位数为 <strong>{median_age:.0f}</strong> 年。
        办学年限最长的是 <strong>{max_age_name}</strong>（{int(max_age)} 年）。
        办学年限分布呈右偏态，大部分高校集中在40-80年区间，
        反映了新中国成立后高等教育的大规模建设。
        </div>
        """, unsafe_allow_html=True)

# ==================== 页面4：聚类分析 ====================

elif page == "🧩 聚类分析":
    st.title("🧩 聚类分析")
    st.markdown("使用K-Means聚类算法对各省份进行聚类，探索不同省份的教育经济特征。")

    # 用户选择聚类数
    k = st.sidebar.slider("选择聚类数 K", min_value=2, max_value=6, value=4, step=1)

    # 准备聚类数据
    cluster_features = ["GDP(亿元)", "人口(万人)", "高校数量", "双一流数量"]
    X = prov_df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means聚类
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    prov_df_cluster = prov_df.copy()
    prov_df_cluster["聚类类别"] = kmeans.fit_predict(X_scaled)
    prov_df_cluster["聚类类别"] = prov_df_cluster["聚类类别"].astype(str)

    # 计算轮廓系数
    sil_score = silhouette_score(X_scaled, kmeans.labels_)

    # PCA降维
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    prov_df_cluster["PCA1"] = X_pca[:, 0]
    prov_df_cluster["PCA2"] = X_pca[:, 1]

    # 显示轮廓系数
    st.metric("轮廓系数 (Silhouette Score)", f"{sil_score:.4f}",
              delta=None, help="轮廓系数范围[-1, 1]，越接近1表示聚类效果越好")

    # 4.1 聚类结果散点图
    st.subheader("4.1 聚类结果散点图（PCA降维）")
    col_cluster_a, col_cluster_b = st.columns([3, 1])
    with col_cluster_a:
        fig10 = px.scatter(
            prov_df_cluster,
            x="PCA1",
            y="PCA2",
            color="聚类类别",
            hover_name="省份",
            title=f"K-Means 聚类结果（K={k}，PCA 2D可视化，轮廓系数={sil_score:.3f}）",
            color_discrete_sequence=COLOR_PALETTE,
            size=[15] * len(prov_df_cluster),
        )
        fig10.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        fig10.update_layout(height=500, xaxis_title="第一主成分", yaxis_title="第二主成分")
        st.plotly_chart(fig10, use_container_width=True)
    with col_cluster_b:
        pca_var = pca.explained_variance_ratio_
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 PCA解释</strong><br>
        PCA降维将4个特征压缩为2个主成分：
        <br>PC1解释方差 <strong>{pca_var[0]*100:.1f}%</strong>
        <br>PC2解释方差 <strong>{pca_var[1]*100:.1f}%</strong>
        <br>合计解释 <strong>{(pca_var[0]+pca_var[1])*100:.1f}%</strong>
        <br><br>轮廓系数 <strong>{sil_score:.3f}</strong>，
        聚类结构较为合理，各省在经济和高等教育特征上呈现出明显的分组特征。
        </div>
        """, unsafe_allow_html=True)

    # 4.2 聚类结果表格
    st.subheader("4.2 聚类结果表格")
    col1, col2 = st.columns([2, 1])

    with col1:
        table_data = prov_df_cluster[["省份", "区域", "聚类类别", "GDP(亿元)", "人口(万人)", "高校数量", "双一流数量"]].sort_values("聚类类别")
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    with col2:
        cluster_counts = prov_df_cluster["聚类类别"].value_counts().sort_index().reset_index()
        cluster_counts.columns = ["聚类类别", "省份数量"]
        fig11 = px.bar(
            cluster_counts,
            x="聚类类别",
            y="省份数量",
            title="各聚类包含省份数量",
            color="聚类类别",
            color_discrete_sequence=COLOR_PALETTE,
            text="省份数量",
        )
        fig11.update_traces(textposition="outside")
        fig11.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig11, use_container_width=True)

    # 4.3 聚类描述
    st.subheader("4.3 各聚类特征描述")
    cluster_means = prov_df_cluster.groupby("聚类类别")[cluster_features].mean()
    for cluster_label in sorted(cluster_means.index, key=lambda x: int(x)):
        cluster_data = cluster_means.loc[cluster_label]
        provinces_in_cluster = prov_df_cluster[prov_df_cluster["聚类类别"] == cluster_label]["省份"].tolist()
        provinces_str = "、".join(provinces_in_cluster[:5])
        if len(provinces_in_cluster) > 5:
            provinces_str += f" 等{len(provinces_in_cluster)}个省份"

        # 根据特征生成描述
        gdp_level = "高" if cluster_data["GDP(亿元)"] > prov_df["GDP(亿元)"].mean() else "中等" if cluster_data["GDP(亿元)"] > prov_df["GDP(亿元)"].median() else "较低"
        pop_level = "人口众多" if cluster_data["人口(万人)"] > prov_df["人口(万人)"].mean() else "人口适中" if cluster_data["人口(万人)"] > prov_df["人口(万人)"].median() else "人口较少"
        uni_level = "高校资源丰富" if cluster_data["高校数量"] > prov_df["高校数量"].mean() else "高校资源适中" if cluster_data["高校数量"] > prov_df["高校数量"].median() else "高校资源较少"
        sl_level = "优质高校集中" if cluster_data["双一流数量"] > prov_df["双一流数量"].mean() else "优质高校适中"

        st.markdown(f"""
        <div class="cluster-desc">
        <strong>聚类 {cluster_label}</strong>：GDP水平<strong>{gdp_level}</strong>，{pop_level}，
        {uni_level}，{sl_level}。<br>
        包含：{provinces_str}<br>
        <small>GDP均值：{cluster_data['GDP(亿元)']:.0f}亿元 |
        人口均值：{cluster_data['人口(万人)']:.0f}万人 |
        高校均值：{cluster_data['高校数量']:.0f}所 |
        双一流均值：{cluster_data['双一流数量']:.1f}所</small>
        </div>
        """, unsafe_allow_html=True)

    # 4.4 雷达图
    st.subheader("4.4 雷达图 - 各聚类特征均值")
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    fig12 = go.Figure()
    for i, cluster_label in enumerate(cluster_means_norm.index):
        fig12.add_trace(
            go.Scatterpolar(
                r=cluster_means_norm.loc[cluster_label].values,
                theta=cluster_features,
                fill="toself",
                name=f"聚类 {cluster_label}",
                line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)]),
            )
        )
    fig12.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="各聚类特征均值雷达图（归一化）",
        height=500,
        showlegend=True,
    )
    st.plotly_chart(fig12, use_container_width=True)

# ==================== 页面5：预测分析 ====================

elif page == "🔮 预测分析":
    st.title("🔮 预测分析")
    st.markdown("使用线性回归模型，基于经济指标预测各省高校数量。")

    # 准备数据
    features = ["GDP(亿元)", "人口(万人)", "人均GDP(元)", "双一流数量"]
    X = prov_df[features].values
    y = prov_df["高校数量"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # 5.1 特征重要性柱状图
    st.subheader("5.1 特征重要性（回归系数）")
    coef_df = pd.DataFrame({"特征": features, "系数": model.coef_}).sort_values("系数", ascending=True)

    col_pred_a, col_pred_b = st.columns([3, 1])
    with col_pred_a:
        fig13 = px.bar(
            coef_df,
            x="系数",
            y="特征",
            orientation="h",
            title="线性回归特征系数",
            color_discrete_sequence=[COLOR_PALETTE[0]],
            text=coef_df["系数"].round(2),
        )
        fig13.update_traces(textposition="outside")
        fig13.update_layout(height=400, xaxis_title="回归系数", yaxis_title="特征")
        st.plotly_chart(fig13, use_container_width=True)
    with col_pred_b:
        top_feature = coef_df.loc[coef_df["系数"].abs().idxmax()]
        st.markdown(f"""
        <div class="insight-card">
        <strong>📌 特征解读</strong><br>
        影响高校数量最重要的特征是 <strong>{top_feature['特征']}</strong>，
        系数为 <strong>{top_feature['系数']:.2f}</strong>。
        这意味着该特征每增加1个单位，预计高校数量增加 {top_feature['系数']:.2f} 所
        （在其他条件不变的情况下）。<br><br>
        由于特征量纲不同，系数大小不直接代表重要性，需结合实际业务解读。
        </div>
        """, unsafe_allow_html=True)

    # 5.2 模型评估指标
    st.subheader("5.2 模型评估指标")
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("训练集 R²", f"{r2_train:.4f}")
    with col_m2:
        st.metric("测试集 R²", f"{r2_test:.4f}")
    with col_m3:
        st.metric("MAE", f"{mae:.2f}")
    with col_m4:
        st.metric("RMSE", f"{rmse:.2f}")
    with col_m5:
        st.metric("样本数", f"{len(y_test)}")

    st.markdown(f"""
    <div class="insight-card-success">
    <strong>📌 模型解读</strong><br>
    测试集 R² = <strong>{r2_test:.3f}</strong>，说明模型能解释约 <strong>{r2_test*100:.1f}%</strong> 的方差。
    MAE = <strong>{mae:.2f}</strong>，表示预测值与真实值的平均绝对误差约为 <strong>{mae:.0f}</strong> 所高校。
    由于样本量较小（仅31个省份），模型存在一定的过拟合风险（训练集R²={r2_train:.3f} vs 测试集R²={r2_test:.3f}）。
    </div>
    """, unsafe_allow_html=True)

    # 5.3 预测值与真实值对比
    st.subheader("5.3 预测值与真实值对比")

    col_a, col_b = st.columns(2)

    with col_a:
        train_compare = pd.DataFrame({"真实值": y_train, "预测值": y_pred_train})
        fig14 = px.scatter(
            train_compare,
            x="真实值",
            y="预测值",
            title=f"训练集：预测 vs 真实 (R²={r2_train:.3f})",
            color_discrete_sequence=[COLOR_PALETTE[0]],
        )
        max_val_train = max(y_train.max(), y_pred_train.max())
        fig14.add_trace(
            go.Scatter(
                x=[0, max_val_train],
                y=[0, max_val_train],
                mode="lines",
                name="理想线",
                line=dict(color="red", dash="dash"),
            )
        )
        fig14.update_layout(height=400, xaxis_title="真实高校数量", yaxis_title="预测高校数量")
        st.plotly_chart(fig14, use_container_width=True)

    with col_b:
        test_compare = pd.DataFrame({"真实值": y_test, "预测值": y_pred_test})
        fig15 = px.scatter(
            test_compare,
            x="真实值",
            y="预测值",
            title=f"测试集：预测 vs 真实 (R²={r2_test:.3f})",
            color_discrete_sequence=[COLOR_PALETTE[1]],
        )
        max_val_test = max(y_test.max(), y_pred_test.max())
        fig15.add_trace(
            go.Scatter(
                x=[0, max_val_test],
                y=[0, max_val_test],
                mode="lines",
                name="理想线",
                line=dict(color="red", dash="dash"),
            )
        )
        fig15.update_layout(height=400, xaxis_title="真实高校数量", yaxis_title="预测高校数量")
        st.plotly_chart(fig15, use_container_width=True)

    # 5.4 残差分布图
    st.subheader("5.4 残差分布图")
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test

    col_res_a, col_res_b = st.columns(2)

    with col_res_a:
        fig_res_train = px.histogram(
            x=residuals_train,
            nbins=15,
            title="训练集残差分布",
            color_discrete_sequence=[COLOR_PALETTE[0]],
        )
        fig_res_train.add_vline(x=0, line_dash="dash", line_color="red")
        fig_res_train.update_layout(
            height=400,
            xaxis_title="残差（真实值 - 预测值）",
            yaxis_title="频数",
        )
        st.plotly_chart(fig_res_train, use_container_width=True)

    with col_res_b:
        fig_res_test = px.histogram(
            x=residuals_test,
            nbins=15,
            title="测试集残差分布",
            color_discrete_sequence=[COLOR_PALETTE[1]],
        )
        fig_res_test.add_vline(x=0, line_dash="dash", line_color="red")
        fig_res_test.update_layout(
            height=400,
            xaxis_title="残差（真实值 - 预测值）",
            yaxis_title="频数",
        )
        st.plotly_chart(fig_res_test, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card">
    <strong>📌 残差分析</strong><br>
    残差均值为 {np.mean(residuals_test):.2f}（理想值为0），标准差为 {np.std(residuals_test):.2f}。
    残差分布应大致呈正态分布且以0为中心，说明模型没有系统性偏差。
    如果残差呈现明显偏态或存在极端值，则表明模型对某些省份的预测存在较大误差。
    </div>
    """, unsafe_allow_html=True)

    # 5.5 用户输入预测
    st.subheader("5.5 交互式预测：输入经济指标预测高校数量")
    st.markdown("输入GDP和人口数据，预测该省可能拥有的高校数量。")

    col_input1, col_input2, col_input3, col_input4 = st.columns(4)
    with col_input1:
        input_gdp = st.number_input("GDP（亿元）", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)
    with col_input2:
        input_pop = st.number_input("人口（万人）", min_value=0.0, max_value=15000.0, value=5000.0, step=100.0)
    with col_input3:
        input_gdp_per_capita = input_gdp / input_pop * 10000 if input_pop > 0 else 0.0
        st.metric("人均GDP（元）", f"{input_gdp_per_capita:.0f}")
    with col_input4:
        input_sl = st.number_input("双一流数量", min_value=0, max_value=50, value=5, step=1)

    if st.button("🔍 预测高校数量", type="primary"):
        input_features = np.array([[input_gdp, input_pop, input_gdp_per_capita, input_sl]])
        prediction = model.predict(input_features)[0]
        prediction = max(0, round(prediction))

        st.success(f"预测该省高校数量约为：**{prediction} 所**")

        # 显示输入在散点图中的位置
        fig16 = px.scatter(
            prov_df,
            x="GDP(亿元)",
            y="高校数量",
            size="人口(万人)",
            color="区域",
            hover_name="省份",
            title="您的输入在全国各省中的位置",
            color_discrete_sequence=COLOR_PALETTE,
            size_max=40,
        )
        fig16.add_trace(
            go.Scatter(
                x=[input_gdp],
                y=[prediction],
                mode="markers",
                marker=dict(color="red", size=20, symbol="star"),
                name="预测值",
            )
        )
        fig16.update_layout(height=500)
        st.plotly_chart(fig16, use_container_width=True)

        st.markdown(f"""
        <div class="insight-card-success">
        <strong>📌 预测解释</strong><br>
        基于线性回归模型，在GDP为 <strong>{input_gdp:.0f}</strong> 亿元、
        人口 <strong>{input_pop:.0f}</strong> 万人、
        人均GDP <strong>{input_gdp_per_capita:.0f}</strong> 元、
        双一流高校 <strong>{input_sl}</strong> 所的条件下，
        预计该省拥有约 <strong>{prediction}</strong> 所本科院校。
        模型测试集R²为 <strong>{r2_test:.3f}</strong>，MAE为 <strong>{mae:.2f}</strong>，
        预测结果供参考，实际值可能因政策、历史等因素有所偏差。
        </div>
        """, unsafe_allow_html=True)

# ==================== 页脚 ====================

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 数说高校 | 数据可视化分析平台")
st.sidebar.caption(f"当前数据：{total_uni} 所高校，{uni_df['省份'].nunique()} 个省份")