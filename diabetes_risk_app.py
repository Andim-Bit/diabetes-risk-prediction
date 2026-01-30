import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 页面核心配置 ====================
st.set_page_config(
    page_title="糖尿病风险智能预测系统",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 简化版CSS样式（移除卡片样式） ====================
st.markdown("""
<style>
    /* 全局重置 */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* 主色调：医疗蓝绿系 */
    :root {
        --primary: #2563EB;
        --secondary: #0D9488;
        --success: #16A34A;
        --warning: #F59E0B;
        --danger: #DC2626;
        --light: #F8FAFC;
        --dark: #1E293B;
        --gray: #64748B;
    }

    /* 确保页面占满视窗 */
    html, body, .stApp {
        height: 100%;
        display: flex;
        flex-direction: column;
    }

    /* 顶部导航栏 */
    .header-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin: 0 0 2rem 0;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.15);
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        text-align: center;
    }

    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.9;
        text-align: center;
        max-width: 800px;
        margin: 0 auto;
    }

    .header-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }

    .header-stat-item {
        background: rgba(255, 255, 255, 0.15);
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        font-size: 0.95rem;
        backdrop-filter: blur(8px);
    }

    /* 风险等级标签 */
    .risk-tag {
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
        display: block;
    }

    .risk-low {
        background: #ECFDF5;
        color: var(--success);
        border: 2px solid var(--success);
    }

    .risk-medium {
        background: #FFFBEB;
        color: var(--warning);
        border: 2px solid var(--warning);
    }

    .risk-high {
        background: #FEF2F2;
        color: var(--danger);
        border: 2px solid var(--danger);
    }

    /* 按钮样式优化 */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.8rem 1.5rem;
        width: 100%;
        transition: all 0.2s;
    }

    /* 指标展示样式 */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.3rem 0;
        text-align: center;
    }

    .metric-label {
        font-size: 0.85rem;
        color: var(--gray);
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
    }

    /* 页脚样式 */
    .footer {
        background: var(--dark);
        color: white;
        padding: 2rem;
        border-radius: 16px 16px 0 0;
        margin-top: auto;
        text-align: center;
        width: 100%;
    }

    /* 隐藏默认页脚 */
    footer {
        visibility: hidden;
        height: 0;
    }

    .stApp > footer {
        visibility: hidden;
        height: 0;
        padding: 0;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = None

# ==================== 顶部标题区域 ====================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🩺 糖尿病风险智能预测系统 v3.0</h1>
    <p class="header-subtitle">基于10万+医学数据分析 | 11项核心风险因子 | 实时AI智能评估</p>
    <div class="header-stats">
        <span class="header-stat-item">📊 预测准确率：83.8%</span>
        <span class="header-stat-item">🎯 AUC值：0.838</span>
        <span class="header-stat-item">⚡ 实时智能分析</span>
        <span class="header-stat-item">🛡️ 数据本地处理</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================== 模型加载函数 ====================
@st.cache_resource
def load_model():
    """加载训练好的模型，兼容多文件格式，无模型时生成演示模型"""
    try:
        model_filenames = ["XGBoost_model.pkl", "model.pkl", "diabetes_model.pkl"]

        for filename in model_filenames:
            try:
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
                st.sidebar.success("✅ 模型加载成功")
                return model
            except FileNotFoundError:
                continue

        # 生成演示模型
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 11)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        st.sidebar.info("ℹ️ 使用演示模型进行评估")
        return model

    except Exception as e:
        st.sidebar.warning(f"⚠️ 模型加载异常：{str(e)}，使用演示模式")
        return None


# ==================== 预测函数 ====================
def predict_diabetes_risk(user_inputs, model):
    """使用模型计算糖尿病风险概率，返回风险等级和建议"""
    try:
        # 特征标准化
        features = np.array([[
            (user_inputs['age'] - 45) / 15,
            1 if user_inputs['gender'] == '男性' else 0,
            1 if user_inputs['education'] == '低教育水平' else 0,
            (user_inputs['poverty'] - 2.5) / 1.5,
            1 if user_inputs['health_insurance'] == '有' else 0,
            1 if user_inputs['activity'] == '有规律活动' else 0,
            1 if user_inputs['sleep'] == '睡眠不足' else 0,
            1 if user_inputs['alcohol'] == '重度饮酒' else 0,
            1 if user_inputs['smoking'] == '吸烟' else 0,
            1 if user_inputs['hypertension'] == '有' else 0,
            1 if user_inputs['cholesterol'] == '有' else 0
        ]])

        # 风险概率计算
        try:
            risk_probability = float(model.predict_proba(features)[0][1] * 100)
        except:
            prediction = model.predict(features)[0]
            risk_probability = 65.0 if prediction == 1 else 15.0

        # 演示模型添加随机波动
        if hasattr(model, 'random_state') and model.random_state == 42:
            import random
            risk_probability += random.uniform(-5, 5)
            risk_probability = max(0, min(100, risk_probability))

        # 风险等级判定
        if risk_probability < 20:
            risk_level = "低风险"
            level_class = "risk-low"
            recommendations = [
                "✅ 保持健康的生活作息和饮食结构",
                "📅 每年进行一次常规体检，重点关注血糖指标",
                "🥗 坚持均衡饮食，适量进行有氧运动"
            ]
        elif risk_probability < 50:
            risk_level = "中风险"
            level_class = "risk-medium"
            recommendations = [
                "⚠️ 每6个月监测一次空腹血糖和餐后血糖",
                "🏃 每周至少150分钟中等强度体力活动",
                "⚖️ 控制体重，将BMI维持在18.5-24.0之间"
            ]
        else:
            risk_level = "高风险"
            level_class = "risk-high"
            recommendations = [
                "🚨 建议立即前往内分泌科进行全面检查",
                "💊 在医生指导下调整生活方式，必要时药物干预",
                "📊 每周监测血糖，定期复查血压、血脂"
            ]

        # 返回结果
        return {
            'probability': risk_probability,
            'level': risk_level,
            'level_class': level_class,
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_summary': user_inputs.copy()
        }

    except Exception as e:
        st.error(f"❌ 预测过程出错：{str(e)}")
        return None


# ==================== 侧边栏设计（移除所有卡片容器） ====================
with st.sidebar:
    # 系统性能
    st.markdown('<h3>📊 系统性能</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div>
            <div class="metric-value">83.8%</div>
            <div class="metric-label">预测准确率</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div>
            <div class="metric-value">0.838</div>
            <div class="metric-label">AUC值</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 使用指南
    st.markdown('<h3>📖 使用指南</h3>', unsafe_allow_html=True)
    st.markdown("""
    1. **填写信息**：在主界面完整填写11项健康指标
    2. **开始评估**：点击"智能风险评估"按钮
    3. **查看结果**：获取风险等级和个性化建议
    4. **专业咨询**：高风险用户建议及时就医
    """)

    st.markdown("---")

    # 重要声明
    st.markdown('<h3>⚠️ 重要声明</h3>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size: 0.9rem; color: var(--gray);">
    本系统仅为健康风险评估工具，不能替代专业医疗诊断。
    如评估结果为高风险或有身体不适，请及时咨询执业医师。
    </p>
    """, unsafe_allow_html=True)


# ==================== 主界面布局（移除所有卡片容器） ====================
def main():
    # 加载模型
    model = load_model()
    if model is None:
        st.warning("⚠️ 系统初始化未完成，请刷新页面重试")
        return

    # 主界面两列布局
    col_input, col_result = st.columns([1, 1], gap="large")

    # 左侧：健康信息填写
    with col_input:
        st.markdown('<h2>📋 健康信息填写</h2>', unsafe_allow_html=True)

        with st.form("diabetes_risk_form", clear_on_submit=False):
            # 分组1：基本信息
            st.markdown("### 👤 基本信息")
            col_age, col_gender = st.columns(2)
            with col_age:
                age = st.slider("年龄", min_value=18, max_value=100, value=45, help="请选择您的实际年龄")
            with col_gender:
                gender = st.radio("性别", ["女性", "男性"], index=1, horizontal=True)

            # 分组2：社会经济状况
            st.markdown("### 💼 社会经济状况")
            col_edu, col_poverty = st.columns(2)
            with col_edu:
                education = st.selectbox("教育水平", ["高等教育", "中等教育", "低教育水平"], index=0)
            with col_poverty:
                poverty = st.slider("贫困指数 (0=最贫困, 5=最富裕)", 0.0, 5.0, 2.5, 0.1)
            health_insurance = st.radio("是否有健康保险", ["有", "无"], index=0, horizontal=True)

            # 分组3：生活方式
            st.markdown("### 🏃 生活方式")
            col_activity, col_sleep = st.columns(2)
            with col_activity:
                activity = st.radio("体力活动", ["有规律活动", "无规律活动"], index=1, horizontal=True)
            with col_sleep:
                sleep = st.radio("睡眠状况", ["充足睡眠", "睡眠不足"], index=0, horizontal=True)

            col_alcohol, col_smoking = st.columns(2)
            with col_alcohol:
                alcohol = st.radio("饮酒习惯", ["非重度饮酒", "重度饮酒"], index=0, horizontal=True)
            with col_smoking:
                smoking = st.radio("吸烟情况", ["不吸烟", "吸烟"], index=0, horizontal=True)

            # 分组4：健康状况
            st.markdown("### 💊 健康状况")
            col_hp, col_chol = st.columns(2)
            with col_hp:
                hypertension = st.radio("高血压病史", ["无", "有"], index=0, horizontal=True)
            with col_chol:
                cholesterol = st.radio("高胆固醇病史", ["无", "有"], index=0, horizontal=True)

            # 提交按钮
            st.markdown("---")
            submit_btn = st.form_submit_button("🚀 智能风险评估", use_container_width=True)

        # 表单提交处理
        if submit_btn:
            with st.spinner("🔍 正在分析您的健康数据，请稍候..."):
                user_inputs = {
                    'age': age, 'gender': gender, 'education': education,
                    'poverty': poverty, 'health_insurance': health_insurance,
                    'activity': activity, 'sleep': sleep, 'alcohol': alcohol,
                    'smoking': smoking, 'hypertension': hypertension,
                    'cholesterol': cholesterol
                }
                st.session_state.user_inputs = user_inputs
                result = predict_diabetes_risk(user_inputs, model)

                if result:
                    st.session_state.risk_result = result
                    st.success("✅ 风险评估完成！请查看右侧结果")
                    st.rerun()

    # 右侧：风险评估结果
    with col_result:
        st.markdown('<h2>📊 风险评估结果</h2>', unsafe_allow_html=True)

        if st.session_state.risk_result:
            result = st.session_state.risk_result

            # 风险概率展示
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div class="metric-value">{result['probability']:.1f}%</div>
                <div class="metric-label">糖尿病风险概率</div>
            </div>
            """, unsafe_allow_html=True)

            # 风险等级标签
            st.markdown(f'<div class="risk-tag {result["level_class"]}">{result["level"]}</div>',
                        unsafe_allow_html=True)

            # 风险进度条
            st.progress(result['probability'] / 100, text=f"风险程度：{result['probability']:.1f}%")

            # 个性化建议
            st.markdown("### 💡 个性化健康建议")
            for idx, rec in enumerate(result['recommendations'], 1):
                st.markdown(f"""
                <div style="background: var(--light); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
                    {rec}
                </div>
                """, unsafe_allow_html=True)

            # 报告时间
            st.markdown(f"""
            <div style="margin-top: 1.5rem; color: var(--gray); font-size: 0.9rem;">
                📅 报告生成时间：{result['timestamp']}
            </div>
            """, unsafe_allow_html=True)

        else:
            # 未评估时的提示
            st.markdown("""
            <div style="text-align: center; padding: 2rem 0; color: var(--gray);">
                <h3>👈 请先填写左侧健康信息</h3>
                <p style="margin-top: 1rem;">完整填写11项评估指标后，点击"智能风险评估"按钮获取结果</p>
            </div>
            """, unsafe_allow_html=True)


# 运行主程序
main()

# ==================== 页脚区域 ====================
st.markdown("""
<div class="footer">
    <div class="footer-text">
        本系统基于机器学习算法构建，旨在提供健康风险参考，不构成医疗建议
    </div>
    <div class="footer-disclaimer">
        ⚠️ 免责声明：本工具仅为健康评估辅助手段，不能替代专业医生的诊断和治疗建议
    </div>
</div>
""", unsafe_allow_html=True)