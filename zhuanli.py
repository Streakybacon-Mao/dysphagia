import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# 加载模型
with open('catboost_model_1.pkl', 'rb') as f:
    model = pickle.load(f)

# 创建UI
st.title("脑卒中后吞咽障碍风险预测模型")
st.markdown("---")

# 定义所有特征（包括gender）
all_features = [
     'age', 'GCS', 'zhongxinglixbao', 'hongxibao',
    'huohuabufenningxuemeiyuanshijian', 'jianxinglinsuanmei',
    'ganyousanzhi', 'gaomiduzhidanbai', 'MEI', 'P',
    'race', 'smoker', 'aki', 'baixuezheng', 'diabetes',
    'xinlishuaijie', 'jietan', 'exingzhongliu', 'shenzangdaitizhiliao'
]

# 分类变量（包括gender）
categorical_vars = ['gender', 'race', 'smoker', 'aki', 'baixuezheng', 'diabetes',
                    'xinlishuaijie', 'jietan', 'exingzhongliu', 'shenzangdaitizhiliao']

# 连续变量（不包括gender）
continuous_vars = ['age', 'GCS', 'zhongxinglixbao', 'hongxibao',
                   'huohuabufenningxuemeiyuanshijian', 'jianxinglinsuanmei',
                   'ganyousanzhi', 'gaomiduzhidanbai', 'MEI', 'P']

# 连续变量的取值范围
continuous_ranges = {
    'age': (18, 120),  # 年龄
    'GCS': (3, 15),  # 格拉斯哥昏迷评分
    'zhongxinglixbao': (30, 90),  # 中性粒细胞百分比
    'hongxibao': (2.0, 6.0),  # 红细胞计数
    'huohuabufenningxuemeiyuanshijian': (20, 100),  # 活化部分凝血活酶时间
    'jianxinglinsuanmei': (30, 300),  # 碱性磷酸酶
    'ganyousanzhi': (0.1, 5.0),  # 甘油三酯
    'gaomiduzhidanbai': (0.5, 2.5),  # 高密度脂蛋白
    'MEI': (0.5, 1.5),  # 镁
    'P': (0.5, 1.8)  # 磷
}

# 连续变量的单位和描述
continuous_descriptions = {
    'age': '岁',
    'GCS': '分 (3-15)',
    'zhongxinglixbao': '% (30-90)',
    'hongxibao': '10^12/L (2.0-6.0)',
    'huohuabufenningxuemeiyuanshijian': '秒 (20-100)',
    'jianxinglinsuanmei': 'U/L (30-300)',
    'ganyousanzhi': 'mmol/L (0.1-5.0)',
    'gaomiduzhidanbai': 'mmol/L (0.5-2.5)',
    'MEI': 'mmol/L (0.5-1.5)',
    'P': 'mmol/L (0.5-1.8)'
}

# 分类变量的描述
categorical_descriptions = {
    'gender': '性别',
    'race': '种族',
    'smoker': '吸烟史',
    'aki': '急性肾损伤',
    'baixuezheng': '白血病',
    'diabetes': '糖尿病',
    'xinlishuaijie': '心力衰竭',
    'jietan': '截瘫',
    'exingzhongliu': '恶性肿瘤',
    'shenzangdaitizhiliao': '肾脏替代治疗'
}

# 特殊处理的分类变量
special_vars = {
    'gender': {
        0: '女',
        1: '男'
    },
    'race': {
        1: '白人',
        2: '黑人',
        3: '亚裔',
        4: '其他'
    }
}
# 使用两列布局
col1, col2 = st.columns(2)

with col1:
    st.header("患者基本信息")

    # 性别输入
    gender_description = categorical_descriptions['gender']
    gender_options = list(special_vars['gender'].values())
    selected_gender = st.radio(f"{gender_description}", gender_options, horizontal=True)
    categorical_input = {'gender': [k for k, v in special_vars['gender'].items() if v == selected_gender][0]}

    # 年龄输入
    age_min, age_max = continuous_ranges['age']
    continuous_input = {'age': st.slider(
        "年龄 (岁)",
        min_value=age_min,
        max_value=age_max,
        value=65,
        step=1
    )}

    # 其他连续变量
    for var in continuous_vars[1:]:
        min_val, max_val = continuous_ranges[var]
        description = continuous_descriptions[var]
        continuous_input[var] = st.slider(
            f"{var} ({description})",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float((min_val + max_val) / 2),
            step=0.1
        )

with col2:
    st.header("患者临床特征")

    # 处理特殊分类变量（除gender外）
    for var in special_vars:
        if var == 'gender':  # 已经在col1处理过
            continue

        options = list(special_vars[var].values())
        description = categorical_descriptions.get(var, var)
        selected = st.selectbox(f"{description}", options)
        categorical_input[var] = [k for k, v in special_vars[var].items() if v == selected][0]

    # 处理其他二元分类变量
    for var in categorical_vars:
        if var in special_vars:  # 已经处理过
            continue

        description = categorical_descriptions.get(var, var)
        selected = st.radio(f"{description}", ['否', '是'], horizontal=True)
        categorical_input[var] = 1 if selected == '是' else 0

# 将输入转换为DataFrame
input_data = {**continuous_input, **categorical_input}
input_df = pd.DataFrame([input_data])

# 确保输入数据的列顺序与模型训练时的特征顺序一致
input_df = input_df[all_features]

# 预测按钮
st.markdown("---")
if st.button("预测吞咽障碍风险", use_container_width=True):
    # 获取预测概率
    probabilities = model.predict_proba(input_df)
    dysphagia_probability = probabilities[0][1] * 100  # 转换为百分比

    # 显示预测结果 - 使用进度条和颜色编码
    st.subheader("预测结果")
    if dysphagia_probability < 30:
        color = "green"
    elif dysphagia_probability < 60:
        color = "orange"
    else:
        color = "red"

    st.metric("吞咽障碍风险概率", f"{dysphagia_probability:.1f}%")
    st.progress(int(dysphagia_probability))
    st.caption(f"风险等级: {'低' if dysphagia_probability < 30 else '中' if dysphagia_probability < 60 else '高'}")

    # 使用SHAP解释预测
    st.subheader("风险因素分析")
    with st.spinner("生成解释..."):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # 绘制SHAP力图
            plt.figure(figsize=(10, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                input_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            plt.title("各因素对预测结果的影响", fontsize=14)
            plt.tight_layout()
            st.pyplot(plt)

            # 绘制SHAP条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
            plt.title("特征重要性", fontsize=14)
            plt.tight_layout()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"生成解释时出错: {str(e)}")

# 显示输入数据
st.markdown("---")
if st.checkbox("显示输入数据详情"):
    st.subheader("输入数据详情")
    # 创建更易读的显示
    display_df = input_df.copy()

    # 重命名列
    display_df.columns = [
        '性别', '年龄', 'GCS评分', '中性粒细胞(%)', '红细胞', '凝血酶时间',
        '碱性磷酸酶', '甘油三酯', '高密度脂蛋白', '镁', '磷',
        '种族', '吸烟', '急性肾损伤', '白血病', '糖尿病',
        '心力衰竭', '截瘫', '恶性肿瘤', '肾脏替代治疗'
    ]

    # 转换分类变量
    display_df['性别'] = display_df['性别'].replace({0: '女', 1: '男'})
    display_df['种族'] = display_df['种族'].replace({1: '白人', 2: '黑人', 3: '亚裔', 4: '其他'})

    # 转换二元变量
    binary_cols = ['吸烟', '急性肾损伤', '白血病', '糖尿病', '心力衰竭', '截瘫', '恶性肿瘤', '肾脏替代治疗']
    display_df[binary_cols] = display_df[binary_cols].replace({0: '否', 1: '是'})

    st.dataframe(display_df.style.highlight_max(axis=0, color='#fffd75'))

# 添加页脚
st.markdown("---")
st.caption("© 2025 脑卒中后吞咽障碍风险预测模型 | 仅供医疗专业人员使用")