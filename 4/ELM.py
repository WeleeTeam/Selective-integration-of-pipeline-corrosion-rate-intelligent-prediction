import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.linalg import pinv
import joblib
import os
import warnings
import time
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)


# 1. 数据生成函数 - 模拟管道腐蚀数据
def generate_corrosion_data(num_samples=5000):
    """生成模拟的管道腐蚀数据集"""
    np.random.seed(42)
    data = {
        'temperature': np.random.uniform(20, 120, num_samples),
        'pressure': np.random.uniform(1, 100, num_samples),
        'flow_rate': np.random.uniform(0.1, 5.0, num_samples),
        'Cl_conc': np.random.uniform(500, 50000, num_samples),
        'pH': np.random.uniform(3, 10, num_samples),
        'O2_content': np.random.uniform(0.1, 10.0, num_samples),
        'CO2_conc': np.random.uniform(0, 500, num_samples),
        'H2S_conc': np.random.uniform(0, 50, num_samples),
        'pipe_age': np.random.randint(1, 30, num_samples),
        'material': np.random.choice(['carbon_steel', 'stainless_steel', 'alloy_steel', 'copper_alloy'], num_samples,
                                     p=[0.6, 0.2, 0.15, 0.05]),
        'coating': np.random.choice(['none', 'epoxy', 'polyethylene', 'fusion_bonded'], num_samples,
                                    p=[0.3, 0.4, 0.2, 0.1]),
        'corrosion_type': np.random.choice(['uniform', 'pitting', 'crevice', 'erosion', 'stress_cracking'], num_samples,
                                           p=[0.5, 0.2, 0.1, 0.15, 0.05]),
        'corrosion_rate': np.zeros(num_samples)
    }

    # 基于物理模型模拟腐蚀速率
    for i in range(num_samples):
        # 基础腐蚀速率
        base_rate = 0.02

        # 温度影响 (20-120°C)
        temp_effect = (data['temperature'][i] - 20) * 0.0015

        # 氯离子影响 (500-50000 ppm)
        cl_effect = data['Cl_conc'][i] * 0.000001

        # 流速影响 (0.1-5.0 m/s)
        flow_effect = data['flow_rate'][i] * 0.02

        # pH值影响 (3-10)
        ph_effect = 0.1 / (data['pH'][i] + 0.1) if data['pH'][i] < 7 else 0

        # 材质影响
        if data['material'][i] == 'carbon_steel':
            material_factor = 1.5
        elif data['material'][i] == 'stainless_steel':
            material_factor = 0.2
        elif data['material'][i] == 'alloy_steel':
            material_factor = 0.8
        else:  # copper_alloy
            material_factor = 0.5

        # 涂层影响
        if data['coating'][i] == 'none':
            coating_factor = 1.0
        elif data['coating'][i] == 'epoxy':
            coating_factor = 0.6
        elif data['coating'][i] == 'polyethylene':
            coating_factor = 0.4
        else:  # fusion_bonded
            coating_factor = 0.3

        # 腐蚀类型影响
        if data['corrosion_type'][i] == 'uniform':
            type_factor = 1.0
        elif data['corrosion_type'][i] == 'pitting':
            type_factor = 2.0
        elif data['corrosion_type'][i] == 'crevice':
            type_factor = 1.8
        elif data['corrosion_type'][i] == 'erosion':
            type_factor = 1.7
        else:  # stress_cracking
            type_factor = 2.5

        # 管龄影响
        age_factor = 1.0 + (data['pipe_age'][i] * 0.005)

        # 计算最终腐蚀速率
        data['corrosion_rate'][i] = (
                (base_rate + temp_effect + cl_effect + flow_effect + ph_effect) *
                material_factor * coating_factor * type_factor * age_factor +
                np.random.normal(0, 0.005)  # 添加随机噪声
        )

    return pd.DataFrame(data)


# 2. 数据探索性分析
def explore_data(df):
    """执行数据探索性分析并生成可视化"""
    print("数据集信息:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1} (包括目标变量腐蚀速率)")

    # 目标变量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df['corrosion_rate'], kde=True, bins=30)
    plt.title('腐蚀速率分布')
    plt.xlabel('腐蚀速率 (mm/year)')
    plt.ylabel('频率')
    plt.savefig('corrosion_rate_distribution.png', dpi=300)
    plt.show()

    # 数值特征相关性
    numerical_features = ['temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
                          'O2_content', 'CO2_conc', 'H2S_conc', 'pipe_age', 'corrosion_rate']

    corr_matrix = df[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('数值特征相关性矩阵')
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

    # 类别特征分布
    categorical_features = ['material', 'coating', 'corrosion_type']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, feature in enumerate(categorical_features):
        sns.countplot(data=df, x=feature, ax=axes[i])
        axes[i].set_title(f'{feature} 分布')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_distribution.png', dpi=300)
    plt.show()

    # 关键特征与腐蚀速率的关系
    key_features = ['temperature', 'Cl_conc', 'flow_rate', 'material']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, feature in enumerate(key_features):
        ax = axes[i // 2, i % 2]
        if feature in numerical_features:
            sns.scatterplot(data=df, x=feature, y='corrosion_rate', alpha=0.5, ax=ax)
        else:
            sns.boxplot(data=df, x=feature, y='corrosion_rate', ax=ax)
            ax.tick_params(axis='x', rotation=45)
        ax.set_title(f'腐蚀速率 vs {feature}')
    plt.tight_layout()
    plt.savefig('key_features_vs_corrosion.png', dpi=300)
    plt.show()


# 3. 特征工程
def feature_engineering(df):
    """创建具有物理意义的特征组合"""
    # 创建物理意义特征组合
    df['temp_cl_index'] = df['temperature'] * df['Cl_conc'] / 10000
    df['flow_pressure_ratio'] = df['flow_rate'] / (df['pressure'] + 1e-6)
    df['corrosion_aggressiveness'] = (
                                             0.4 * df['Cl_conc'] +
                                             0.3 * df['temperature'] +
                                             0.2 * df['CO2_conc'] +
                                             0.1 * df['H2S_conc']
                                     ) / 1000

    # 管龄相关特征
    df['age_degradation'] = df['pipe_age'] * 0.005 * df['corrosion_aggressiveness']

    return df


# 4. 数据预处理
def preprocess_data(df):
    """预处理数据：分割、转换、缩放"""
    # 分离特征和目标
    X = df.drop('corrosion_rate', axis=1)
    y = df['corrosion_rate'].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 定义数值和类别特征
    numerical_features = [
        'temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
        'O2_content', 'CO2_conc', 'H2S_conc', 'pipe_age',
        'temp_cl_index', 'flow_pressure_ratio',
        'corrosion_aggressiveness', 'age_degradation'
    ]

    categorical_features = ['material', 'coating', 'corrosion_type']

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'  # 忽略其他列
    )

    # 应用预处理
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f"训练数据形状: {X_train_preprocessed.shape}")
    print(f"测试数据形状: {X_test_preprocessed.shape}")

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor


# 5. 极限学习机(ELM)实现
class ExtremeLearningMachine(BaseEstimator, RegressorMixin):
    """
    极限学习机(ELM)回归器实现
    参数:
        n_hidden: 隐层神经元数量
        activation: 激活函数 ('sigmoid', 'relu', 'tanh')
        alpha: L2正则化系数
        random_state: 随机种子
    """

    def __init__(self, n_hidden=100, activation='sigmoid', alpha=0.001, random_state=None):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state
        self.input_weights_ = None
        self.biases_ = None
        self.output_weights_ = None

    def _activation_function(self, x):
        """应用激活函数"""
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

    def fit(self, X, y):
        """训练ELM模型"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # 1. 随机初始化输入权重和偏置
        self.input_weights_ = np.random.normal(size=(n_features, self.n_hidden))
        self.biases_ = np.random.normal(size=(self.n_hidden,))

        # 2. 计算隐层输出
        H = np.dot(X, self.input_weights_) + self.biases_
        H = self._activation_function(H)

        # 3. 计算输出权重 (使用正则化最小二乘)
        # H^T H + αI
        HtH = np.dot(H.T, H)
        regularization = self.alpha * np.eye(HtH.shape[0])

        # 计算伪逆 (H^T H + αI)^{-1} H^T
        try:
            # 尝试直接求逆
            inv_H = np.linalg.inv(HtH + regularization)
        except np.linalg.LinAlgError:
            # 如果求逆失败，使用伪逆
            inv_H = pinv(HtH + regularization)

        # 输出权重 β = (H^T H + αI)^{-1} H^T y
        self.output_weights_ = np.dot(inv_H, np.dot(H.T, y))

        return self

    def predict(self, X):
        """使用ELM模型进行预测"""
        # 计算隐层输出
        H = np.dot(X, self.input_weights_) + self.biases_
        H = self._activation_function(H)

        # 计算预测值
        y_pred = np.dot(H, self.output_weights_)

        return y_pred


# 6. 模型训练与调优
def train_and_tune_elm(X_train, y_train):
    """训练并调优ELM模型"""
    # 创建基础ELM模型
    elm = ExtremeLearningMachine(random_state=42)

    # 定义超参数搜索空间
    param_dist = {
        'n_hidden': randint(50, 500),
        'activation': ['sigmoid', 'relu', 'tanh'],
        'alpha': uniform(1e-6, 1e-2)
    }

    # 使用随机搜索进行超参数调优
    random_search = RandomizedSearchCV(
        estimator=elm,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("开始ELM超参数搜索...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"超参数搜索完成，耗时: {end_time - start_time:.2f}秒")
    print("最佳参数:", random_search.best_params_)
    print("最佳MAE:", -random_search.best_score_)

    # 使用最佳参数训练最终模型
    best_elm = random_search.best_estimator_

    return best_elm


# 7. 模型评估与可视化
def evaluate_model(model, X_test, y_test):
    """评估模型性能并生成可视化"""
    # 测试集预测
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n模型评估结果:")
    print(f"测试集MAE: {mae:.6f} mm/year")
    print(f"测试集RMSE: {rmse:.6f} mm/year")
    print(f"测试集R²: {r2:.4f}")
    print(f"预测速度: {prediction_time:.4f}秒 (共{X_test.shape[0]}个样本)")

    # 可视化预测 vs 实际值
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('预测值 vs 实际值')
    plt.xlabel('实际腐蚀速率 (mm/year)')
    plt.ylabel('预测腐蚀速率 (mm/year)')
    plt.grid(True)
    plt.savefig('elm_prediction_vs_actual.png', dpi=300)
    plt.show()

    # 误差分布
    plt.figure(figsize=(10, 6))
    errors = y_pred - y_test
    sns.histplot(errors, kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('预测误差分布')
    plt.xlabel('预测误差 (mm/year)')
    plt.savefig('elm_error_distribution.png', dpi=300)
    plt.show()

    # 残差图
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('预测残差图')
    plt.xlabel('预测腐蚀速率 (mm/year)')
    plt.ylabel('残差 (实际 - 预测)')
    plt.savefig('elm_residuals.png', dpi=300)
    plt.show()

    # 模型性能报告
    performance_report = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Max_Error': np.max(np.abs(errors)),
        'Mean_Absolute_Percentage_Error': np.mean(np.abs(residuals / y_test)) * 100,
        'Prediction_Time': prediction_time
    }

    return performance_report


# 8. 特征重要性分析
def analyze_feature_importance(model, preprocessor, feature_names):
    """分析特征重要性（基于输出权重）"""
    # 获取特征名称
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(model.input_weights_.shape[0])]

    # 计算特征重要性（基于输入权重的绝对值之和）
    feature_importance = np.sum(np.abs(model.input_weights_), axis=1)

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature',
                data=importance_df.head(20).sort_values('Importance', ascending=True))
    plt.title('Top 20 腐蚀预测特征重要性 (基于ELM输入权重)')
    plt.tight_layout()
    plt.savefig('elm_feature_importance.png', dpi=300)
    plt.show()

    return importance_df


# 9. 模型部署函数
class CorrosionPredictor:
    """管道腐蚀预测器类，用于加载模型并进行预测"""

    def __init__(self, model_path='best_corrosion_elm_model.pkl',
                 preprocessor_path='corrosion_preprocessor.pkl'):
        """初始化预测器，加载模型和预处理器"""
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"预处理器文件 {preprocessor_path} 不存在")

        # 加载模型和预处理器
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        # 获取特征名称
        self.numeric_features = [
            'temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
            'O2_content', 'CO2_conc', 'H2S_conc', 'pipe_age',
            'temp_cl_index', 'flow_pressure_ratio',
            'corrosion_aggressiveness', 'age_degradation'
        ]

        # 获取类别特征名称
        cat_encoder = self.preprocessor.named_transformers_['cat']
        self.categorical_features = cat_encoder.get_feature_names_out(
            ['material', 'coating', 'corrosion_type']
        )

        self.all_feature_names = np.concatenate([self.numeric_features, self.categorical_features])

    def predict(self, input_data):
        """预测管道腐蚀速率"""
        # 如果输入是字典，转换为DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("输入数据必须是字典或DataFrame")

        # 添加衍生特征
        input_df['temp_cl_index'] = input_df['temperature'] * input_df['Cl_conc'] / 10000
        input_df['flow_pressure_ratio'] = input_df['flow_rate'] / (input_df['pressure'] + 1e-6)
        input_df['corrosion_aggressiveness'] = (
                                                       0.4 * input_df['Cl_conc'] +
                                                       0.3 * input_df['temperature'] +
                                                       0.2 * input_df['CO2_conc'] +
                                                       0.1 * input_df['H2S_conc']
                                               ) / 1000
        input_df['age_degradation'] = input_df['pipe_age'] * 0.005 * input_df['corrosion_aggressiveness']

        # 预处理
        processed_data = self.preprocessor.transform(input_df)

        # 预测
        prediction = self.model.predict(processed_data)[0]

        # 添加安全边界（腐蚀速率不可能为负）
        prediction = max(prediction, 0.001)

        return prediction


# 主函数
def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    print("=" * 50)
    print("管道腐蚀预测模型 - 极限学习机(ELM)实现")
    print("=" * 50)

    # 步骤1: 生成模拟数据
    print("\n步骤1: 生成模拟管道腐蚀数据...")
    corrosion_data = generate_corrosion_data(8000)
    print(f"生成数据完成: {corrosion_data.shape[0]} 个样本")

    # 保存数据用于后续分析
    corrosion_data.to_csv('corrosion_data.csv', index=False)
    print("已保存数据到 'corrosion_data.csv'")

    # 步骤2: 数据探索性分析
    print("\n步骤2: 执行数据探索性分析...")
    explore_data(corrosion_data)

    # 步骤3: 特征工程
    print("\n步骤3: 执行特征工程...")
    corrosion_data = feature_engineering(corrosion_data)
    print("衍生特征已添加: temp_cl_index, flow_pressure_ratio, corrosion_aggressiveness, age_degradation")

    # 步骤4: 数据预处理
    print("\n步骤4: 数据预处理...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(corrosion_data)

    # 步骤5: 训练并调优ELM模型
    print("\n步骤5: 训练并调优ELM模型...")
    best_elm = train_and_tune_elm(X_train, y_train)

    # 保存最佳模型
    joblib.dump(best_elm, 'best_corrosion_elm_model.pkl')
    print("已保存最佳ELM模型为 'best_corrosion_elm_model.pkl'")

    # 步骤6: 评估模型
    print("\n步骤6: 评估模型性能...")
    performance = evaluate_model(best_elm, X_test, y_test)

    # 保存性能报告
    pd.Series(performance).to_csv('elm_model_performance.csv')
    print("模型性能已保存到 'elm_model_performance.csv'")

    # 步骤7: 特征重要性分析
    print("\n步骤7: 分析特征重要性...")
    # 获取特征名称
    cat_encoder = preprocessor.named_transformers_['cat']
    categorical_features = cat_encoder.get_feature_names_out(
        ['material', 'coating', 'corrosion_type']
    )
    numeric_features = [
        'temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
        'O2_content', 'CO2_conc', 'H2S_conc', 'pipe_age',
        'temp_cl_index', 'flow_pressure_ratio',
        'corrosion_aggressiveness', 'age_degradation'
    ]
    all_feature_names = np.concatenate([numeric_features, categorical_features])

    importance_df = analyze_feature_importance(best_elm, preprocessor, all_feature_names)

    # 保存重要特征
    importance_df.to_csv('output/elm_feature_importance.csv', index=False)
    print("特征重要性已保存到 output/elm_feature_importance.csv")

    # 步骤8: 保存预处理器
    print("\n步骤8: 保存预处理器...")
    joblib.dump(preprocessor, 'corrosion_preprocessor.pkl')
    print("预处理器已保存")

    # 步骤9: 创建预测器示例
    print("\n步骤9: 创建预测器示例...")
    predictor = CorrosionPredictor()

    # 示例预测
    sample_input = {
        'temperature': 85.3,
        'pressure': 42.7,
        'flow_rate': 2.8,
        'Cl_conc': 25300,
        'pH': 6.2,
        'O2_content': 4.1,
        'CO2_conc': 120,
        'H2S_conc': 8.5,
        'pipe_age': 12,
        'material': 'carbon_steel',
        'coating': 'epoxy',
        'corrosion_type': 'pitting'
    }

    prediction = predictor.predict(sample_input)
    print(f"\n示例预测结果:")
    print(f"输入参数: {sample_input}")
    print(f"预测腐蚀速率: {prediction:.6f} mm/year")

    # 步骤10: 与其他模型比较
    print("\n步骤10: 与其他模型比较...")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor

    models = {
        'ELM': best_elm,
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVM': SVR(kernel='rbf', C=10),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name != 'ELM':
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
        else:
            train_time = "N/A"  # ELM训练时间已在之前记录

        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Train_Time': train_time,
            'Prediction_Time': pred_time
        }

    # 打印比较结果
    print("\n模型性能比较:")
    print(f"{'模型':<15} {'MAE':<10} {'RMSE':<10} {'训练时间(秒)':<15} {'预测时间(秒)':<15}")
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['MAE']:.6f} {metrics['RMSE']:.6f} "
              f"{str(metrics['Train_Time'])[:6]:<15} {metrics['Prediction_Time']:.6f}")

    print("\n管道腐蚀预测模型构建完成!")


if __name__ == "__main__":
    main()