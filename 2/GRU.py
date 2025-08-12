import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)


# 1. 数据生成函数 - 模拟管道腐蚀数据（含时间序列）
def generate_corrosion_data(num_pipes=200, time_steps=100):
    """
    生成模拟的管道腐蚀数据集（含时间序列）
    参数:
        num_pipes: 管道数量
        time_steps: 每个管道的时间步长
    返回:
        DataFrame: 包含特征和腐蚀速率的模拟数据
    """
    np.random.seed(42)
    data = {
        'pipe_id': np.repeat(np.arange(num_pipes), time_steps),
        'time_step': np.tile(np.arange(time_steps), num_pipes),
        'temperature': np.zeros(num_pipes * time_steps),
        'pressure': np.zeros(num_pipes * time_steps),
        'flow_rate': np.zeros(num_pipes * time_steps),
        'Cl_conc': np.zeros(num_pipes * time_steps),
        'pH': np.zeros(num_pipes * time_steps),
        'O2_content': np.zeros(num_pipes * time_steps),
        'CO2_conc': np.zeros(num_pipes * time_steps),
        'H2S_conc': np.zeros(num_pipes * time_steps),
        'material': np.zeros(num_pipes * time_steps, dtype=object),
        'coating': np.zeros(num_pipes * time_steps, dtype=object),
        'corrosion_rate': np.zeros(num_pipes * time_steps)
    }

    # 为每个管道生成静态属性
    materials = ['carbon_steel', 'stainless_steel', 'alloy_steel', 'copper_alloy']
    coatings = ['none', 'epoxy', 'polyethylene', 'fusion_bonded']

    for pipe_id in range(num_pipes):
        start_idx = pipe_id * time_steps
        end_idx = (pipe_id + 1) * time_steps

        # 随机生成静态属性
        material = np.random.choice(materials, p=[0.6, 0.2, 0.15, 0.05])
        coating = np.random.choice(coatings, p=[0.3, 0.4, 0.2, 0.1])
        data['material'][start_idx:end_idx] = material
        data['coating'][start_idx:end_idx] = coating

        # 生成基础环境参数（随时间变化）
        base_temp = np.random.uniform(40, 80)
        temp_trend = np.linspace(0, np.random.uniform(-5, 5), time_steps)
        data['temperature'][start_idx:end_idx] = base_temp + temp_trend + np.random.normal(0, 2, time_steps)

        base_pressure = np.random.uniform(20, 60)
        pressure_trend = np.linspace(0, np.random.uniform(-3, 3), time_steps)
        data['pressure'][start_idx:end_idx] = base_pressure + pressure_trend + np.random.normal(0, 1, time_steps)

        base_flow = np.random.uniform(0.5, 3.0)
        flow_trend = np.linspace(0, np.random.uniform(-0.5, 0.5), time_steps)
        data['flow_rate'][start_idx:end_idx] = base_flow + flow_trend + np.random.normal(0, 0.1, time_steps)

        base_cl = np.random.uniform(5000, 30000)
        cl_trend = np.linspace(0, np.random.uniform(-1000, 3000), time_steps)
        data['Cl_conc'][start_idx:end_idx] = base_cl + cl_trend + np.random.normal(0, 500, time_steps)

        # 其他参数
        data['pH'][start_idx:end_idx] = np.random.uniform(5.5, 8.5, time_steps)
        data['O2_content'][start_idx:end_idx] = np.random.uniform(0.5, 5.0, time_steps)
        data['CO2_conc'][start_idx:end_idx] = np.random.uniform(50, 300, time_steps)
        data['H2S_conc'][start_idx:end_idx] = np.random.uniform(0, 20, time_steps)

        # 基于物理模型模拟腐蚀速率
        base_rate = 0.02
        temp_effect = (data['temperature'][start_idx:end_idx] - 40) * 0.001
        cl_effect = data['Cl_conc'][start_idx:end_idx] * 0.000001
        flow_effect = data['flow_rate'][start_idx:end_idx] * 0.01

        # 材质影响
        if material == 'carbon_steel':
            material_factor = 1.5
        elif material == 'stainless_steel':
            material_factor = 0.2
        elif material == 'alloy_steel':
            material_factor = 0.8
        else:  # copper_alloy
            material_factor = 0.5

        # 涂层影响
        if coating == 'none':
            coating_factor = 1.0
        elif coating == 'epoxy':
            coating_factor = 0.6
        elif coating == 'polyethylene':
            coating_factor = 0.4
        else:  # fusion_bonded
            coating_factor = 0.3

        # 腐蚀积累效应
        corrosion_accumulation = np.cumsum(temp_effect + cl_effect + flow_effect) * 0.01

        # 计算腐蚀速率
        data['corrosion_rate'][start_idx:end_idx] = (
                (base_rate + temp_effect + cl_effect + flow_effect) *
                material_factor * coating_factor + corrosion_accumulation +
                np.random.normal(0, 0.005, time_steps)  # 添加随机噪声
        )

    return pd.DataFrame(data)


# 2. 数据探索性分析
def explore_data(df):
    """执行数据探索性分析并生成可视化"""
    print("数据集信息:")
    print(f"样本数量: {df.shape[0]}")
    print(f"特征数量: {df.shape[1] - 1} (包括目标变量腐蚀速率)")
    print(f"管道数量: {df['pipe_id'].nunique()}")
    print(f"时间步长: {df['time_step'].max() + 1}")

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
                          'O2_content', 'CO2_conc', 'H2S_conc', 'corrosion_rate']

    corr_matrix = df[numerical_features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('数值特征相关性矩阵')
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

    # 类别特征分布
    categorical_features = ['material', 'coating']
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, feature in enumerate(categorical_features):
        sns.countplot(data=df, x=feature, ax=axes[i])
        axes[i].set_title(f'{feature} 分布')
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('categorical_distribution.png', dpi=300)
    plt.show()

    # 随机选择几个管道展示腐蚀速率随时间变化
    sample_pipes = np.random.choice(df['pipe_id'].unique(), 3, replace=False)
    plt.figure(figsize=(12, 8))
    for pipe_id in sample_pipes:
        pipe_data = df[df['pipe_id'] == pipe_id]
        plt.plot(pipe_data['time_step'], pipe_data['corrosion_rate'],
                 label=f'Pipe {pipe_id} ({pipe_data["material"].iloc[0]})')

    plt.title('腐蚀速率随时间变化示例')
    plt.xlabel('时间步长')
    plt.ylabel('腐蚀速率 (mm/year)')
    plt.legend()
    plt.grid(True)
    plt.savefig('corrosion_rate_over_time.png', dpi=300)
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

    # 滞后特征（前3个时间步的腐蚀速率）
    df['corrosion_rate_lag1'] = df.groupby('pipe_id')['corrosion_rate'].shift(1)
    df['corrosion_rate_lag2'] = df.groupby('pipe_id')['corrosion_rate'].shift(2)
    df['corrosion_rate_lag3'] = df.groupby('pipe_id')['corrosion_rate'].shift(3)

    # 移动平均特征
    df['corrosion_rate_ma3'] = df.groupby('pipe_id')['corrosion_rate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # 填充缺失的滞后值
    df.fillna(method='bfill', inplace=True)

    return df


# 4. 数据预处理
def preprocess_data(df, time_steps=10):
    """
    预处理数据：分割、转换、缩放
    参数:
        df: 包含特征和目标的数据DataFrame
        time_steps: 时间序列长度
    返回:
        处理后的训练和测试数据
    """
    # 分离特征和目标
    X = df.drop(['corrosion_rate', 'pipe_id', 'time_step'], axis=1)
    y = df['corrosion_rate'].values

    # 定义数值和类别特征
    numerical_features = [
        'temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
        'O2_content', 'CO2_conc', 'H2S_conc',
        'temp_cl_index', 'flow_pressure_ratio',
        'corrosion_aggressiveness',
        'corrosion_rate_lag1', 'corrosion_rate_lag2', 'corrosion_rate_lag3',
        'corrosion_rate_ma3'
    ]

    categorical_features = ['material', 'coating']

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'  # 忽略其他列
    )

    # 应用预处理
    X_preprocessed = preprocessor.fit_transform(X)

    # 创建时间序列数据集
    def create_sequences(data, targets, pipe_ids, seq_length):
        sequences = []
        target_seq = []
        pipe_seq = []

        for pipe_id in np.unique(pipe_ids):
            pipe_indices = np.where(pipe_ids == pipe_id)[0]
            pipe_data = data[pipe_indices]
            pipe_targets = targets[pipe_indices]

            for i in range(len(pipe_data) - seq_length):
                sequences.append(pipe_data[i:i + seq_length])
                target_seq.append(pipe_targets[i + seq_length])
                pipe_seq.append(pipe_id)

        return np.array(sequences), np.array(target_seq), np.array(pipe_seq)

    # 创建时间序列数据
    sequences, targets, pipe_ids = create_sequences(
        X_preprocessed, y, df['pipe_id'].values, time_steps
    )

    print(f"时间序列数据形状: {sequences.shape}")
    print(f"目标变量形状: {targets.shape}")

    # 划分训练集和测试集（按管道划分，避免数据泄露）
    unique_pipes = np.unique(pipe_ids)
    train_pipes, test_pipes = train_test_split(
        unique_pipes, test_size=0.2, random_state=42
    )

    train_indices = np.where(np.isin(pipe_ids, train_pipes))[0]
    test_indices = np.where(np.isin(pipe_ids, test_pipes))[0]

    X_train = sequences[train_indices]
    y_train = targets[train_indices]
    X_test = sequences[test_indices]
    y_test = targets[test_indices]

    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor


# 5. 构建GRU模型
def build_gru_model(input_shape, units=128, dropout_rate=0.3, learning_rate=0.001):
    """
    构建用于腐蚀预测的GRU模型
    参数:
        input_shape: 输入数据的形状 (时间步长, 特征数)
        units: GRU单元数量
        dropout_rate: Dropout比率
        learning_rate: 学习率
    返回:
        编译好的Keras模型
    """
    model = Sequential(name="PipeCorrosionGRUPredictor")

    # 输入层
    model.add(GRU(
        units=units,
        input_shape=input_shape,
        return_sequences=True,  # 返回完整序列以供下一层使用
        kernel_regularizer=l2(1e-4),
        name="gru1"
    ))
    model.add(BatchNormalization(name="bn1"))
    model.add(Dropout(dropout_rate, name="dropout1"))

    # 第二GRU层
    model.add(GRU(
        units=units // 2,
        return_sequences=False,  # 只返回最后一个输出
        kernel_regularizer=l2(1e-4),
        name="gru2"
    ))
    model.add(BatchNormalization(name="bn2"))
    model.add(Dropout(dropout_rate, name="dropout2"))

    # 全连接层
    model.add(Dense(units // 2, activation='relu', kernel_regularizer=l2(1e-4), name="dense1"))
    model.add(Dropout(dropout_rate * 0.8, name="dropout3"))
    model.add(Dense(units // 4, activation='relu', kernel_regularizer=l2(1e-4), name="dense2"))

    # 输出层 - 腐蚀速率预测
    model.add(Dense(1, activation='linear', name="output"))

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)
    model.compile(
        optimizer=optimizer,
        loss='huber_loss',  # 对异常值鲁棒的损失函数
        metrics=['mae', 'mse']
    )

    return model


# 6. 模型训练
def train_model(model, X_train, y_train, X_val, y_val):
    """
    训练GRU模型并返回训练历史
    参数:
        model: 编译好的Keras模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
    返回:
        训练历史对象
    """
    # 创建回调
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks
    )

    # 训练结束后保存最佳模型
    model.save('best_corrosion_gru_model.keras')
    print("已保存最佳模型为 'best_corrosion_gru_model.keras'")

    return history


# 7. 模型评估与可视化
def evaluate_model(model, X_test, y_test, history=None):
    """
    评估模型性能并生成可视化
    参数:
        model: 训练好的Keras模型
        X_test, y_test: 测试数据
        history: 训练历史对象（可选）
    """
    # 测试集预测
    y_pred = model.predict(X_test).flatten()

    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n模型评估结果:")
    print(f"测试集MAE: {mae:.6f} mm/year")
    print(f"测试集RMSE: {rmse:.6f} mm/year")
    print(f"测试集R²: {r2:.4f}")

    # 可视化训练过程
    if history:
        plt.figure(figsize=(15, 10))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('训练和验证损失')
        plt.ylabel('损失')
        plt.xlabel('Epoch')
        plt.legend()

        # MAE曲线
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.title('训练和验证MAE')
        plt.ylabel('MAE (mm/year)')
        plt.xlabel('Epoch')
        plt.legend()

        # 预测 vs 实际值
        plt.subplot(2, 2, 3)
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        plt.title('预测值 vs 实际值')
        plt.xlabel('实际腐蚀速率 (mm/year)')
        plt.ylabel('预测腐蚀速率 (mm/year)')

        # 误差分布
        plt.subplot(2, 2, 4)
        errors = y_pred - y_test
        sns.histplot(errors, kde=True, bins=30)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('预测误差分布')
        plt.xlabel('预测误差 (mm/year)')

        plt.tight_layout()
        plt.savefig('model_training_results.png', dpi=300)
        plt.show()

    # 残差图
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('预测残差图')
    plt.xlabel('预测腐蚀速率 (mm/year)')
    plt.ylabel('残差 (实际 - 预测)')
    plt.savefig('prediction_residuals.png', dpi=300)
    plt.show()

    # 模型性能报告
    performance_report = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Max_Error': np.max(np.abs(errors)),
        'Mean_Absolute_Percentage_Error': np.mean(np.abs(residuals / y_test)) * 100
    }

    return performance_report


# 8. 时间序列预测可视化
def visualize_time_series_predictions(model, df, preprocessor, pipe_id, time_steps=10):
    """
    可视化特定管道的腐蚀速率预测
    参数:
        model: 训练好的模型
        df: 原始数据
        preprocessor: 数据预处理器
        pipe_id: 要可视化的管道ID
        time_steps: 时间序列长度
    """
    # 获取特定管道的数据
    pipe_data = df[df['pipe_id'] == pipe_id].copy()

    # 应用特征工程和预处理
    pipe_data = feature_engineering(pipe_data)
    X_pipe = pipe_data.drop(['corrosion_rate', 'pipe_id', 'time_step'], axis=1)
    X_preprocessed = preprocessor.transform(X_pipe)

    # 创建时间序列
    sequences = []
    for i in range(len(X_preprocessed) - time_steps):
        sequences.append(X_preprocessed[i:i + time_steps])

    sequences = np.array(sequences)

    # 预测
    predictions = model.predict(sequences).flatten()

    # 实际值（与预测对应的时间点）
    actual_values = pipe_data['corrosion_rate'].values[time_steps:]

    # 可视化
    plt.figure(figsize=(14, 8))
    plt.plot(pipe_data['time_step'], pipe_data['corrosion_rate'], 'b-', label='实际腐蚀速率')

    # 预测值（在时间轴上偏移）
    prediction_timesteps = pipe_data['time_step'].values[time_steps:]
    plt.plot(prediction_timesteps, predictions, 'r--', label='预测腐蚀速率', linewidth=2)

    # 填充预测区间
    errors = predictions - actual_values
    std_error = np.std(errors)
    plt.fill_between(
        prediction_timesteps,
        predictions - 1.96 * std_error,
        predictions + 1.96 * std_error,
        color='r', alpha=0.1, label='95% 预测区间'
    )

    plt.title(f'管道 {pipe_id} 腐蚀速率预测 ({pipe_data["material"].iloc[0]}材质)')
    plt.xlabel('时间步长')
    plt.ylabel('腐蚀速率 (mm/year)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'corrosion_prediction_pipe_{pipe_id}.png', dpi=300)
    plt.show()


# 9. 模型部署函数
class CorrosionPredictor:
    """
    管道腐蚀预测器类，用于加载模型并进行预测
    """

    def __init__(self, model_path='best_corrosion_gru_model.keras',
                 preprocessor_path='corrosion_preprocessor.pkl',
                 time_steps=10):
        """
        初始化预测器，加载模型和预处理器
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"预处理器文件 {preprocessor_path} 不存在")

        # 加载模型和预处理器
        self.model = load_model(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.time_steps = time_steps

        # 获取特征名称
        self.numeric_features = [
            'temperature', 'pressure', 'flow_rate', 'Cl_conc', 'pH',
            'O2_content', 'CO2_conc', 'H2S_conc',
            'temp_cl_index', 'flow_pressure_ratio',
            'corrosion_aggressiveness',
            'corrosion_rate_lag1', 'corrosion_rate_lag2', 'corrosion_rate_lag3',
            'corrosion_rate_ma3'
        ]

        # 获取类别特征名称
        cat_encoder = self.preprocessor.named_transformers_['cat']
        self.categorical_features = cat_encoder.get_feature_names_out(
            ['material', 'coating']
        )

        self.all_feature_names = np.concatenate([self.numeric_features, self.categorical_features])

    def predict(self, historical_data):
        """
        预测管道腐蚀速率
        参数:
            historical_data: DataFrame包含历史数据（按时间顺序排列）
        """
        # 确保数据按时间顺序排列
        if not historical_data['time_step'].is_monotonic_increasing:
            historical_data = historical_data.sort_values('time_step')

        # 添加衍生特征
        historical_data = feature_engineering(historical_data)

        # 仅保留需要的特征
        features = historical_data[self.numeric_features + ['material', 'coating']]

        # 预处理
        processed_data = self.preprocessor.transform(features)

        # 检查是否有足够的历史数据
        if len(processed_data) < self.time_steps:
            raise ValueError(f"需要至少 {self.time_steps} 个时间步的历史数据，当前只有 {len(processed_data)}")

        # 创建序列（使用最近的time_steps个数据点）
        sequence = processed_data[-self.time_steps:].reshape(1, self.time_steps, -1)

        # 预测
        prediction = self.model.predict(sequence).flatten()[0]

        # 添加安全边界（腐蚀速率不可能为负）
        prediction = max(prediction, 0.001)

        return prediction


# 主函数
def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    print("=" * 50)
    print("管道腐蚀预测模型 - GRU实现")
    print("=" * 50)

    # 步骤1: 生成模拟数据
    print("\n步骤1: 生成模拟管道腐蚀数据（含时间序列）...")
    corrosion_data = generate_corrosion_data(num_pipes=300, time_steps=50)
    print(f"生成数据完成: {corrosion_data.shape[0]} 个样本")
    print(f"管道数量: {corrosion_data['pipe_id'].nunique()}")
    print(f"时间步长范围: 0-{corrosion_data['time_step'].max()}")

    # 步骤2: 数据探索性分析
    print("\n步骤2: 执行数据探索性分析...")
    explore_data(corrosion_data)

    # 步骤3: 特征工程
    print("\n步骤3: 执行特征工程...")
    corrosion_data = feature_engineering(corrosion_data)
    print("衍生特征已添加: temp_cl_index, flow_pressure_ratio, corrosion_aggressiveness, 滞后特征, 移动平均")

    # 步骤4: 数据预处理
    print("\n步骤4: 数据预处理...")
    time_steps = 15  # 使用15个时间步预测下一个时间步
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(corrosion_data, time_steps)

    # 进一步划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # 步骤5: 构建模型
    print("\n步骤5: 构建GRU模型...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_gru_model(input_shape, units=192, dropout_rate=0.25, learning_rate=0.0005)
    model.summary()

    # 步骤6: 训练模型
    print("\n步骤6: 训练模型...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 步骤7: 评估模型
    print("\n步骤7: 评估模型性能...")
    # 加载最佳模型
    best_model = load_model('best_corrosion_gru_model.keras')
    performance = evaluate_model(best_model, X_test, y_test, history)

    # 步骤8: 可视化时间序列预测
    print("\n步骤8: 可视化时间序列预测...")
    # 随机选择几个管道进行可视化
    sample_pipes = np.random.choice(corrosion_data['pipe_id'].unique(), 2, replace=False)
    for pipe_id in sample_pipes:
        visualize_time_series_predictions(best_model, corrosion_data, preprocessor, pipe_id, time_steps)

    # 步骤9: 保存预处理器和模型
    print("\n步骤9: 保存模型和预处理器...")
    joblib.dump(preprocessor, 'corrosion_preprocessor.pkl')
    best_model.save('corrosion_gru_model_final.keras')
    print("模型和预处理器已保存")

    # 步骤10: 创建预测器示例
    print("\n步骤10: 创建预测器示例...")
    predictor = CorrosionPredictor(time_steps=time_steps)

    # 示例预测 - 获取特定管道的历史数据
    pipe_id = sample_pipes[0]
    pipe_data = corrosion_data[corrosion_data['pipe_id'] == pipe_id]

    # 使用前time_steps+10个数据点进行预测
    historical_data = pipe_data.head(time_steps + 10)

    # 预测下一个时间步的腐蚀速率
    prediction = predictor.predict(historical_data)

    # 实际值（下一个时间步）
    actual_value = pipe_data.iloc[time_steps + 10]['corrosion_rate']

    print(f"\n示例预测结果 (管道 {pipe_id}):")
    print(f"时间步 {time_steps + 10} 的预测腐蚀速率: {prediction:.6f} mm/year")
    print(f"时间步 {time_steps + 10} 的实际腐蚀速率: {actual_value:.6f} mm/year")
    print(f"预测误差: {abs(prediction - actual_value):.6f} mm/year")

    print("\n管道腐蚀预测模型构建完成!")


if __name__ == "__main__":
    main()