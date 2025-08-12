import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 1. 数据生成函数 - 模拟管道腐蚀数据
def generate_corrosion_data(num_samples=5000):
    """
    生成模拟的管道腐蚀数据集
    参数:
        num_samples: 生成的样本数量
    返回:
        DataFrame: 包含特征和腐蚀速率的模拟数据
    """
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
    """
    执行数据探索性分析并生成可视化
    参数:
        df: 包含管道腐蚀数据的DataFrame
    """
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
    """
    创建具有物理意义的特征组合
    参数:
        df: 原始数据DataFrame
    返回:
        DataFrame: 包含新特征的数据
    """
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

    # 腐蚀类型与材质交互
    df['material_corrosion_interaction'] = df['material'].astype('category').cat.codes * df['corrosion_type'].astype(
        'category').cat.codes

    return df


# 4. 数据预处理
def preprocess_data(df):
    """
    预处理数据：分割、转换、缩放
    参数:
        df: 包含特征和目标的数据DataFrame
    返回:
        处理后的训练和测试数据
    """
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

    # 获取特征数量
    num_features = X_train_preprocessed.shape[1]

    # 重塑为3D格式 (样本数, 时间步长=1, 特征数)
    X_train_reshaped = X_train_preprocessed.reshape(-1, 1, num_features)
    X_test_reshaped = X_test_preprocessed.reshape(-1, 1, num_features)

    print(f"训练数据形状: {X_train_reshaped.shape}")
    print(f"测试数据形状: {X_test_reshaped.shape}")

    return X_train_reshaped, X_test_reshaped, y_train, y_test, preprocessor


# 5. 构建1D CNN模型
def build_cnn_model(input_shape):
    """
    构建用于腐蚀预测的1D CNN模型
    参数:
        input_shape: 输入数据的形状
    返回:
        编译好的Keras模型
    """
    model = Sequential(name="PipeCorrosionPredictor")

    # 输入层
    model.add(Conv1D(
        filters=64,
        kernel_size=1,  # 使用1x1卷积进行特征变换
        activation='swish',
        input_shape=input_shape,
        kernel_regularizer=l2(1e-4),
        name="conv1d_input"
    ))
    model.add(BatchNormalization(name="bn1"))

    # 特征提取层
    model.add(Conv1D(
        filters=128,
        kernel_size=1,
        activation='swish',
        dilation_rate=2,  # 空洞卷积增加感受野
        kernel_regularizer=l2(1e-4),
        name="conv1d_dilated"
    ))
    model.add(BatchNormalization(name="bn2"))
    model.add(MaxPooling1D(pool_size=1, name="maxpool1"))
    model.add(Dropout(0.3, name="dropout1"))

    # 深度特征提取
    model.add(Conv1D(
        filters=256,
        kernel_size=1,
        activation='swish',
        kernel_regularizer=l2(1e-4),
        name="conv1d_deep1"
    ))
    model.add(BatchNormalization(name="bn3"))
    model.add(Conv1D(
        filters=256,
        kernel_size=1,
        activation='swish',
        kernel_regularizer=l2(1e-4),
        name="conv1d_deep2"
    ))
    model.add(BatchNormalization(name="bn4"))
    model.add(GlobalAveragePooling1D(name="global_avg_pool"))

    # 全连接层
    model.add(Dense(256, activation='swish', kernel_regularizer=l2(1e-4), name="dense1"))
    model.add(Dropout(0.4, name="dropout2"))
    model.add(Dense(128, activation='swish', kernel_regularizer=l2(1e-4), name="dense2"))
    model.add(Dense(64, activation='swish', kernel_regularizer=l2(1e-4), name="dense3"))

    # 输出层 - 腐蚀速率预测
    model.add(Dense(1, activation='linear', name="output"))

    # 编译模型
    optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(
        optimizer=optimizer,
        loss='huber_loss',  # 对异常值鲁棒的损失函数
        metrics=['mae', 'mse']
    )

    return model


# 6. 模型训练
def train_model(model, X_train, y_train, X_val, y_val):
    """
    训练CNN模型并返回训练历史
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
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='best_corrosion_cnn_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=callbacks
    )

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
        plt.scatter(y_test, y_pred, alpha=0.5)
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
    plt.scatter(y_pred, residuals, alpha=0.5)
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
        'Mean_Absolute_Percentage_Error': np.mean(np.abs(errors / y_test)) * 100
    }

    return performance_report


# 8. 特征重要性分析
def analyze_feature_importance(model, preprocessor, X_sample, feature_names):
    """
    分析特征重要性
    参数:
        model: 训练好的Keras模型
        preprocessor: 数据预处理器
        X_sample: 样本数据
        feature_names: 特征名称列表
    返回:
        特征重要性DataFrame
    """
    # 预处理样本数据
    X_sample_preprocessed = preprocessor.transform(X_sample)
    X_sample_reshaped = X_sample_preprocessed.reshape(-1, 1, X_sample_preprocessed.shape[1])

    # 使用梯度计算特征重要性
    input_tensor = tf.convert_to_tensor(X_sample_reshaped, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)

    # 计算梯度
    grads = tape.gradient(predictions, input_tensor)
    abs_grads = tf.abs(grads)
    avg_grads = tf.reduce_mean(abs_grads, axis=[0, 1])  # 跨样本和时间步平均

    # 转换为numpy并归一化
    importances = avg_grads.numpy().flatten()
    importances /= np.sum(importances)

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature',
                data=importance_df.head(15).sort_values('Importance', ascending=True))
    plt.title('Top 15 腐蚀预测特征重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

    return importance_df


# 9. 模型部署函数
class CorrosionPredictor:
    """
    管道腐蚀预测器类，用于加载模型并进行预测
    """

    def __init__(self, model_path='best_corrosion_cnn_model.h5',
                 preprocessor_path='corrosion_preprocessor.pkl'):
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
        """
        预测管道腐蚀速率
        参数:
            input_data: 包含输入特征的DataFrame或字典
        返回:
            腐蚀速率预测值 (mm/year)
        """
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
        reshaped_data = processed_data.reshape(-1, 1, processed_data.shape[1])

        # 预测
        predictions = self.model.predict(reshaped_data).flatten()

        # 添加安全边界（腐蚀速率不可能为负）
        predictions = np.maximum(predictions, 0.001)

        return predictions[0] if len(predictions) == 1 else predictions


# 主函数
def main():
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    print("=" * 50)
    print("管道腐蚀预测模型 - CNN实现")
    print("=" * 50)

    # 步骤1: 生成模拟数据
    print("\n步骤1: 生成模拟管道腐蚀数据...")
    corrosion_data = generate_corrosion_data(8000)
    print(f"生成数据完成: {corrosion_data.shape[0]} 个样本")

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

    # 进一步划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    # 步骤5: 构建模型
    print("\n步骤5: 构建CNN模型...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    model.summary()

    # 步骤6: 训练模型
    print("\n步骤6: 训练模型...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 步骤7: 评估模型
    print("\n步骤7: 评估模型性能...")
    # 加载最佳模型
    best_model = load_model('best_corrosion_cnn_model.h5')
    performance = evaluate_model(best_model, X_test, y_test, history)

    # 步骤8: 特征重要性分析
    print("\n步骤8: 分析特征重要性...")
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

    # 随机采样100个测试样本
    sample_idx = np.random.choice(len(X_test), 100, replace=False)
    X_sample = corrosion_data.drop('corrosion_rate', axis=1).iloc[sample_idx]

    importance_df = analyze_feature_importance(best_model, preprocessor, X_sample, all_feature_names)

    # 保存重要特征
    importance_df.to_csv('output/feature_importance.csv', index=False)
    print("特征重要性已保存到 output/feature_importance.csv")

    # 步骤9: 保存预处理器和模型
    print("\n步骤9: 保存模型和预处理器...")
    joblib.dump(preprocessor, 'corrosion_preprocessor.pkl')
    best_model.save('corrosion_cnn_model_final.h5')
    print("模型和预处理器已保存")

    # 步骤10: 创建预测器示例
    print("\n步骤10: 创建预测器示例...")
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

    print("\n管道腐蚀预测模型构建完成!")


if __name__ == "__main__":
    main()