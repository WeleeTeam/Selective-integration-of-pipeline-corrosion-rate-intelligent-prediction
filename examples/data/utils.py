import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.datasets import make_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def ensure_directory(directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def ensure_tabular_csv(csv_path: str, n_samples: int = 3000, n_features: int = 25, noise: float = 10.0, random_state: int = 42) -> str:
    """若 csv 不存在则生成一个用于回归的表格数据集并保存。返回 csv 路径。"""
    directory = os.path.dirname(csv_path)
    ensure_directory(directory)
    if not os.path.isfile(csv_path):
        features, targets = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.6),
            noise=noise,
            random_state=random_state,
        )
        columns = [f"x{i+1}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=columns)
        df["y"] = targets
        df.to_csv(csv_path, index=False)
    return csv_path


def load_tabular_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    features = df.drop(columns=["y"]).values
    targets = df["y"].values
    return features, targets


# -------------------- Corrosion dataset helpers --------------------

def corrosion_columns(csv_path: str) -> Tuple[List[str], List[str], str]:
    """根据已知列名划分数值列、类别列与目标列。
    目标列: corrosion_rate
    数值列: temperature, pressure, flow_rate, Cl_conc, pH, O2_content, CO2_conc, H2S_conc, pipe_age
    类别列: material, coating, corrosion_type
    若列名不完整，则回退为基于 dtype 的自动推断。
    """
    df_head = pd.read_csv(csv_path, nrows=5)
    if "corrosion_rate" not in df_head.columns:
        raise ValueError("数据集中未找到目标列 'corrosion_rate'")
    target_column = "corrosion_rate"

    expected_numeric = [
        "temperature",
        "pressure",
        "flow_rate",
        "Cl_conc",
        "pH",
        "O2_content",
        "CO2_conc",
        "H2S_conc",
        "pipe_age",
    ]
    expected_categorical = ["material", "coating", "corrosion_type"]

    if all(col in df_head.columns for col in expected_numeric + expected_categorical + [target_column]):
        numeric_columns = expected_numeric
        categorical_columns = expected_categorical
    else:
        # 回退：基于 dtype 推断
        categorical_columns = [c for c in df_head.columns if df_head[c].dtype == "object" and c != target_column]
        numeric_columns = [c for c in df_head.columns if c not in categorical_columns + [target_column]]

    return numeric_columns, categorical_columns, target_column


def load_corrosion_dataset(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    从 examples/corrosion_data.csv 读取腐蚀表格数据，构建适用于 scikit-learn 的预处理：
    - 数值列: StandardScaler
    - 类别列: OneHotEncoder(handle_unknown='ignore')
    返回: X_train, X_test, y_train, y_test, preprocessor
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    numeric_columns, categorical_columns, target_column = corrosion_columns(csv_path)
    # 强制类别列为字符串，避免被误解析为分类编码数字
    for c in categorical_columns:
        if c in df.columns:
            df[c] = df[c].astype(str)
    X = df[numeric_columns + categorical_columns]
    y = df[target_column].values

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_columns),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, preprocessor


def ensure_time_series_npz(npz_path: str, n_samples: int = 5000, seq_len: int = 60, noise: float = 0.05, random_state: int = 7) -> str:
    """若 npz 不存在则生成单变量时序数据集：输入为长度 seq_len 的窗口，目标为下一步值。"""
    directory = os.path.dirname(npz_path)
    ensure_directory(directory)
    if not os.path.isfile(npz_path):
        rng = np.random.default_rng(random_state)
        x = np.linspace(0, 300, n_samples + seq_len + 1)
        signal = np.sin(x) + 0.3 * np.sin(3 * x) + 0.2 * np.cos(0.7 * x)
        signal += noise * rng.normal(size=signal.shape)
        sequences = []
        targets = []
        for i in range(n_samples):
            sequences.append(signal[i : i + seq_len])
            targets.append(signal[i + seq_len])
        sequences = np.array(sequences, dtype=np.float32)[..., None]
        targets = np.array(targets, dtype=np.float32)
        num_train = int(0.8 * len(sequences))
        np.savez_compressed(
            npz_path,
            x_train=sequences[:num_train],
            y_train=targets[:num_train],
            x_test=sequences[num_train:],
            y_test=targets[num_train:],
            seq_len=seq_len,
        )
    return npz_path


def load_time_series_npz(npz_path: str):
    data = np.load(npz_path)
    return (
        data["x_train"],
        data["y_train"],
        data["x_test"],
        data["y_test"],
        int(data["seq_len"]),
    )

