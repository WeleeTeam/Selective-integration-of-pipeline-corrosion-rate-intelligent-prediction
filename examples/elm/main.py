import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from examples.data.utils import ensure_tabular_csv, load_tabular_csv, load_corrosion_dataset
from pathlib import Path


class SimpleELMRegressor:
    """极限学习机(ELM)回归的简易实现，单隐层随机特征 + 岭回归输出。

    参考思想：
    - 隐层权重随机固定，不参与梯度训练
    - 输出层通过最小二乘/岭回归一次性闭式求解
    """

    def __init__(self, hidden_units: int = 256, activation: str = "relu", alpha: float = 1e-2, random_state: int = 42):
        self.hidden_units = hidden_units
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state
        self.hidden_weights = None
        self.hidden_bias = None
        self.beta = None
        self.scaler = StandardScaler()

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0.0, x)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def fit(self, features: np.ndarray, targets: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        features_scaled = self.scaler.fit_transform(features)
        n_features = features_scaled.shape[1]

        self.hidden_weights = rng.normal(loc=0.0, scale=1.0, size=(n_features, self.hidden_units))
        self.hidden_bias = rng.normal(loc=0.0, scale=1.0, size=(self.hidden_units,))

        hidden_linear = features_scaled @ self.hidden_weights + self.hidden_bias
        hidden_outputs = self._activate(hidden_linear)

        # 岭回归闭式解 beta = (H^T H + alpha I)^(-1) H^T y
        hth = hidden_outputs.T @ hidden_outputs
        reg = self.alpha * np.eye(hth.shape[0])
        self.beta = np.linalg.solve(hth + reg, hidden_outputs.T @ targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        features_scaled = self.scaler.transform(features)
        hidden_linear = features_scaled @ self.hidden_weights + self.hidden_bias
        hidden_outputs = self._activate(hidden_linear)
        return hidden_outputs @ self.beta


def main():
    # 优先使用真实腐蚀数据
    try:
        from sklearn.pipeline import Pipeline
        csv_path = Path(__file__).resolve().parents[1] / "corrosion_data.csv"
        x_train, x_test, y_train, y_test, preprocessor = load_corrosion_dataset(str(csv_path))

        # 使用预处理将类别与数值特征转换为数值矩阵，再喂入ELM
        # 先拟合预处理，再取 transform 后的数组给 ELM
        preprocessor.fit(x_train)
        x_train_transformed = preprocessor.transform(x_train)
        x_test_transformed = preprocessor.transform(x_test)

        model = SimpleELMRegressor(hidden_units=256, activation="relu", alpha=1e-1, random_state=42)
        model.fit(x_train_transformed, y_train)
        predictions = model.predict(x_test_transformed)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("极限学习机(ELM) - 腐蚀数据集")
        print(f"测试集 MSE: {mse:.4f}")
        print(f"测试集 R2 : {r2:.4f}")
        return
    except Exception as e:
        print(f"使用腐蚀数据失败，改用合成数据。原因: {e}")

    csv_path = ensure_tabular_csv("examples/data/regression_train.csv")
    features, targets = load_tabular_csv(csv_path)
    features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    model = SimpleELMRegressor(hidden_units=256, activation="relu", alpha=1e-1, random_state=42)
    model.fit(features_train, targets_train)
    predictions = model.predict(features_test)

    mse = mean_squared_error(targets_test, predictions)
    r2 = r2_score(targets_test, predictions)

    print("极限学习机(ELM) 回归示例")
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R2 : {r2:.4f}")


if __name__ == "__main__":
    main()

