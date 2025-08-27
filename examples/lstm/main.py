import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from examples.data.utils import ensure_time_series_npz, load_time_series_npz


def make_sine_dataset(n_samples: int = 5000, seq_len: int = 30, noise: float = 0.05):
    rng = np.random.default_rng(42)
    x = np.linspace(0, 200, n_samples + seq_len + 1)
    signal = np.sin(x) + 0.5 * np.sin(0.5 * x)
    signal += noise * rng.normal(size=signal.shape)

    sequences = []
    targets = []
    for i in range(n_samples):
        sequences.append(signal[i : i + seq_len])
        targets.append(signal[i + seq_len])
    sequences = np.array(sequences, dtype=np.float32)[..., None]
    targets = np.array(targets, dtype=np.float32)
    return sequences, targets


def build_model(seq_len: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, 1)),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main():
    npz_path = ensure_time_series_npz("examples/data/ts_dataset.npz", seq_len=30)
    x_train, y_train, x_test, y_test, seq_len = load_time_series_npz(npz_path)

    model = build_model(int(seq_len))
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)
    test_loss = model.evaluate(x_test, y_test, verbose=0)
    preds = model.predict(x_test[:5], verbose=0).flatten()

    print("长短期记忆网络(LSTM) 时序回归示例")
    print(f"测试集 MSE: {test_loss:.6f}")
    print("示例预测:", preds)


if __name__ == "__main__":
    main()

