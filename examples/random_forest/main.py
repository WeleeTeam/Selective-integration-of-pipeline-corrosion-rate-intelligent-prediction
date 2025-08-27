from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from examples.data.utils import ensure_tabular_csv, load_tabular_csv, load_corrosion_dataset
from pathlib import Path


def main():
    # 优先使用真实腐蚀数据
    try:
        csv_path = Path(__file__).resolve().parents[1] / "corrosion_data.csv"
        x_train, x_test, y_train, y_test, preprocessor = load_corrosion_dataset(str(csv_path))
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
        ])
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("随机森林 - 腐蚀数据集")
        print(f"测试集 MSE: {mse:.4f}")
        print(f"测试集 R2 : {r2:.4f}")
        return
    except Exception as e:
        print(f"使用腐蚀数据失败，改用合成数据。原因: {e}")

    csv_path = ensure_tabular_csv("examples/data/regression_train.csv")
    features, targets = load_tabular_csv(csv_path)

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(features_train, targets_train)
    predictions = model.predict(features_test)
    mse = mean_squared_error(targets_test, predictions)
    r2 = r2_score(targets_test, predictions)

    print("随机森林 回归示例")
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R2 : {r2:.4f}")


if __name__ == "__main__":
    main()

