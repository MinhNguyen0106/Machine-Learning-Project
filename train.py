import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge


DATA_FILE = Path(__file__).parent / "data" / "houses.csv"
MODEL_FILE = Path(__file__).parent / "bagging_model.pkl"
FEATURE_COLUMNS = ("dientich", "so_phong_ngu")
TARGET_COLUMN = "gia_ty"


def load_training_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc feature và giá nhà từ file CSV."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
        missing_columns = required_columns - set(fieldnames)
        if missing_columns:
            raise ValueError(
                f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing_columns))}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("CSV không có dữ liệu huấn luyện.")

    try:
        values = [
            [float(row[column]) for column in (*FEATURE_COLUMNS, TARGET_COLUMN)]
            for row in rows
        ]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Các cột dientich, so_phong_ngu và gia_ty phải chứa số."
        ) from error

    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Dữ liệu CSV không được chứa NaN hoặc vô cực.")
    if (data[:, 0] <= 0).any() or (data[:, 1] < 0).any() or (data[:, 2] < 0).any():
        raise ValueError(
            "Diện tích phải lớn hơn 0; số phòng ngủ và giá phải không âm."
        )

    return data[:, :2], data[:, 2]


X, y = load_training_data(DATA_FILE)

# Bagging: nhiều cây quyết định được huấn luyện song song rồi lấy trung bình.
bagging_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Boosting: các cây được huấn luyện tuần tự để sửa lỗi của các cây trước.
boosting_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    random_state=42,
)

# Stacking: dùng dự đoán của các model cơ sở làm đầu vào cho model cuối.
stacking_model = StackingRegressor(
    estimators=[
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        (
            "gradient_boosting",
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
        ),
    ],
    final_estimator=Ridge(),
    cv=5,
)

models = {
    "bagging": bagging_model,
    "boosting": boosting_model,
    "stacking": stacking_model,
}

for model in models.values():
    model.fit(X, y)

# Lưu cả ba model trong cùng một file để API có thể lựa chọn phương pháp.
joblib.dump(models, MODEL_FILE)
print(f"Training completed. Model saved to '{MODEL_FILE.name}'.")