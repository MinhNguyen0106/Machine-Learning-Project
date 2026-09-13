import joblib
import numpy as np

# Tải các mô hình Bagging, Boosting và Stacking đã lưu
models = joblib.load("bagging_model.pkl")

# Căn nhà mới cần định giá: Diện tích 70m2, 3 phòng ngủ
nha_moi = np.array([[70, 3]])

for method, model in models.items():
    gia_du_doan = model.predict(nha_moi)
    print(
        f"{method.title()} prediction (70m2, 3 bedrooms): "
        f"{gia_du_doan[0]:.2f} billion VND"
    )