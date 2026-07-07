import joblib

encoders = joblib.load("Python/models/dashboard_encoders.pkl")

for column, encoder in encoders.items():
    print("=" * 40)
    print(column)
    print(list(encoder.classes_))