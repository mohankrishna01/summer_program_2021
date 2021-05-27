import joblib
mind = joblib.load("model_task1.pk1")
print(mind.predict([[11]]))
