from pandas import read_csv
ds = read_csv("/root/SalaryData.csv")
x = ds["YearsExperience"]
y = ds["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
x_train = x_train.values.reshape(24,1)
x_test = x_test.values.reshape(6,1)

from sklearn.linear_model import LinearRegression
mind = LinearRegression()
mind.fit(x_train, y_train)
pred_data = mind.predict(x_test)

print("Predicted Salary : ", pred_data)

import joblib
joblib.dump(mind, "model_task1.pk1")
print("Model saved successfully")
