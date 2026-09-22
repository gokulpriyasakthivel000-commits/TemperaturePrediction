import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv("weather.csv")


X = data[["Humidity", "Pressure", "WindSpeed", "CloudCover"]]


y = data["Temperature"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as model.pkl")