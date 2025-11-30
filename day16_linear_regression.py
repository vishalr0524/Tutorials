import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


IMG_FOLDER = "/home/hp/Documents/Daily_Task/Day_2/Assets/images"  
OUTPUT_CSV = "/home/hp/Documents/Daily_Task/Day_2/Assets/images/dataset.csv"

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None 

    B_mean = np.mean(img[:, :, 0])
    G_mean = np.mean(img[:, :, 1])
    R_mean = np.mean(img[:, :, 2])


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

    return [R_mean, G_mean, B_mean, area]

# data = []
# files = os.listdir(IMG_FOLDER)

# for f in files:
#     path = os.path.join(IMG_FOLDER, f)
#     if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
#         result = extract_features(path)
#         if result:
#             data.append(result)

# df = pd.DataFrame(data, columns=["R_mean", "G_mean", "B_mean", "Area"])
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"Dataset saved as {OUTPUT_CSV}")
# print(df.head())

df = pd.read_csv(OUTPUT_CSV)
print(df.head())

X = df[["Area"]]       # feature
y = df["R_mean"]       # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- Model Evaluation -----")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)
print("----------------------------")

sample_area = X_train.iloc[0][0]          # pick Area value
true_rmean = y_train.iloc[0]              # true R_mean
predicted_rmean = model.predict([[sample_area]])[0]

print(f"Sample Area value: {sample_area}")
print(f"True R_mean      : {true_rmean}")
print(f"Predicted R_mean : {predicted_rmean}")

plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, y_pred, label="Predicted", linewidth=2)
plt.xlabel("Area")
plt.ylabel("R_mean")
plt.title("Linear Regression: Area → R_mean")
plt.legend()
plt.show()
