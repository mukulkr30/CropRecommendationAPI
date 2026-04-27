import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

df = pd.read_csv("Crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision_Tree": DecisionTreeClassifier(),
    "Random_Forest": RandomForestClassifier(),
    "Naive_Bayes": GaussianNB(),
    "SVM": SVC(kernel='rbf'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

# results = []
#
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred, average='weighted')
#     rec = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#
#     results.append([name, acc * 100, prec, rec, f1])
#
# results_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)", "Precision", "Recall", "F1-Score"])
# print(results_df)

tmodel  = models["Random_Forest"]
tmodel.fit(X_train, y_train)

result = tmodel.predict([[75,75,75,35.0,82.1,7.2,200.1]])
predicted_crop = le.inverse_transform(result)

tmodel.fit(X_train, y_train)
# y_pred = tmodel.predict(X_test)

# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred, average='weighted')
# rec = recall_score(y_test, y_pred, average='weighted')
# f1 = f1_score(y_test, y_pred, average='weighted')
# print("randomForest", acc * 100, prec, rec, f1)
#
# print("Predicted Crop:", predicted_crop[0])

joblib.dump(tmodel, "Crop_recommendation_model.pkl")
pickle.dump(le, open("label_encoder.pkl", "wb"))