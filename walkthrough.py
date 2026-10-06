# %%
import pandas as pd

df = pd.read_csv("/Users/ay/Downloads/railway.csv", parse_dates=["Date of Purchase", "Date of Journey"])
print(df.shape)
df.head()
# %%
 ## the target variable

df["disrupted"] = df["Journey Status"].isin(["Delayed", "Cancelled"]).astype(int)

print(df["disrupted"].value_counts())
print(df["disrupted"].mean())

# %%
# identifying the features
df["booking_lead_days"] = (df["Date of Journey"] - df["Date of Purchase"]).dt.days

df[["Date of Purchase", "Date of Journey", "booking_lead_days"]].head()
# %%
# extracting day and month from dataset
df["journey_day_of_week"] = df["Date of Journey"].dt.day_name()
df["journey_month"] = df["Date of Journey"].dt.month

df[["Date of Journey", "journey_day_of_week", "journey_month"]].head()
# %%
#  Scheduled journey duration 
dep = pd.to_timedelta(df["Departure Time"])
arr = pd.to_timedelta(df["Arrival Time"])
duration = (arr - dep).dt.total_seconds() / 60
duration = duration.where(duration >= 0, duration + 24 * 60)  # fix midnight crossovers
df["scheduled_duration_mins"] = duration
# %%
 # Fix the Railcard blank-value issue
print("Before fix:", df["Railcard"].isna().sum(), "missing")
df["Railcard"] = df["Railcard"].fillna("None")
print("After fix:", df["Railcard"].isna().sum(), "missing")
# %%
feature_cols = [
    "Departure Station", "Arrival Destination",
    "Ticket Class", "Ticket Type", "Railcard",
    "Purchase Type", "Payment Method",
    "Price", "booking_lead_days",
    "journey_day_of_week", "journey_month",
    "scheduled_duration_mins",
]

model_df = df[feature_cols + ["disrupted"]].dropna()
print(f"Rows after cleaning: {len(model_df)}")
model_df.to_csv("data/model_ready.csv", index=False)
model_df.head()
# %%

from sklearn.model_selection import train_test_split

CATEGORICAL = [
    "Departure Station", "Arrival Destination", "Ticket Class",
    "Ticket Type", "Railcard", "Purchase Type", "Payment Method",
    "journey_day_of_week",
]
NUMERIC = ["Price", "booking_lead_days", "journey_month", "scheduled_duration_mins"]

X = model_df[CATEGORICAL + NUMERIC]
y = model_df["disrupted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
# %% 
# Build the preprocessing + Logistic Regression pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

log_reg = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

log_reg.fit(X_train, y_train)
print("Logistic Regression trained.")

# %%
# Evaluate Logistic Regression
from sklearn.metrics import classification_report, roc_auc_score

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["On Time", "Disrupted"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
#%%
# Build Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    ))
])

rf.fit(X_train, y_train)
print("Random Forest trained.")

#%%  Evaluate Random Forest
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf, target_names=["On Time", "Disrupted"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")
# %%
# Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)
print("                Predicted On Time   Predicted Disrupted")
print(f"Actual On Time       {cm[0][0]:>10}          {cm[0][1]:>10}")
print(f"Actual Disrupted     {cm[1][0]:>10}          {cm[1][1]:>10}")
# %%
# Feature importance
feature_names = rf.named_steps["prep"].get_feature_names_out()
importances = rf.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names, "importance": importances
}).sort_values("importance", ascending=False).head(10)

importance_df
# %%