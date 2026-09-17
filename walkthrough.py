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