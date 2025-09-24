#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib


# ---------------------------
# Feature selection
# ---------------------------
features = [
    'day_of_week', 'month', 'quarter', 'year',
    'lag_1_mentions', 'lag_3_mentions', 'lag_6_mentions',
    'moving_avg_3_mentions', 'moving_avg_6_mentions', 'moving_avg_9_mentions',
    'relative_performance', 'month_sin', 'month_cos'
]
train_data=pd.read_csv("data\\processed\\train_processed.csv")
test_data=pd.read_csv("data\\processed\\test_processed.csv")

# ---------------------------
# Merge train/test sets
# ---------------------------

train_data['dataset'] = 'train'
test_data['dataset'] = 'test'
df = pd.concat([train_data, test_data], ignore_index=True)

# ---------------------------
# Scaling numeric features
# ---------------------------
scaler = MinMaxScaler()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# ---------------------------
# Anomaly label creation (3-sigma rule on 'Apple')
# ---------------------------
threshold = 3
mean_value = df['Apple'].mean()
std_value = df['Apple'].std()

df['anomaly'] = ((df['Apple'] - mean_value).abs() > threshold * std_value).astype(int)

# ---------------------------
# Split back if needed
# ---------------------------
train_data = df[df['dataset'] == 'train'].drop(columns=['dataset'])
test_data = df[df['dataset'] == 'test'].drop(columns=['dataset'])

# ---------------------------
# Classification pipeline
# ---------------------------


X_train, X_test = train_data[features] , test_data[features]
y_train, y_test = train_data['anomaly'] , test_data['anomaly']

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, 'model.pkl')



# %%
