#%%
import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv("data\\raw\\train.csv")
test_data=pd.read_csv("data\\raw\\test.csv")

# Data Cleaning 
# %%
def correct_timestamp(df):
    df["timestamp"]=pd.to_datetime(df["timestamp"])
    return df
def handle_missing_vallue(df):
    for column in df.columns : 
        if df[column].isnull().any() : 
            median= df[column].median()
            df[column].fillna(median,inplace=True)
    return df 

train_data=correct_timestamp(train_data)
test_data=correct_timestamp(test_data)

train_data=handle_missing_vallue(train_data)
test_data=handle_missing_vallue(test_data)
#%%
# EDA : 
train_data.aggregate(['min', 'mean', 'max'])
test_data.aggregate(['min', 'mean', 'max'])
def plot_evolution_of_Volume(df,df1):
    # Assuming df has a column "timestamp" of dtype datetime
    plt.figure(figsize=(12, 6))

    # Loop over all columns except 'timestamp'
    for col in df.columns:
        for col2 in df1.columns: 
            if col==col2 and col != "timestamp":
                plt.scatter(df["timestamp"], df[col], label='Train')
                plt.scatter(df1["timestamp"], df1[col2], label='Test')

                plt.xlabel("Timestamp")
                plt.ylabel("Values")
                plt.title(f"Time Series of {col}")
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

plot_evolution_of_Volume(train_data,test_data)


 
def plot_Heat_Map(df):
    # Set the theme for seaborn
    sns.set_theme()
    # Load the example flights dataset and convert to long-form
    flights_long = sns.load_dataset("flights")
    # Pivot the dataset to create a matrix where rows represent months, columns represent years, and values represent passengers
    flights = flights_long.pivot(index="month", columns="year", values="passengers")
    # Create a heatmap with the numeric values in each cell
    fig, ax = plt.subplots(figsize=(9, 6))
    heatmap = sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
    # Set labels for x and y axis
    heatmap.set_xlabel('Company')
    heatmap.set_ylabel('Year')
    # Set title for the plot
    heatmap.set_title('Twitter Mentions Volume by Company and Year')
    # Show the plot
    plt.show()
#%% 
# Feature Engineering : 
def New_Features_Creation(df): 
    # Date-Time Features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    # Lag Features
    lag_periods = [1, 3, 6]  # Lag periods to consider
    for period in lag_periods:
        df[f'lag_{period}_mentions'] = df['Apple'].shift(periods=period)
    
    # Moving Averages
    window_sizes = [3, 6, 9]  # Window sizes for moving averages
    for window_size in window_sizes:
        df[f'moving_avg_{window_size}_mentions'] = df['Apple'].rolling(window=window_size).mean()

    # Relative Performance
    benchmark = df[['Amazon', 'Google']].mean(axis=1)  # Using the mean of Amazon and Google as benchmark
    df['relative_performance'] = df['Apple'] / benchmark - 1

    # Seasonality Indicators
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df 

# Add a column to remember origin
train_data['dataset'] = 'train'
test_data['dataset'] = 'test'

# Concatenate
df = pd.concat([train_data, test_data], ignore_index=True)

# Sort by date (assuming your date column is named 'date')
df = df.sort_values(by='timestamp').reset_index(drop=True)
df=New_Features_Creation(df)

# Split back
train_data = df[df['dataset'] == 'train'].drop(columns=['dataset'])
test_data = df[df['dataset'] == 'test'].drop(columns=['dataset'])


# %%
data_path=os.path.join("data","processed")
os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)