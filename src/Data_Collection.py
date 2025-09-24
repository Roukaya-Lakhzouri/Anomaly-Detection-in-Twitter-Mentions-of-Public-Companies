import kagglehub
import os
import pandas as pd
#from sklearn.model_selection import train_test_split

#Dowload Dataset from Kaggle 
dataset_folder = kagglehub.dataset_download("julienjta/twitter-mentions-volumes")

# List all CSV files
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
print("CSV files found:", csv_files)

# Load the first CSV file
data = pd.read_csv(os.path.join(dataset_folder, csv_files[0]))
data_path=os.path.join("data","raw")
os.makedirs(data_path, exist_ok=True)

# Split data into training and testing sets
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)