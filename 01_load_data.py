import pandas as pd

train = pd.read_csv(r"D:\Proj\data\train.csv")
test = pd.read_csv(r"D:\Proj\data\test.csv")

print("train:", train.shape)  
print("test:", test.shape)
print(train.head())           
print(train.columns.tolist())  