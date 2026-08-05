import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "online_shoppers_intention.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "imakash3011/online-shoppers-purchasing-intention-dataset",
    file_path
)

print(df.head())
print(df.shape)
print(df.columns)