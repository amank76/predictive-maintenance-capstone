import pandas as pd
from pathlib import Path
path=Path(__file__).resolve().parents[1]/"data"/"engine_data.csv"
df=pd.read_csv(path)
expected=["Engine rpm","Lub oil pressure","Fuel pressure","Coolant pressure","lub oil temp","Coolant temp","Engine Condition"]
assert list(df.columns)==expected, f"Unexpected columns: {list(df.columns)}"
assert set(df["Engine Condition"].dropna().unique()).issubset({0,1})
print("Rows:",len(df),"Columns:",len(df.columns))
print("Missing values:\n",df.isna().sum().to_string())
print("Class counts:\n",df["Engine Condition"].value_counts().to_string())
print(df.describe().round(2).to_string())
