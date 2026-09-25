import pandas as pd
import sys

df = pd.read_csv(r"C:\Users\ak\OneDrive\Desktop\Stuff\Python projects\Data\F_F_Research_Data_5_Factors_2x3.csv", skiprows=3)
df = df.rename(columns={df.columns[0]: "Date"})

# The file appends an annual-data section after the monthly rows, marked by
# a row like " Annual Factors: January-December ". Cut everything from there on.
annual_marker = df["Date"].astype(str).str.contains("Annual Factors", na=False)
if annual_marker.any():
    df = df.loc[: annual_marker.idxmax() - 1]

df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="%Y%m")
print(df.head(10))
print(df.tail(10))
#print(sys.executable)