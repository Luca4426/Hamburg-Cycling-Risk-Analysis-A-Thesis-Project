import pandas as pd

# Beispiel-Daten
df = pd.read_csv("Acc_segbased_final_classified.csv", sep=";", decimal=",")

# Spalte, die normalisiert werden soll
column_name = "Car_DTV"

# Min-Max-Normalisierung (0-1)
min_val = df[column_name].min()
max_val = df[column_name].max()

df[column_name + "_normalized"] = (df[column_name] - min_val) / (max_val - min_val)

# Prüfen
print(df[[column_name, column_name + "_normalized"]].head())
print(f"\nMin: {df[column_name + '_normalized'].min()}, Max: {df[column_name + '_normalized'].max()}")

# Speichern (optional)
df.to_csv("Acc_250Grid_HH_enhanced_final.csv", sep=";", decimal=",", index=False)
