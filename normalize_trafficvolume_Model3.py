# ===================================================================
# MASTER THESIS
# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)
# SCRIPT: Normalize Traffic Volume (DTV) - Model 3 (Point-based)
# DATE: 23.11.2025
# DESCRIPTION: Applies Min-Max normalization (0-1 scaling) to 
# 	       the Car_DTV column for  machine learning model preparation.
# ===================================================================

import pandas as pd

# ===================================================================
# HELPER FUNCTION
# ===================================================================
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# ===================================================================
# GLOBAL CONFIGURATION
# ===================================================================
print_header('TRAFFIC VOLUME NORMALIZATION - Model 3')

# Input/Output paths
INPUT_CSV = "Acc_segbased_final_classified.csv"
OUTPUT_CSV = "Acc_segbased_final_normalized.csv"

# Column to normalize
COLUMN_NAME = "Car_DTV"

# ===================================================================
# STEP 1: LOAD DATA
# ===================================================================
print_header('STEP 1: Load Data')

df = pd.read_csv(INPUT_CSV, sep=";", decimal=",")
print(f"✓ Loaded {len(df):,} records from {INPUT_CSV}")

# ===================================================================
# STEP 2: APPLY MIN-MAX NORMALIZATION
# ===================================================================
print_header('STEP 2: Apply Min-Max Normalization (0-1)')

min_val = df[COLUMN_NAME].min()
max_val = df[COLUMN_NAME].max()

df[COLUMN_NAME + "_normalized"] = (df[COLUMN_NAME] - min_val) / (max_val - min_val)

print(f"\nOriginal values:")
print(f"  Min: {min_val:,.2f}")
print(f"  Max: {max_val:,.2f}")

print(f"\nNormalized values:")
print(f"  Min: {df[COLUMN_NAME + '_normalized'].min():.6f}")
print(f"  Max: {df[COLUMN_NAME + '_normalized'].max():.6f}")

# ===================================================================
# STEP 3: PREVIEW RESULTS
# ===================================================================
print_header('STEP 3: Preview Results')

print("\nFirst 5 rows (original vs normalized):")
print(df[[COLUMN_NAME, COLUMN_NAME + "_normalized"]].head())

# ===================================================================
# STEP 4: EXPORT RESULTS
# ===================================================================
print_header('STEP 4: Export Results')

df.to_csv(OUTPUT_CSV, sep=";", decimal=",", index=False)
print(f"\n✓ File saved: {OUTPUT_CSV}")
print(f"✓ Total rows: {len(df):,}")
print(f"✓ New column added: {COLUMN_NAME}_normalized")

print_header('✓ SCRIPT COMPLETED SUCCESSFULLY')