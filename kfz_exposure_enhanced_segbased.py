# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Add Car (DTV) Exposure Features
# DATE: 20.10.2025
# DESCRIPTION: Loads the accident file and the DTV (Daily Traffic Volume) data.
# Matches each accident to the nearest DTV measurement point within a 50m radius.
# For accidents without a nearby measurement, it imputes a DTV value
# based on the road type ('strtyp').
# Adds the final columns: Car_DTV_AVG, Car_DTV_SOURCE.
#
# ===================================================================

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import warnings
import os

warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# -------------------
# STEP 1: CONFIGURATION
# -------------------
print_header("STEP 1: Configuration")
BASE_DIR = r"C:\Users\lucad\analyse_python\final"
os.chdir(BASE_DIR)

ACCIDENT_CSV = "unfaelle_enriched_V3_with_radexp.csv"
DTV_CSV = "de_hh_up_verkehrsmengen_dtv_hvs_2019_EPSG_25832.csv"
OUTPUT_CSV = "unfaelle_enriched_V4_complete.csv"

MATCH_DISTANCE_M = 50
CRS_METRIC = "EPSG:25832"

print(f"Base directory: {BASE_DIR}")
print(f"Input file: {ACCIDENT_CSV}")
print(f"Output file: {OUTPUT_CSV}")

# -------------------
# STEP 2: LOAD DTV (CAR TRAFFIC) DATA
# -------------------
print_header("STEP 2: Load DTV (Car Traffic) Data")
dtv_df = pd.read_csv(DTV_CSV, sep=";", encoding="utf-8-sig", decimal=",")
dtv_df["geometry"] = dtv_df["geom"].apply(wkt.loads)
dtv_gdf = gpd.GeoDataFrame(dtv_df, geometry="geometry", crs=CRS_METRIC)
dtv_gdf["dtv"] = pd.to_numeric(dtv_gdf["dtv"], errors="coerce")
dtv_gdf = dtv_gdf[["dtv", "geometry"]].dropna(subset=["dtv"]).reset_index(drop=True)
dtv_gdf["dtv_id"] = dtv_gdf.index
print(f"  {len(dtv_gdf):,} DTV measurement points loaded.")

# -------------------
# STEP 3: LOAD ACCIDENT DATA
# -------------------
print_header("STEP 3: Load Accident Data")
udf = pd.read_csv(ACCIDENT_CSV, sep=";", encoding="utf-8-sig", decimal=",")
print(f"  {len(udf):,} accidents loaded.")

# Clean coordinates
for col in ["LINREFX","LINREFY"]:
    udf[col] = pd.to_numeric(udf[col], errors="coerce")

udf = udf.dropna(subset=["LINREFX","LINREFY"])
udf = udf[(udf["LINREFX"]!=0) & (udf["LINREFY"]!=0)]

# Create GeoDataFrame
ugdf = gpd.GeoDataFrame(udf, geometry=gpd.points_from_xy(udf["LINREFX"], udf["LINREFY"]), crs=CRS_METRIC)
ugdf = ugdf[ugdf.geometry.is_valid & ~ugdf.geometry.is_empty].copy()
print(f"  {len(ugdf):,} valid accident geometries prepared.")

# -------------------
# STEP 4: MATCH ACCIDENTS TO DTV POINTS
# -------------------
print_header("STEP 4: Match Accidents to nearest DTV point")
# Use original index to re-merge later
ugdf_indexed = ugdf[["geometry"]].copy().reset_index(drop=False)

# Spatial join to nearest DTV point within 50m
unfall_to_dtv = gpd.sjoin_nearest(
    ugdf_indexed,
    dtv_gdf[["dtv_id", "dtv", "geometry"]],
    how="left",
    max_distance=MATCH_DISTANCE_M,
    distance_col="dist_dtv_m"
)
print(f"  Spatial join complete (Max distance: {MATCH_DISTANCE_M}m).")

# -------------------
# STEP 5: AGGREGATE AND IMPUTE DATA
# -------------------
print_header("STEP 5: Aggregate and Impute Data")

# In case one accident is near multiple points, take the average
unfall_dtv_agg = unfall_to_dtv.groupby("index")["dtv"].mean().reset_index()
unfall_dtv_agg.columns = ["index", "KFZ_DTV_MEASURED"]

# Merge measured data back to the original dataframe
udf_final = udf.reset_index(drop=False).merge(unfall_dtv_agg, on="index", how="left")

# --- Imputation Step ---
print("  Imputing missing DTV values based on 'strtyp'...")
if "strtyp" in udf_final.columns:
    # NOTE: These keys refer to data values, not UI text, so they remain in German.
    DTV_MAP = {
        "Hauptstraße": 8000.0, "Hauptstrasse": 8000.0,
        "Nebenstraße": 800.0, "Nebenstrasse": 800.0,
        "Unbekannt": 1500.0
    }
    udf_final["_default"] = udf_final["strtyp"].map(DTV_MAP).fillna(1500.0)
else:
    print("  Warning: 'strtyp' column not found. Using default imputation value 1500.0 for all.")
    udf_final["_default"] = 1500.0

# --- Create the final columns (as requested) ---
udf_final["Car_DTV_AVG"] = udf_final["KFZ_DTV_MEASURED"].fillna(udf_final["_default"]).round(0)
udf_final["Car_DTV_SOURCE"] = udf_final["KFZ_DTV_MEASURED"].apply(
    lambda x: "measured" if pd.notna(x) else "imputed"
)

# Clean up temporary columns
udf_final = udf_final.drop(columns=["index", "_default", "KFZ_DTV_MEASURED"])

n_measured = (udf_final["Car_DTV_SOURCE"] == "measured").sum()
print(f"  Measured: {n_measured:,}, Imputed: {len(udf_final)-n_measured:,}")

# -------------------
# STEP 6: SAVE FINAL FILE
# -------------------
print_header("STEP 6: Save Final File")
udf_final.to_csv(OUTPUT_CSV, index=False, sep=";", encoding="utf-8-sig", decimal=",")

print(f"\nSUCCESS: {OUTPUT_CSV}")
print(f"{len(udf_final):,} rows x {len(udf_final.columns)} columns")
print("\nNew columns added:")
print("  - Car_DTV_AVG: Car traffic (vehicles/day)")
print("  - Car_DTV_SOURCE: 'measured' or 'imputed'")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*7m0)