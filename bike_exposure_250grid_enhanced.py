# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Bicycle Traffic Exposure - 250m Grid
# DATE: 19.10.2025
# DESCRIPTION: Loads various years of bicycle traffic data (Geoportal Hamburg: Stadtradeln, DB Rad+),
# spatially aggregates them onto the 250m grid, calculates an
# average annual and daily exposure value per grid cell,
# and merges this information with the main accident dataset.
#
# ===================================================================

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import os
import warnings
warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# Global Configuration
print_header('Bicycle Traffic Exposure - Average per Grid Cell')
# Use absolute paths as defined by user
BASE_PATH = r"C:\Users\lucad\OneDrive - HafenCity Universität Hamburg\Studium_HCU\MA06\Thesis\Unfallstudie\Unfallstudy_Maschinelearn\imporved infrastructure"

GRID_FILE = f"{BASE_PATH}\\Gitter_HH_250x250.gpkg"
ACCIDENT_FILE = f"{BASE_PATH}\\UN_250Grid_HH_improved_with_OSM_raw.csv"

# List of bicycle traffic data files
RAD_FILES = [
    f"{BASE_PATH}\\de_hh_up_jahr2022_EPSG_25832.csv",
    f"{BASE_PATH}\\de_hh_up_jahr2023_EPSG_25832.csv",
    f"{BASE_PATH}\\de_hh_up_stadtradeln2018_EPSG_25832.csv",
    f"{BASE_PATH}\\de_hh_up_stadtradeln2019_EPSG_25832.csv",
    f"{BASE_PATH}\\de_hh_up_stadtradeln2020_EPSG_25832.csv"
]

OUTPUT_FILE = f"{BASE_PATH}\\UN_250Grid_HH_with_RadExposure.csv"
TARGET_CRS = "EPSG:25832"

# -------------------
# STEP 1: LOAD GRID GEOMETRY
# -------------------
print_header('STEP 1: Load Grid Geometry')
try:
    grid = gpd.read_file(GRID_FILE, engine='fiona')
    print(f"✓ {len(grid)} Grid cells loaded (using Fiona)")
except ImportError:
    grid = gpd.read_file(GRID_FILE, engine='pyogrio')
    print(f"✓ {len(grid)} Grid cells loaded (using Pyogrio)")
except Exception as e:
    print(f"ERROR: Could not load grid file: {e}")
    exit()

if grid.crs != TARGET_CRS:
    grid = grid.to_crs(TARGET_CRS)
    print(f"✓ Grid CRS transformed to {TARGET_CRS}")

# Ensure Grid_ID exists
if 'Grid_ID' not in grid.columns:
    print("  'Grid_ID' not found. Creating it...")
    grid['centroid'] = grid.geometry.centroid
    grid['Grid_X'] = (grid['centroid'].x // 250) * 250
    grid['Grid_Y'] = (grid['centroid'].y // 250) * 250
    grid['Grid_ID'] = (grid['Grid_X'].astype(int).astype(str) + '_' + grid['Grid_Y'].astype(int).astype(str))
    grid = grid.drop(columns=['centroid'])
    print("  ✓ 'Grid_ID' created.")
else:
    print("  ✓ 'Grid_ID' already exists.")

# -------------------
# STEP 2: PROCESS BICYCLE TRAFFIC DATA
# -------------------
print_header('STEP 2: Process Bicycle Traffic Data (by Year)')
grid_year_sums = {}

for rad_file in RAD_FILES:
    if not os.path.exists(rad_file):
        print(f"File not found (skipped): {os.path.basename(rad_file)}")
        continue
    
    # Extract year from filename
    year = None
    for y in [2018, 2019, 2020, 2022, 2023]:
        if str(y) in os.path.basename(rad_file):
            year = y
            break
    
    if year is None:
        print(f"Could not determine year for (skipped): {os.path.basename(rad_file)}")
        continue
    
    print(f"\n  Processing Year {year}...")
    
    try:
        rad_df = pd.read_csv(rad_file, sep=';', decimal=',')
        
        # Find the count column
        anzahl_col = None
        for col in ['anzahl', 'Anzahl', 'count']:
            if col in rad_df.columns:
                anzahl_col = col
                break
        
        # If no count column found, assume 1 per row
        if anzahl_col is None:
            rad_df['anzahl'] = 1
            anzahl_col = 'anzahl'
        
        rad_df[anzahl_col] = pd.to_numeric(rad_df[anzahl_col], errors='coerce')
        rad_df = rad_df.dropna(subset=[anzahl_col])
        
        # Load geometry (WKT)
        if 'geom' in rad_df.columns:
            rad_df['geometry'] = rad_df['geom'].apply(wkt.loads)
        
        rad_gdf = gpd.GeoDataFrame(rad_df, geometry='geometry', crs=TARGET_CRS)
        
        # Spatially join traffic data to the grid
        intersections = gpd.sjoin(
            rad_gdf[[anzahl_col, 'geometry']],
            grid[['Grid_ID', 'geometry']],
            how='inner',
            predicate='intersects'
        )
        
        # Sum counts per grid cell
        grid_sums = intersections.groupby('Grid_ID')[anzahl_col].sum().reset_index()
        grid_sums.columns = ['Grid_ID', 'anzahl_sum']
        grid_year_sums[year] = grid_sums
        
        print(f"    ✓ Data processed for {len(grid_sums)} grid cells.")
        
    except Exception as e:
        print(f"ERROR processing file: {e}")

# -------------------
# STEP 3: CALCULATE AVERAGE EXPOSURE
# -------------------
print_header('STEP 3: Calculate Average Exposure per Grid Cell')

if len(grid_year_sums) == 0:
    print("ERROR: No bicycle traffic data was successfully processed. Exiting.")
    exit(1)
else:
    print(f"✓ Data from {len(grid_year_sums)} years available for averaging.")

grid_summary = grid[['Grid_ID']].copy()

# Merge all yearly data into one summary dataframe
for year, year_data in grid_year_sums.items():
    grid_summary = grid_summary.merge(
        year_data.rename(columns={'anzahl_sum': f'anzahl_{year}'}),
        on='Grid_ID',
        how='left'
    )
    grid_summary[f'anzahl_{year}'] = grid_summary[f'anzahl_{year}'].fillna(0)

# Calculate totals and averages
year_cols = [col for col in grid_summary.columns if col.startswith('anzahl_')]
grid_summary['anzahl_total'] = grid_summary[year_cols].sum(axis=1)
grid_summary['Years_with_Data'] = (grid_summary[year_cols] > 0).sum(axis=1)

grid_summary['Bike_exposure_avg'] = 0.0
has_data = grid_summary['Years_with_Data'] > 0
grid_summary.loc[has_data, 'Bike_exposure_avg'] = (
    grid_summary.loc[has_data, 'anzahl_total'] / grid_summary.loc[has_data, 'Years_with_Data']
)

# Calculate per-day exposure (average annual / 365)
grid_summary['Bike_exposure_day'] = grid_summary['Bike_exposure_avg'] / 365

print(f"✓ {has_data.sum()} grid cells have exposure data.")
print(f"✓ Average annual exposure (mean): {grid_summary['Bike_exposure_avg'].mean():.2f}")
print(f"✓ Average daily exposure (mean): {grid_summary['Bike_exposure_day'].mean():.2f}")


# Clean duplicates before merge
print("\n  Checking grid_summary for duplicates...")
duplicates = grid_summary.duplicated(subset=['Grid_ID']).sum()

if duplicates > 0:
    print(f"{duplicates} duplicate Grid_IDs found! Dropping...")
    grid_summary = grid_summary.drop_duplicates(subset=['Grid_ID'], keep='first')
    print(f"✓ Duplicates removed.")
else:
    print(f"✓ No duplicates found ({len(grid_summary)} unique grid cells).")

# -------------------
# STEP 4: MERGE EXPOSURE DATA WITH ACCIDENTS
# -------------------
print_header('STEP 4: Merge Exposure Data with Accidents')

try:
    accidents = pd.read_csv(ACCIDENT_FILE, sep=';', decimal=',')
    original_count = len(accidents)
    print(f"✓ {original_count} accident records loaded.")
except FileNotFoundError:
    print(f"ERROR: Accident file not found: {ACCIDENT_FILE}")
    exit()

# Ensure Grid_ID exists in accident file
if 'Grid_ID' not in accidents.columns:
    print("  'Grid_ID' not found in accident file. Creating it from coordinates...")
    accidents['LINREFX'] = pd.to_numeric(accidents['LINREFX'], errors='coerce')
    accidents['LINREFY'] = pd.to_numeric(accidents['LINREFY'], errors='coerce')
    accidents['Grid_X'] = (accidents['LINREFX'] // 250) * 250
    accidents['Grid_Y'] = (accidents['LINREFY'] // 250) * 250
    accidents['Grid_ID'] = (accidents['Grid_X'].astype(int).astype(str) + '_' + accidents['Grid_Y'].astype(int).astype(str))
    print("  ✓ 'Grid_ID' created for accidents.")

# Define features to merge (LOG features removed as requested)
features_to_merge = ['Bike_exposure_avg', 'Bike_exposure_day', 'Years_with_Data']

accidents_final = accidents.merge(
    grid_summary[['Grid_ID'] + features_to_merge],
    on='Grid_ID',
    how='left'
)

# Fill NaNs with 0 for accidents in grid cells without exposure data
for col in features_to_merge:
    accidents_final[col] = accidents_final[col].fillna(0)

print(f"✓ Merge complete. Final record count: {len(accidents_final)} (Original: {original_count})")
if len(accidents_final) != original_count:
    print("  Warning: Row count changed during merge. Check for duplicates in accident file.")

# -------------------
# STEP 5: SAVE ENRICHED FILE
# -------------------
print_header('STEP 5: Save Enriched File')
try:
    accidents_final.to_csv(OUTPUT_FILE, sep=';', decimal=',', index=False)
    print(f"\n✓ SUCCESSFULLY SAVED TO: {OUTPUT_FILE}")
except PermissionError:
    print(f"\nERROR: Permission denied. Is the file '{OUTPUT_FILE}' open?")
except Exception as e:
    print(f"\nERROR: Could not save file: {e}")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*70)