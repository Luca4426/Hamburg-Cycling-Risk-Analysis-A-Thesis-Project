# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Car (DTV) Traffic Exposure - 250m Grid
# DATE: 19.10.2025
# DESCRIPTION: Loads official DTV (Daily Traffic Volume) data for cars (Source: Geoportal Hamburg).
# Spatially intersects traffic segments with the 250m grid,
# Using Traffic Volumes for primary streets per cell.
# Estimates traffic for unmeasured residential roads using OSM data.
# Merges the final exposure data with the main accident file.
#
# ===================================================================

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import re
import warnings
warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# Global Configuration
print_header('Car (KFZ) Exposure - Configuration')
BASE_PATH = r"C:\Users\lucad\OneDrive - HafenCity Universität Hamburg\Studium_HCU\MA06\Thesis\Unfallstudie\Unfallstudy_Maschinelearn\imporved infrastructure"

GRID_FILE = f"{BASE_PATH}\\Gitter_HH_250x250.gpkg"
ACCIDENT_FILE = f"{BASE_PATH}\\UN_250Grid_HH_with_RadExposure.csv"
KFZ_FILE = f"{BASE_PATH}\\de_hh_up_verkehrsmengen_dtv_hvs_2019_EPSG_25832.csv"
OUTPUT_FILE = f"{BASE_PATH}\\UN_250Grid_HH_with_ALL_Exposure.csv"

# Assumed DTV for unmeasured residential roads
RESIDENTIAL_DTV = 1500
TARGET_CRS = "EPSG:25832"

# -------------------
# STEP 1: LOAD GRID GEOMETRY
# -------------------
print_header('STEP 1: Load Grid Geometry')
try:
    grid = gpd.read_file(GRID_FILE, engine='fiona')
except:
    grid = gpd.read_file(GRID_FILE)
print(f"✓ {len(grid)} Grid cells loaded.")

if grid.crs is None or str(grid.crs) != TARGET_CRS:
    grid = grid.to_crs(TARGET_CRS)
    print(f"✓ Grid CRS transformed to {TARGET_CRS}")

# Ensure Grid_ID exists
if 'Grid_ID' not in grid.columns:
    print("  'Grid_ID' not found. Creating it...")
    grid['centroid'] = grid.geometry.centroid
    grid['Grid_X'] = (grid['centroid'].x // 250) * 250
    grid['Grid_Y'] = (grid['centroid'].y // 250) * 250
    grid['Grid_ID'] = (grid['Grid_X'].astype(int).astype(str) + '_' + 
                       grid['Grid_Y'].astype(int).astype(str))
    grid = grid.drop(columns=['centroid'])
    print("'Grid_ID' created.")
else:
    print("'Grid_ID' already exists.")

# -------------------
# STEP 2: LOAD CAR TRAFFIC (DTV) DATA
# -------------------
print_header('STEP 2: Load Car Traffic (DTV) Data')
try:
    kfz_df = pd.read_csv(KFZ_FILE, sep=';', decimal=',')
    print(f"✓ {len(kfz_df)} raw traffic records loaded.")
except FileNotFoundError:
    print(f"ERROR: Car traffic file not found: {KFZ_FILE}")
    exit()

kfz_df['dtv'] = pd.to_numeric(kfz_df['dtv'], errors='coerce')
kfz_df = kfz_df.dropna(subset=['dtv'])
print(f"{len(kfz_df)} valid traffic records remaining after cleaning.")

# -------------------
# STEP 3: PREPARE GEOMETRIES (WKT)
# -------------------
print_header('STEP 3: Prepare Geometries (WKT)')

def parse_wkt_clean(geom_str):
    """Helper function to parse WKT strings, removing EPSG tags."""
    if pd.isna(geom_str):
        return None
    geom_str = str(geom_str).strip()
    # Remove trailing EPSG tags
    geom_str = re.sub(r'\s+EPSG:\d+$', '', geom_str)
    try:
        return wkt.loads(geom_str)
    except:
        return None

kfz_df['geometry'] = kfz_df['geom'].apply(parse_wkt_clean)
kfz_df = kfz_df.dropna(subset=['geometry'])
kfz_gdf = gpd.GeoDataFrame(kfz_df, geometry='geometry', crs=TARGET_CRS)

print(f"{len(kfz_gdf)} valid geometries prepared.")

if kfz_gdf.crs != grid.crs:
    kfz_gdf = kfz_gdf.to_crs(grid.crs)

# -------------------
# STEP 4: SPATIAL JOIN AND AGGREGATION
# -------------------
print_header('STEP 4: Spatial Join and Aggregation')

joined = gpd.sjoin(kfz_gdf, grid[['Grid_ID', 'geometry']], 
                   how='inner', predicate='intersects')

print(f"✓ {len(joined)} intersections found between traffic data and grid.")

if len(joined) > 0:
    print("  Calculating length-weighted traffic volume in grid cells...")
    results = []
    
    # Iterate over intersections to clip road segments and calculate weighted DTV
    for idx, row in joined.iterrows():
        try:
            street_geom = row.geometry
            grid_poly = grid[grid['Grid_ID'] == row['Grid_ID']].geometry.iloc[0]
            
            # Clip the road segment to the grid cell
            clipped = street_geom.intersection(grid_poly)
            
            length = clipped.length if hasattr(clipped, 'length') else 0
            
            if length > 0:
                results.append({
                    'Grid_ID': row['Grid_ID'],
                    'dtv': row['dtv'],
                    'length_in_grid': length,
                    'kfz_total': length * row['dtv'] # Weighted DTV
                })
        except:
            continue # Skip if clipping fails
    
    results_df = pd.DataFrame(results)
    print(f"  ✓ {len(results_df)} valid clipped road segments calculated.")
    
    # Aggregate results by Grid_ID
    grid_kfz = results_df.groupby('Grid_ID').agg({
        'kfz_total': 'sum',
        'dtv': ['max', 'mean', 'count'],
        'length_in_grid': 'sum'
    }).reset_index()
    
    grid_kfz.columns = ['Grid_ID', 'Car_DTV_Grid', 'Car_DTV_Max', 'Car_DTV_Avg', 
                        'Road_Count_Measured', 'Road_Length_Measured']
    
    print(f"✓ {len(grid_kfz)} grid cells have measured car traffic data.")
    
    # Merge with the full grid to keep all cells
    grid_summary = grid[['Grid_ID']].copy()
    grid_summary = grid_summary.merge(grid_kfz, on='Grid_ID', how='left')
    
    grid_summary['Car_Data_Source'] = 'measured'
    grid_summary.loc[grid_summary['Car_DTV_Grid'].isna(), 'Car_Data_Source'] = 'none'
    
    # Fill NaNs for cells with no measured data
    for col in ['Car_DTV_Grid', 'Car_DTV_Max', 'Car_DTV_Avg', 
                'Road_Count_Measured', 'Road_Length_Measured']:
        grid_summary[col] = grid_summary[col].fillna(0)
else:
    print("No intersections found. Creating empty summary.")
    grid_summary = grid[['Grid_ID']].copy()
    for col in ['Car_DTV_Grid', 'Car_DTV_Max', 'Car_DTV_Avg', 
                'Road_Count_Measured', 'Road_Length_Measured']:
        grid_summary[col] = 0
    grid_summary['Car_Data_Source'] = 'none'

# Check and remove duplicates
duplicates = grid_summary.duplicated(subset=['Grid_ID']).sum()
if duplicates > 0:
    grid_summary = grid_summary.drop_duplicates(subset=['Grid_ID'], keep='first')
    print(f"{duplicates} duplicate Grid_IDs removed from summary.")

# -------------------
# STEP 5: MERGE WITH ACCIDENT DATA
# -------------------
print_header('STEP 5: Merge with Accident Data')

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
    accidents['Grid_ID'] = (accidents['Grid_X'].astype(int).astype(str) + '_' + 
                            accidents['Grid_Y'].astype(int).astype(str))
    print("  ✓ 'Grid_ID' created for accidents.")

# Define features to merge (LOG features removed)
kfz_features = ['Car_DTV_Grid', 'Car_DTV_Max', 'Car_DTV_Avg', 
                'Road_Length_Measured', 'Road_Count_Measured', 'Car_Data_Source']

accidents_final = accidents.merge(
    grid_summary[['Grid_ID'] + kfz_features],
    on='Grid_ID',
    how='left'
)

# Fill NaNs post-merge
for col in kfz_features:
    if col == 'Car_Data_Source':
        accidents_final[col] = accidents_final[col].fillna('none')
    else:
        accidents_final[col] = accidents_final[col].fillna(0)

# -------------------
# STEP 6: ESTIMATE RESIDENTIAL TRAFFIC
# -------------------
print_header('STEP 6: Estimate Residential Traffic (for unmeasured roads)')

if 'OSM_Road_Length_Total' in accidents_final.columns:
    # Find rows with no measured DTV data but with existing roads (from OSM)
    no_kfz = (accidents_final['Car_Data_Source'] == 'none')
    has_osm = (accidents_final['OSM_Road_Length_Total'] > 0)
    estimate_mask = no_kfz & has_osm
    
    print(f"  Found {estimate_mask.sum()} records with OSM roads but no measured DTV data.")
    
    # Apply the residential DTV estimate
    accidents_final.loc[estimate_mask, 'Car_DTV_Grid'] = (
        accidents_final.loc[estimate_mask, 'OSM_Road_Length_Total'] * RESIDENTIAL_DTV
    )
    accidents_final.loc[estimate_mask, 'Car_DTV_Max'] = RESIDENTIAL_DTV
    accidents_final.loc[estimate_mask, 'Car_DTV_Avg'] = RESIDENTIAL_DTV
    accidents_final.loc[estimate_mask, 'Car_Data_Source'] = 'estimated_residential'
    
    print(f"✓ {estimate_mask.sum()} records updated with estimated traffic.")
else:
    print("  'OSM_Road_Length_Total' column not found. Skipping estimation step.")

print(f"✓ {len(accidents_final)} final accident records processed.")

# -------------------
# STEP 7: FINAL STATISTICS
# -------------------
print_header('STEP 7: Final Statistics')

measured = (accidents_final['Car_Data_Source'] == 'measured').sum()
estimated = (accidents_final['Car_Data_Source'] == 'estimated_residential').sum()
none = (accidents_final['Car_Data_Source'] == 'none').sum()
total = len(accidents_final)

print(f"\nCar Data Source Distribution:")
print(f"  Measured:  {measured:6d} ({measured/total*100:5.1f}%)")
print(f"  Estimated: {estimated:6d} ({estimated/total*100:5.1f}%)")
print(f"  None:      {none:6d} ({none/total*100:5.1f}%)")

if measured > 0:
    measured_data = accidents_final[accidents_final['Car_Data_Source'] == 'measured']
    print(f"\nStatistics for Measured Data (per record):")
    print(f"  Total Car DTV (Grid): {measured_data['Car_DTV_Grid'].mean():,.0f} (Average)")
    print(f"  Max DTV:              {measured_data['Car_DTV_Max'].mean():,.0f} (Average)")

# -------------------
# STEP 8: SAVE FINAL FILE
# -------------------
print_header('STEP 8: Save Final File')

try:
    accidents_final.to_csv(OUTPUT_FILE, sep=';', decimal=',', index=False)
    print(f"\n✓ SUCCESSFULLY SAVED TO: {OUTPUT_FILE}")
    print(f"  Rows:   {len(accidents_final)}")
    print(f"  Columns: {len(accidents_final.columns)}")
except PermissionError:
    print(f"\nERROR: Permission denied. Is the file '{OUTPUT_FILE}' open?")
except Exception as e:
    print(f"\nERROR: Could not save file: {e}")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*7Am0)