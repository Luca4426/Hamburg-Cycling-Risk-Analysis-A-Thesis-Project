# ===================================================================
# MASTER THESIS
# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)
# SCRIPT: Car Exposure Estimation - Grid-based Classification with Measurement Correction
# DATE: 23.11.2025
# DESCRIPTION: Loads accident data with existing DTV values and 250m grid geometry.
#              Re-adjusts estimated DTV values using grid-based OSM classification.
#              Classifies grid cells by highest road type and generates improved
#              DTV estimates. Keeps measured values unchanged, replaces previous
#              estimates with refined grid-based values (second adjustment).
# ===================================================================

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import warnings
warnings.filterwarnings('ignore')

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
print_header('CAR EXPOSURE - GRID CLASSIFICATION WITH MEASUREMENT CORRECTION')

# Input paths
ACCIDENTS_CSV = "Acc_250Grid_HH_enhanced.csv"
GRID_GPKG = r"C:\Users\lucad\OneDrive - HafenCity Universität Hamburg\Studium_HCU\MA06\Thesis\Verschriftlichung\BikeAcc_ML\3 Enhanced\v3\Grid_HH_250x250.gpkg"

# CRS
TARGET_CRS = 'EPSG:25832'

# 4-category road classification
aadt_mapping = {
    'Highway': 30000, 
    'Main Road': 10000, 
    'Connector Road': 3000,
    'Residential Street': 800, 
    'Not Found': 500, 
    'Off Street': 0
}

class_hierarchy = ['Highway', 'Main Road', 'Connector Road', 'Residential Street', 'Not Found', 'Off Street']

highway_classification = {
    'motorway': 'Highway', 'motorway_link': 'Highway',
    'trunk': 'Highway', 'trunk_link': 'Highway',
    'primary': 'Main Road', 'primary_link': 'Main Road',
    'secondary': 'Main Road', 'secondary_link': 'Main Road',
    'tertiary': 'Connector Road', 'tertiary_link': 'Connector Road',
    'unclassified': 'Connector Road',
    'residential': 'Residential Street',
    'living_street': 'Residential Street',
    'service': 'Residential Street'
}

# ===================================================================
# STEP 1: LOAD ACCIDENT DATA
# ===================================================================
print_header('STEP 1: Load Accident Data')

df = pd.read_csv(ACCIDENTS_CSV, sep=';', decimal=',')
df['geometry'] = df.apply(lambda row: Point(row['LINREFX'], row['LINREFY']), axis=1)
gdf_accidents = gpd.GeoDataFrame(df, geometry='geometry', crs=TARGET_CRS)
print(f"✓ Loaded {len(gdf_accidents):,} accident records")

# ===================================================================
# STEP 2: LOAD GRID GEOMETRY
# ===================================================================
print_header('STEP 2: Load Grid Geometry')

gdf_grid = gpd.read_file(GRID_GPKG)
if gdf_grid.crs != TARGET_CRS:
    gdf_grid = gdf_grid.to_crs(TARGET_CRS)
gdf_grid['Grid_ID_Internal'] = range(len(gdf_grid))
print(f"✓ Loaded {len(gdf_grid):,} grid cells")

# ===================================================================
# STEP 3: SPATIAL JOIN - ACCIDENTS TO GRID
# ===================================================================
print_header('STEP 3: Assign Accidents to Grid Cells')

acc_with_grid = gpd.sjoin(gdf_accidents, gdf_grid[['Grid_ID_Internal', 'geometry']], how='left', predicate='within')
matched = acc_with_grid['Grid_ID_Internal'].notna().sum()
print(f"✓ {matched:,} accidents matched to grid cells")

# ===================================================================
# STEP 4: DOWNLOAD OSM ROAD NETWORK
# ===================================================================
print_header('STEP 4: Download OSM Road Network')

print("⏳ Downloading road network (this may take 2-3 minutes)...")
try:
    G = ox.graph_from_place('Hamburg, Germany', network_type='drive')
    gdf_roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
    gdf_roads = gdf_roads.to_crs(TARGET_CRS)
    print(f"✓ Downloaded {len(gdf_roads):,} road segments")
except Exception as e:
    print(f"⚠ Error downloading OSM data: {e}")
    raise

# ===================================================================
# STEP 5: CLASSIFY ROADS AND DETERMINE HIGHEST CLASS PER GRID
# ===================================================================
print_header('STEP 5: Road Classification per Grid Cell')

# Helper function to classify highway tags
def classify_highway(highway_tag):
    if highway_tag is None or (isinstance(highway_tag, float) and pd.isna(highway_tag)):
        return 'Not Found'
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0] if highway_tag else 'Not Found'
    return highway_classification.get(str(highway_tag).lower().strip(), 'Not Found')

print("⏳ Classifying roads...")
gdf_roads['Street_Type_Classified'] = gdf_roads['highway'].apply(classify_highway)
print(f"✓ Classified {len(gdf_roads):,} road segments")

# Spatial join: Roads → Grids
print("⏳ Finding roads within grid cells (this may take several minutes)...")
roads_in_grids = gpd.sjoin(gdf_roads[['Street_Type_Classified', 'geometry']],
                            gdf_grid[['Grid_ID_Internal', 'geometry']],
                            how='inner', predicate='intersects')

# Helper function to get highest class
def get_highest_class(classes):
    for cls in class_hierarchy:
        if cls in classes.tolist():
            return cls
    return 'Off Street'

grid_class = roads_in_grids.groupby('Grid_ID_Internal', as_index=False).agg({
    'Street_Type_Classified': lambda x: get_highest_class(x)
})
grid_class.rename(columns={'Street_Type_Classified': 'Street_Type_Grid'}, inplace=True)
grid_class['AADT_Grid_Estimated'] = grid_class['Street_Type_Grid'].map(aadt_mapping)
print(f"✓ Classified {len(grid_class):,} grid cells")

# ===================================================================
# STEP 6: MERGE AND CREATE FINAL DTV COLUMNS
# ===================================================================
print_header('STEP 6: Merge and Create Final DTV Values')

# Merge back to accidents
output_df = pd.DataFrame(acc_with_grid.drop(columns='geometry'))
output_df = output_df.merge(grid_class, on='Grid_ID_Internal', how='left')
output_df['Street_Type_Grid'].fillna('Off Street', inplace=True)
output_df['AADT_Grid_Estimated'].fillna(0, inplace=True)

print("✓ Street_Type_Grid merged")

# Calculate estimated AADT from grid classification
aadt_grid_estimated = output_df['Street_Type_Grid'].map(aadt_mapping)

# Check for existing measured values
has_car_dtv_grid = 'Car_DTV_Grid' in output_df.columns
has_car_data_source = 'Car_Data_Source' in output_df.columns
has_car_dtv_source = 'Car_DTV_Source' in output_df.columns
source_col = 'Car_Data_Source' if has_car_data_source else ('Car_DTV_Source' if has_car_dtv_source else None)

print(f"\nChecking for measured values:")
print(f"  Car_DTV_Grid column: {'FOUND' if has_car_dtv_grid else 'NOT FOUND'}")
print(f"  Source column: {source_col if source_col else 'NOT FOUND'}")

# Create final DTV column with measurement priority
if has_car_dtv_grid and source_col:
    mask_measured = output_df[source_col] == 'measured'
    
    # Start with estimated values
    output_df['Car_DTV_Grid_Final'] = aadt_grid_estimated.copy()
    
    # Overwrite with measured values where available
    output_df.loc[mask_measured, 'Car_DTV_Grid_Final'] = output_df.loc[mask_measured, 'Car_DTV_Grid']
    
    # Set source
    output_df['Car_Data_Source_Final'] = 'estimated_from_grid_classification'
    output_df.loc[mask_measured, 'Car_Data_Source_Final'] = 'measured'
    
    print(f"\n✓ Car_DTV_Grid_Final created:")
    print(f"  - {mask_measured.sum():,} from measured values (Car_DTV_Grid)")
    print(f"  - {(~mask_measured).sum():,} from grid classification estimates")
else:
    # No measured values available
    output_df['Car_DTV_Grid_Final'] = aadt_grid_estimated
    output_df['Car_Data_Source_Final'] = 'estimated_from_grid_classification'
    print(f"\n✓ Car_DTV_Grid_Final created (all from grid classification)")

print("✓ Car_Data_Source_Final created")

# Remove internal ID
if 'Grid_ID_Internal' in output_df.columns:
    output_df = output_df.drop(columns=['Grid_ID_Internal'])

# ===================================================================
# STEP 7: EXPORT RESULTS
# ===================================================================
print_header('STEP 7: Export Results')

# Reorder columns
priority_cols = ['LINREFX', 'LINREFY', 'Street_Type_Grid', 'Car_DTV_Grid_Final', 'Car_Data_Source_Final',
                 'Car_DTV_Grid', source_col, 'AADT_Grid_Estimated']
cols_to_front = [col for col in priority_cols if col in output_df.columns]
other_cols = [col for col in output_df.columns if col not in cols_to_front]
output_df = output_df[cols_to_front + other_cols]

# Save
output_csv = ACCIDENTS_CSV.replace('.csv', '_grid_classified.csv')
output_df.to_csv(output_csv, sep=';', decimal=',', index=False)

print(f"\n✓ File saved: {output_csv}")
print(f"✓ Total rows: {len(output_df):,}")
print(f"✓ Total columns: {len(output_df.columns)}")

# ===================================================================
# FINAL STATISTICS
# ===================================================================
print_header('FINAL STATISTICS')

print("\nGrid Classification Distribution:")
for cls in class_hierarchy:
    count = (output_df['Street_Type_Grid'] == cls).sum()
    if count > 0:
        pct = count / len(output_df) * 100
        print(f"  {cls:25s}: {count:6,} ({pct:5.1f}%)")

if 'Car_Data_Source_Final' in output_df.columns:
    print("\nCar DTV Data Source Distribution:")
    for source, count in output_df['Car_Data_Source_Final'].value_counts().items():
        pct = count / len(output_df) * 100
        print(f"  {source:40s}: {count:6,} ({pct:5.1f}%)")

print_header('✓ SCRIPT COMPLETED SUCCESSFULLY')