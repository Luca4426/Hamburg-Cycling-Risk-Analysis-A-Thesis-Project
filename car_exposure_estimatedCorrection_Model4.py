# ===================================================================
# MASTER THESIS
# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)
# SCRIPT: Car Exposure Estimation - Point-based Classification with Measurement Correction
# DATE: 23.11.2025
# DESCRIPTION: Loads accident data with existing DTV values.
#              Re-adjusts estimated DTV values using point-based OSM classification.
#              Matches each accident to nearest road (within 25m) and generates improved
#              DTV estimates from OSM road type. Keeps measured values unchanged, 
#              replaces previous estimates with refined point-level values (second adjustment).
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
print_header('CAR EXPOSURE - POINT CLASSIFICATION WITH MEASUREMENT CORRECTION')

# Input path
ACCIDENTS_CSV = "Acc_segbased_final.csv"

# CRS
TARGET_CRS = 'EPSG:25832'

# Maximum distance for road matching
MAX_DISTANCE = 25  # meters

# 4-category road classification
aadt_mapping = {
    'Highway': 30000, 
    'Main Road': 10000, 
    'Connector Road': 3000,
    'Residential Street': 800, 
    'Not Found': 500, 
    'Off Street': 0
}

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
print(f"✓ Loaded {len(df):,} accident records")

# ===================================================================
# STEP 2: CREATE GEOMETRIES
# ===================================================================
print_header('STEP 2: Create Point Geometries')

df['geometry'] = df.apply(lambda row: Point(row['LINREFX'], row['LINREFY']), axis=1)
gdf_accidents = gpd.GeoDataFrame(df, geometry='geometry', crs=TARGET_CRS)
print(f"✓ Created {len(gdf_accidents):,} point geometries")

# ===================================================================
# STEP 3: DOWNLOAD OSM ROAD NETWORK
# ===================================================================
print_header('STEP 3: Download OSM Road Network')

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
# STEP 4: CLASSIFY ROADS BY HIGHWAY TYPE
# ===================================================================
print_header('STEP 4: Classify Road Segments')

# Helper function to classify highway tags
def classify_highway(highway_tag):
    if highway_tag is None or (isinstance(highway_tag, float) and pd.isna(highway_tag)):
        return 'Not Found'
    if isinstance(highway_tag, list):
        if len(highway_tag) == 0:
            return 'Not Found'
        highway_tag = highway_tag[0]
    highway_str = str(highway_tag).lower().strip()
    return highway_classification.get(highway_str, 'Not Found')

print("⏳ Classifying road segments...")
gdf_roads['Street_Type_Classified'] = gdf_roads['highway'].apply(classify_highway)
gdf_roads['AADT_Estimated'] = gdf_roads['Street_Type_Classified'].map(aadt_mapping)
print(f"✓ Classified {len(gdf_roads):,} road segments")

# ===================================================================
# STEP 5: SPATIAL JOIN - NEAREST ROAD MATCHING
# ===================================================================
print_header('STEP 5: Match Accidents to Nearest Roads')

print(f"⏳ Finding nearest roads (max distance: {MAX_DISTANCE}m)...")
gdf_accidents['original_index'] = gdf_accidents.index
gdf_joined = gpd.sjoin_nearest(
    gdf_accidents,
    gdf_roads[['geometry', 'Street_Type_Classified', 'AADT_Estimated', 'highway']],
    how='left', 
    max_distance=MAX_DISTANCE, 
    distance_col='Distance_to_Road'
)

# Remove duplicates (keep first match)
gdf_joined = gdf_joined.drop_duplicates(subset=['original_index'], keep='first')
gdf_joined = gdf_joined.drop(columns=['original_index'])

matched = gdf_joined['Distance_to_Road'].notna().sum()
print(f"✓ {matched:,} accidents matched to roads within {MAX_DISTANCE}m")

# ===================================================================
# STEP 6: CREATE FINAL CLASSIFICATION COLUMNS
# ===================================================================
print_header('STEP 6: Create Final Classification Columns')

# 1. Street_Type_New
print("⏳ Creating Street_Type_New...")
mask_no_road = gdf_joined['Distance_to_Road'].isna()
mask_not_classified = (~mask_no_road) & (gdf_joined['Street_Type_Classified'] == 'Not Found')
mask_classified = (~mask_no_road) & (gdf_joined['Street_Type_Classified'] != 'Not Found')

gdf_joined['Street_Type_New'] = 'Not Found'
gdf_joined.loc[mask_no_road, 'Street_Type_New'] = 'Off Street'
gdf_joined.loc[mask_not_classified, 'Street_Type_New'] = 'Not Found'
gdf_joined.loc[mask_classified, 'Street_Type_New'] = gdf_joined.loc[mask_classified, 'Street_Type_Classified']
print("✓ Street_Type_New created")

# 2. Calculate estimated AADT from classification
aadt_estimated = gdf_joined['Street_Type_New'].map(aadt_mapping)

# 3. Create Car_DTV_Final with measurement priority
has_car_dtv = 'Car_DTV' in gdf_joined.columns
has_source = 'Car_DTV_Source' in gdf_joined.columns

print(f"\nChecking for measured values:")
print(f"  Car_DTV column: {'FOUND' if has_car_dtv else 'NOT FOUND'}")
print(f"  Car_DTV_Source column: {'FOUND' if has_source else 'NOT FOUND'}")

if has_car_dtv and has_source:
    # Both columns exist - prioritize measured values
    mask_measured = gdf_joined['Car_DTV_Source'] == 'measured'
    
    # Start with estimated values
    gdf_joined['Car_DTV_Final'] = aadt_estimated.copy()
    
    # Overwrite with measured values where available
    gdf_joined.loc[mask_measured, 'Car_DTV_Final'] = gdf_joined.loc[mask_measured, 'Car_DTV']
    
    # Set source
    gdf_joined['Car_DTV_Final_Source'] = 'estimated_from_classification'
    gdf_joined.loc[mask_measured, 'Car_DTV_Final_Source'] = 'measured'
    
    print(f"\n✓ Car_DTV_Final created:")
    print(f"  - {mask_measured.sum():,} from measured values (Car_DTV)")
    print(f"  - {(~mask_measured).sum():,} from point classification estimates")
else:
    # No measured values available
    gdf_joined['Car_DTV_Final'] = aadt_estimated
    gdf_joined['Car_DTV_Final_Source'] = 'estimated_from_classification'
    print(f"\n✓ Car_DTV_Final created (all from point classification)")

print("✓ Car_DTV_Final_Source created")

# ===================================================================
# STEP 7: EXPORT RESULTS
# ===================================================================
print_header('STEP 7: Export Results')

# Convert to DataFrame
output_df = pd.DataFrame(gdf_joined.drop(columns='geometry'))

# Reorder columns - new columns first
priority_cols = ['LINREFX', 'LINREFY', 'Street_Type_New', 'Car_DTV_Final', 'Car_DTV_Final_Source',
                 'Car_DTV', 'Car_DTV_Source', 'Distance_to_Road']
cols_to_front = [col for col in priority_cols if col in output_df.columns]
other_cols = [col for col in output_df.columns if col not in cols_to_front]
output_df = output_df[cols_to_front + other_cols]

# Save
output_csv = ACCIDENTS_CSV.replace('.csv', '_classified.csv')
output_df.to_csv(output_csv, sep=';', decimal=',', index=False)

print(f"\n✓ File saved: {output_csv}")
print(f"✓ Total rows: {len(output_df):,}")
print(f"✓ Total columns: {len(output_df.columns)}")

print(f"\nFirst 10 columns in output:")
for i, col in enumerate(output_df.columns[:10]):
    print(f"  {i+1}. {col}")

# ===================================================================
# FINAL STATISTICS
# ===================================================================
print_header('FINAL STATISTICS')

print("\nStreet Classification Distribution:")
for cat in ['Highway', 'Main Road', 'Connector Road', 'Residential Street', 'Off Street', 'Not Found']:
    count = (output_df['Street_Type_New'] == cat).sum()
    if count > 0:
        pct = count / len(output_df) * 100
        print(f"  {cat:25s}: {count:6,} ({pct:5.1f}%)")

if 'Car_DTV_Final_Source' in output_df.columns:
    print("\nCar DTV Data Source Distribution:")
    for source, count in output_df['Car_DTV_Final_Source'].value_counts().items():
        pct = count / len(output_df) * 100
        print(f"  {source:40s}: {count:6,} ({pct:5.1f}%)")

print_header('✓ SCRIPT COMPLETED SUCCESSFULLY')