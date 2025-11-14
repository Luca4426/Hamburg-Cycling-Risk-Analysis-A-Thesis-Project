import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
accidents_csv = "Acc_250Grid_HH_enhanced.csv"
grid_gpkg = r"C:\Users\lucad\OneDrive - HafenCity Universität Hamburg\Studium_HCU\MA06\Thesis\Verschriftlichung\BikeAcc_ML\3 Enhanced\v3\Grid_HH_250x250.gpkg"

# 4 GROUPS
aadt_mapping = {
    'Highway': 30000, 'Main Road': 10000, 'Connector Road': 3000,
    'Residential Street': 800, 'Not Found': 500, 'Off Street': 0
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

print("="*70)
print("GRID-BASED CLASSIFICATION")
print("="*70)

# === LOAD ACCIDENTS ===
print("\n[1/6] Loading accidents...")
df = pd.read_csv(accidents_csv, sep=';', decimal=',')
df['geometry'] = df.apply(lambda row: Point(row['LINREFX'], row['LINREFY']), axis=1)
gdf_accidents = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:25832')
print(f"    ✓ {len(gdf_accidents):,} accidents")

# === LOAD GRID ===
print("\n[2/6] Loading grid...")
gdf_grid = gpd.read_file(grid_gpkg)
if gdf_grid.crs != 'EPSG:25832':
    gdf_grid = gdf_grid.to_crs('EPSG:25832')
gdf_grid['Grid_ID_Internal'] = range(len(gdf_grid))
print(f"    ✓ {len(gdf_grid):,} grid cells")

# === SPATIAL JOIN: ACCIDENTS → GRIDS ===
print("\n[3/6] Assigning accidents to grids...")
acc_with_grid = gpd.sjoin(gdf_accidents, gdf_grid[['Grid_ID_Internal', 'geometry']], how='left', predicate='within')

# === DOWNLOAD OSM ===
print("\n[4/6] Downloading OSM roads...")
print("    ⏳ 2-3 minutes...")
G = ox.graph_from_place('Hamburg, Germany', network_type='drive')
gdf_roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
gdf_roads = gdf_roads.to_crs('EPSG:25832')
print(f"    ✓ {len(gdf_roads):,} roads")

# === CLASSIFY ROADS ===
print("\n[5/6] Classifying roads...")
def classify_highway(highway_tag):
    if highway_tag is None or (isinstance(highway_tag, float) and pd.isna(highway_tag)):
        return 'Not Found'
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0] if highway_tag else 'Not Found'
    return highway_classification.get(str(highway_tag).lower().strip(), 'Not Found')

gdf_roads['Street_Type_Classified'] = gdf_roads['highway'].apply(classify_highway)

# === SPATIAL JOIN: ROADS → GRIDS ===
print("\n[6/6] Finding highest class per grid...")
print("    ⏳ Several minutes...")
roads_in_grids = gpd.sjoin(gdf_roads[['Street_Type_Classified', 'geometry']], 
                           gdf_grid[['Grid_ID_Internal', 'geometry']], 
                           how='inner', predicate='intersects')

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
print(f"    ✓ {len(grid_class):,} grids classified")

# === MERGE BACK ===
print("\nMerging to accidents...")
output_df = pd.DataFrame(acc_with_grid.drop(columns='geometry'))
output_df = output_df.merge(grid_class, on='Grid_ID_Internal', how='left')
output_df['Street_Type_Grid'].fillna('Off Street', inplace=True)
output_df['AADT_Grid_Estimated'].fillna(0, inplace=True)

# === CREATE NEW COLUMNS ===
print("\n" + "="*70)
print("CREATING NEW COLUMNS")
print("="*70)

print(f"\n[1/3] Street_Type_Grid created ✓")

# Calculate estimated AADT from grid classification
aadt_grid_estimated = output_df['Street_Type_Grid'].map(aadt_mapping)

# Find source and data columns
has_car_dtv_grid = 'Car_DTV_Grid' in output_df.columns  # ← CORRECT COLUMN NAME!
has_car_data_source = 'Car_Data_Source' in output_df.columns
has_car_dtv_source = 'Car_DTV_Source' in output_df.columns

source_col = 'Car_Data_Source' if has_car_data_source else ('Car_DTV_Source' if has_car_dtv_source else None)

print(f"\n[2/3] Creating Car_DTV_Grid_Final...")
print(f"    Car_DTV_Grid column: {'FOUND' if has_car_dtv_grid else 'NOT FOUND'}")
print(f"    Source column: {source_col if source_col else 'NOT FOUND'}")

if has_car_dtv_grid and source_col:
    # CORRECT LOGIC:
    # IF Car_Data_Source = "measured" → use Car_DTV_Grid
    # ELSE → use AADT_Grid_Estimated
    
    mask_measured = output_df[source_col] == 'measured'
    
    output_df['Car_DTV_Grid_Final'] = aadt_grid_estimated.copy()
    output_df.loc[mask_measured, 'Car_DTV_Grid_Final'] = output_df.loc[mask_measured, 'Car_DTV_Grid']
    
    output_df['Car_Data_Source_Final'] = 'estimated_from_grid_classification'
    output_df.loc[mask_measured, 'Car_Data_Source_Final'] = 'measured'
    
    print(f"    ✓ Created:")
    print(f"      - {mask_measured.sum():,} from Car_DTV_Grid (measured)")
    print(f"      - {(~mask_measured).sum():,} from AADT_Grid_Estimated")
else:
    output_df['Car_DTV_Grid_Final'] = aadt_grid_estimated
    output_df['Car_Data_Source_Final'] = 'estimated_from_grid_classification'
    print(f"    ✓ Created (all from grid classification)")

print(f"\n[3/3] Car_Data_Source_Final created ✓")

# Remove internal ID
if 'Grid_ID_Internal' in output_df.columns:
    output_df = output_df.drop(columns=['Grid_ID_Internal'])

# === EXPORT ===
print("\n" + "="*70)
print("EXPORTING")
print("="*70)

priority_cols = ['LINREFX', 'LINREFY', 'Street_Type_Grid', 'Car_DTV_Grid_Final', 'Car_Data_Source_Final',
                 'Car_DTV_Grid', source_col, 'AADT_Grid_Estimated']
cols_to_front = [col for col in priority_cols if col in output_df.columns]
other_cols = [col for col in output_df.columns if col not in cols_to_front]
output_df = output_df[cols_to_front + other_cols]

output_csv = accidents_csv.replace('.csv', '_grid_classified.csv')
output_df.to_csv(output_csv, sep=';', decimal=',', index=False)

print(f"\n✓ File saved: {output_csv}")
print(f"✓ Total rows: {len(output_df):,}")
print(f"✓ Total columns: {len(output_df.columns)}")

print("\n" + "="*70)
print("✓ COMPLETED")
print("="*70)

print(f"\nGrid Classification:")
for cls in class_hierarchy:
    count = (output_df['Street_Type_Grid'] == cls).sum()
    if count > 0:
        pct = count / len(output_df) * 100
        print(f"  {cls:25s}: {count:6,} ({pct:5.1f}%)")

if 'Car_Data_Source_Final' in output_df.columns:
    print(f"\nCar_Data_Source_Final:")
    for source, count in output_df['Car_Data_Source_Final'].value_counts().items():
        pct = count / len(output_df) * 100
        print(f"  {source:40s}: {count:6,} ({pct:5.1f}%)")

print("="*70)