import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
accidents_csv = "Acc_segbased_final.csv"

# 4 GROUPS WITH DTV VALUES
aadt_mapping = {
    'Highway': 30000, 'Main Road': 10000, 'Connector Road': 3000,
    'Residential Street': 800, 'Not Found': 500, 'Off Street': 0
}

# OSM HIGHWAY TAGS → 4 GROUPS
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
print("ROAD CLASSIFICATION - 4 GROUPS + FINAL DTV")
print("="*70)

# === LOAD & PROCESS ===
print("\n[1/5] Loading accidents...")
df = pd.read_csv(accidents_csv, sep=';', decimal=',')
print(f"    ✓ {len(df):,} accidents loaded")

print("\n[2/5] Creating geometries...")
df['geometry'] = df.apply(lambda row: Point(row['LINREFX'], row['LINREFY']), axis=1)
gdf_accidents = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:25832')

print("\n[3/5] Downloading Hamburg roads from OSM...")
print("    ⏳ 2-3 minutes...")
G = ox.graph_from_place('Hamburg, Germany', network_type='drive')
gdf_roads = ox.graph_to_gdfs(G, nodes=False, edges=True)
gdf_roads = gdf_roads.to_crs('EPSG:25832')
print(f"    ✓ {len(gdf_roads):,} road segments")

print("\n[4/5] Classifying roads...")
def classify_highway(highway_tag):
    if highway_tag is None:
        return 'Not Found'
    if isinstance(highway_tag, float) and pd.isna(highway_tag):
        return 'Not Found'
    if isinstance(highway_tag, list):
        if len(highway_tag) == 0:
            return 'Not Found'
        highway_tag = highway_tag[0]
    highway_str = str(highway_tag).lower().strip()
    return highway_classification.get(highway_str, 'Not Found')

gdf_roads['Street_Type_Classified'] = gdf_roads['highway'].apply(classify_highway)
gdf_roads['AADT_Estimated'] = gdf_roads['Street_Type_Classified'].map(aadt_mapping)

print("\n[5/5] Spatial join...")
gdf_accidents['original_index'] = gdf_accidents.index
gdf_joined = gpd.sjoin_nearest(
    gdf_accidents,
    gdf_roads[['geometry', 'Street_Type_Classified', 'AADT_Estimated', 'highway']],
    how='left', max_distance=25, distance_col='Distance_to_Road'
)
gdf_joined = gdf_joined.drop_duplicates(subset=['original_index'], keep='first')
gdf_joined = gdf_joined.drop(columns=['original_index'])

# === CREATE NEW COLUMNS ===
print("\n" + "="*70)
print("CREATING NEW COLUMNS")
print("="*70)

# 1. Street_Type_New
mask_no_road = gdf_joined['Distance_to_Road'].isna()
mask_not_classified = (~mask_no_road) & (gdf_joined['Street_Type_Classified'] == 'Not Found')
mask_classified = (~mask_no_road) & (gdf_joined['Street_Type_Classified'] != 'Not Found')

gdf_joined['Street_Type_New'] = 'Not Found'
gdf_joined.loc[mask_no_road, 'Street_Type_New'] = 'Off Street'
gdf_joined.loc[mask_not_classified, 'Street_Type_New'] = 'Not Found'
gdf_joined.loc[mask_classified, 'Street_Type_New'] = gdf_joined.loc[mask_classified, 'Street_Type_Classified']

print(f"\n[1/3] Street_Type_New created ✓")

# 2. Calculate estimated AADT from classification
aadt_estimated = gdf_joined['Street_Type_New'].map(aadt_mapping)

# 3. Create Car_DTV_Final: measured if available, otherwise estimated
has_car_dtv = 'Car_DTV' in gdf_joined.columns
has_source = 'Car_DTV_Source' in gdf_joined.columns

print(f"\n[2/3] Creating Car_DTV_Final...")
print(f"    Original Car_DTV column: {'FOUND' if has_car_dtv else 'NOT FOUND'}")
print(f"    Original Car_DTV_Source column: {'FOUND' if has_source else 'NOT FOUND'}")

if has_car_dtv and has_source:
    # Both columns exist - use measured where available
    mask_measured = gdf_joined['Car_DTV_Source'] == 'measured'
    
    gdf_joined['Car_DTV_Final'] = aadt_estimated.copy()
    gdf_joined.loc[mask_measured, 'Car_DTV_Final'] = gdf_joined.loc[mask_measured, 'Car_DTV']
    
    gdf_joined['Car_DTV_Final_Source'] = 'estimated_from_classification'
    gdf_joined.loc[mask_measured, 'Car_DTV_Final_Source'] = 'measured'
    
    print(f"    ✓ Created with mixed sources:")
    print(f"      - {mask_measured.sum():,} from measured values")
    print(f"      - {(~mask_measured).sum():,} from classification")
else:
    # Use only estimates
    gdf_joined['Car_DTV_Final'] = aadt_estimated
    gdf_joined['Car_DTV_Final_Source'] = 'estimated_from_classification'
    print(f"    ✓ Created from classification only")

print(f"\n[3/3] Car_DTV_Final_Source created ✓")

# Verify
print(f"\n" + "="*70)
print("COLUMN VERIFICATION")
print("="*70)
print(f"  Street_Type_New:       {'✓' if 'Street_Type_New' in gdf_joined.columns else '✗ MISSING'}")
print(f"  Car_DTV_Final:         {'✓' if 'Car_DTV_Final' in gdf_joined.columns else '✗ MISSING'}")
print(f"  Car_DTV_Final_Source:  {'✓' if 'Car_DTV_Final_Source' in gdf_joined.columns else '✗ MISSING'}")

# === EXPORT ===
print(f"\n" + "="*70)
print("EXPORTING")
print("="*70)

output_df = pd.DataFrame(gdf_joined.drop(columns='geometry'))

# NEW columns first
priority_cols = ['LINREFX', 'LINREFY', 'Street_Type_New', 'Car_DTV_Final', 'Car_DTV_Final_Source', 
                 'Car_DTV', 'Car_DTV_Source', 'Distance_to_Road']
cols_to_front = [col for col in priority_cols if col in output_df.columns]
other_cols = [col for col in output_df.columns if col not in cols_to_front]
output_df = output_df[cols_to_front + other_cols]

output_csv = accidents_csv.replace('.csv', '_classified.csv')
output_df.to_csv(output_csv, sep=';', decimal=',', index=False)

print(f"\n✓ File saved: {output_csv}")
print(f"✓ Total rows: {len(output_df):,}")
print(f"✓ Total columns: {len(output_df.columns)}")

print(f"\nFirst columns in output:")
for i, col in enumerate(output_df.columns[:10]):
    print(f"  {i+1}. {col}")

print(f"\n" + "="*70)
print("✓ COMPLETED SUCCESSFULLY")
print("="*70)

# Summary statistics
print(f"\nClassification Summary:")
for cat in ['Highway', 'Main Road', 'Connector Road', 'Residential Street', 'Off Street']:
    count = (output_df['Street_Type_New'] == cat).sum()
    if count > 0:
        pct = count / len(output_df) * 100
        print(f"  {cat:25s}: {count:6,} ({pct:5.1f}%)")

if 'Car_DTV_Final_Source' in output_df.columns:
    print(f"\nCar_DTV_Final Sources:")
    for source, count in output_df['Car_DTV_Final_Source'].value_counts().items():
        pct = count / len(output_df) * 100
        print(f"  {source:30s}: {count:6,} ({pct:5.1f}%)")

print("="*70)