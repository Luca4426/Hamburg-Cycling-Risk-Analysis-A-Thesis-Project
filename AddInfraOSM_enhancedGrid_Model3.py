# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: OSM Feature Enrichment - Hamburg 250m Grid
# DATE: 19.10.2025
# DESCRIPTION: Loads geometry data for the 250m grid and accident data.
# Downloads road network and infrastructure data from OpenStreetMap (OSM)
# and spatially aggregates it onto the grid cells.
#
# ===================================================================

import pandas as pd
import geopandas as gpd
import osmnx as ox
import numpy as np
from tqdm import tqdm
import warnings
from pyproj import Transformer
warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# Global Configuration
print_header('OSM FEATURE ENRICHMENT - Hamburg 250m Grid')
CSV_PATH = r"Acc_250Grid_HH_enhanced.csv"
GRID_GEOM_PATH = r"Grid_HH_250x250.gpkg"
GRID_SIZE = 250
TARGET_CRS = "EPSG:25832" # ETRS89 / UTM zone 32N

# -------------------
# STEP 1: LOAD DATA
# -------------------
print_header('STEP 1: Load Data (Accidents and Grid Geometry)')
try:
    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    print(f"✓ {len(df)} accident records loaded.")
    
    grid_geom = gpd.read_file(GRID_GEOM_PATH)
    print(f"✓ {len(grid_geom)} grid cells loaded from GeoPackage.")

except FileNotFoundError as e:
    print(f"ERROR: File not found. Stopping script.")
    print(e)
    exit()

# Ensure CRS
if grid_geom.crs is None:
    grid_geom = grid_geom.set_crs(TARGET_CRS)
    print(f"Warning: Grid geometry had no CRS. Setting to {TARGET_CRS}.")
elif grid_geom.crs != TARGET_CRS:
    grid_geom = grid_geom.to_crs(TARGET_CRS)
    print(f"✓ Grid geometry transformed to {TARGET_CRS}.")

# -------------------
# STEP 2: PREPARE GRID-ID
# -------------------
print_header('STEP 2: Prepare Grid-ID')

if 'Grid_ID' not in grid_geom.columns:
    print("  'Grid_ID' not found. Creating it from geometry centroids...")
    grid_geom['centroid'] = grid_geom.geometry.centroid
    grid_geom['Grid_X'] = (grid_geom['centroid'].x // GRID_SIZE) * GRID_SIZE
    grid_geom['Grid_Y'] = (grid_geom['centroid'].y // GRID_SIZE) * GRID_SIZE
    grid_geom['Grid_ID'] = (grid_geom['Grid_X'].astype(int).astype(str) + '_' + 
                           grid_geom['Grid_Y'].astype(int).astype(str))
    grid_geom = grid_geom.drop(columns=['centroid'])
    print(f"  ✓ 'Grid_ID' created successfully.")
else:
    print("  ✓ 'Grid_ID' already exists.")

# -------------------
# STEP 3: SPATIAL JOIN (Accidents to Grid)
# -------------------
print_header('STEP 3: Spatial Join (Accidents to Grid)')

df['LINREFX'] = pd.to_numeric(df['LINREFX'], errors='coerce')
df['LINREFY'] = pd.to_numeric(df['LINREFY'], errors='coerce')

# Create GeoDataFrame from accident data
df_geo = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['LINREFX'], df['LINREFY']),
    crs=TARGET_CRS
)

# Perform spatial join
df_with_grid = gpd.sjoin(df_geo, grid_geom[['Grid_ID', 'geometry']], 
                         how='left', predicate='within')

matched = (~df_with_grid['Grid_ID'].isna()).sum()
print(f"✓ {matched} of {len(df)} accidents matched to grid cells.")

# -------------------
# STEP 4: DOWNLOAD OSM DATA
# -------------------
print_header('STEP 4: Download OSM Data')

# Prepare Bounding Box for download (transform to EPSG:4326)
transformer = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
bounds = grid_geom.total_bounds
# 500m buffer to avoid edge effects
lon_min, lat_min = transformer.transform(bounds[0] - 500, bounds[1] - 500)
lon_max, lat_max = transformer.transform(bounds[2] + 500, bounds[3] + 500)
print(f"  Bounding Box for OSM download: {lat_min:.4f}-{lat_max:.4f}, {lon_min:.4f}-{lon_max:.4f}")

try:
    print("  Starting download of road network (may take 2-5 minutes)...")
    
    G = ox.graph_from_bbox(
        north=lat_max,
        south=lat_min,
        east=lon_max,
        west=lon_min,
        network_type='all',
        simplify=False
    )
    
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.to_crs(TARGET_CRS)
    edges = edges.to_crs(TARGET_CRS)
    print(f"✓ OSM download successful: {len(nodes)} nodes and {len(edges)} edges.")
    
except Exception as e:
    print(f"ERROR during download via Bounding Box: {e}")
    print("\n  Trying alternative download method (via place name)...")
    
    try:
        # Fallback: Load by city name
        G = ox.graph_from_place("Hamburg, Germany", network_type='all', simplify=False)
        nodes, edges = ox.graph_to_gdfs(G)
        nodes = nodes.to_crs(TARGET_CRS)
        edges = edges.to_crs(TARGET_CRS)
        print(f"✓ Alternative download method successful: {len(nodes)} nodes, {len(edges)} edges.")
    except Exception as e2:
        print(f"FATAL ERROR: Download failed: {e2}")
        exit()

# -------------------
# STEP 5: FEATURE CALCULATION (OSM)
# -------------------
print_header('STEP 5: Feature Calculation (OSM)')

edges_sindex = edges.sindex
nodes_sindex = nodes.sindex

grid_features = {}
print("  Calculating infrastructure features per grid cell...")

for idx in tqdm(grid_geom.index, desc="Processing grid cells"):
    
    grid_id = grid_geom.loc[idx, 'Grid_ID']
    grid_poly = grid_geom.loc[idx, 'geometry']
    
    features = {
        'OSM_Junction_Count': 0,
        'OSM_Road_Length_Total': 0.0,
        'OSM_Primary_Road_Length': 0.0,
        'OSM_Speed_Limit_Avg': np.nan,
        'OSM_Speed_Limit_Max': np.nan,
        'OSM_Road_Density': 0.0
    }
    
    # 1. Junctions (Nodes)
    possible_nodes_idx = list(nodes_sindex.intersection(grid_poly.bounds))
    if possible_nodes_idx:
        nodes_in_cell = nodes.iloc[possible_nodes_idx]
        nodes_in_cell = nodes_in_cell[nodes_in_cell.geometry.within(grid_poly)]
        features['OSM_Junction_Count'] = len(nodes_in_cell)
    
    # 2. Roads (Edges)
    possible_edges_idx = list(edges_sindex.intersection(grid_poly.bounds))
    if possible_edges_idx:
        roads_in_cell = edges.iloc[possible_edges_idx]
        # Only those that actually intersect (not just bounding box)
        roads_in_cell = roads_in_cell[roads_in_cell.geometry.intersects(grid_poly)]
        
        if len(roads_in_cell) > 0:
            # Clip segments to the cell
            road_segments = roads_in_cell.geometry.intersection(grid_poly)
            total_length = road_segments.length.sum()
            features['OSM_Road_Length_Total'] = total_length
            
            # Primary road length
            if 'highway' in roads_in_cell.columns:
                primary_types = ['primary', 'secondary', 'trunk',
                                 'primary_link', 'secondary_link', 'trunk_link']
                primary = roads_in_cell[roads_in_cell['highway'].isin(primary_types)]
                if len(primary) > 0:
                    primary_segments = primary.geometry.intersection(grid_poly)
                    features['OSM_Primary_Road_Length'] = primary_segments.length.sum()
            
            # Speed limits
            if 'maxspeed' in roads_in_cell.columns:
                speeds = []
                for speed in roads_in_cell['maxspeed'].dropna():
                    try:
                        # Clean 'maxspeed' (e.g., "50 km/h", "30")
                        speed_str = str(speed).split()[0].replace(',', '.')
                        speed_val = float(speed_str)
                        if 10 <= speed_val <= 200: # Plausibility check
                            speeds.append(speed_val)
                    except:
                        continue # Ignore invalid entries like 'none', 'signals'
                
                if speeds:
                    features['OSM_Speed_Limit_Avg'] = np.mean(speeds)
                    features['OSM_Speed_Limit_Max'] = np.max(speeds)
            
            # Road density (Length in m / Area in km²)
            grid_area_km2 = grid_poly.area / 1_000_000
            if total_length > 0 and grid_area_km2 > 0:
                features['OSM_Road_Density'] = total_length / grid_area_km2
    
    grid_features[grid_id] = features

print(f"\n✓ Features calculated for {len(grid_features)} grid cells.")

# -------------------
# STEP 6: MERGE FEATURES AND SAVE
# -------------------
print_header('STEP 6: Merge Features and Save')

# Convert feature dictionary to DataFrame
grid_features_df = pd.DataFrame.from_dict(grid_features, orient='index')
grid_features_df.index.name = 'Grid_ID'
grid_features_df = grid_features_df.reset_index()

# Merge features onto the main DataFrame
df_enriched = df_with_grid.merge(grid_features_df, on='Grid_ID', how='left')

# Clean up unnecessary columns
cols_to_drop = ['index_right', 'geometry']
df_enriched = df_enriched.drop(columns=[c for c in cols_to_drop if c in df_enriched.columns])

print(f"✓ OSM features merged onto {len(df_enriched)} records.")

# Show statistics of new features
print("\nStatistics of new OSM Features:")
print(30*"-")
for col in ['OSM_Junction_Count', 'OSM_Road_Length_Total', 
            'OSM_Primary_Road_Length', 'OSM_Speed_Limit_Avg']:
    if col in df_enriched.columns:
        non_zero = (df_enriched[col].fillna(0) > 0).sum()
        coverage = non_zero / len(df_enriched) * 100
        mean_val = df_enriched[col].mean()
        print(f"{col:25s}: {coverage:5.1f}% Coverage | Mean={mean_val:8.2f}")
    else:
        print(f"{col:25s}: Column not found.")
print(30*"-")

# Save
output_path = CSV_PATH.replace('.csv', '_with_OSM.csv')
try:
    df_enriched.to_csv(output_path, sep=';', decimal=',', index=False)
    print(f"\n✓ Enriched file saved to: {output_path}")
except PermissionError:
    print(f"\nERROR: Permission denied. Is the file '{output_path}' open?")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*7Z0)