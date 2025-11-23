# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Dynamic OSM Feature Enrichment (Segment-based)
# DATE: 20.10.2025
# DESCRIPTION: Loads accident data and the OSM road network.
# Critically, it calculates time-correct "dynamic" features.
# For each accident in year Y, it calculates historical accident
# clusters (e.g., 'Acc_hist_junction_bike') using only data from years <Y.
# It then enriches each accident point with the features of the
# nearest OSM street segment and junction.
#
# ===================================================================

import pandas as pd
import geopandas as gpd
import osmnx as ox
import warnings
from tqdm import tqdm

# Disable minor warnings
warnings.filterwarnings('ignore')

# --- Helper Functions ---

def print_header(title):
    """Helper function for clean terminal output."""
    print('-' * 70)
    print(title)
    print('-' * 70)

def get_street_type(v):
    """Classifies the OSM highway type into Main or Residential road."""
    # This function creates 'Main_road'/'Residential_road', 
    # but the output column will be renamed to 'StrTyp'
    if isinstance(v, (list, tuple)) and len(v) > 0:
        v = v[0]
    if isinstance(v, str):
        if v in ['primary', 'secondary', 'tertiary', 'primary_link', 'secondary_link', 'tertiary_link']:
            return 'Main_road'
        if v in ['residential', 'living_street', 'unclassified', 'service', 'road']:
            return 'Residential_road'
    return 'Unknown'

def get_max_speed(v):
    """Extracts the numerical speed limit from the OSM tag."""
    if isinstance(v, (list, tuple)) and len(v) > 0:
        v = v[0]
    if pd.isna(v):
        return None
    try:
        s = str(v).split(' ')[0]
        return int(s)
    except:
        return None

# -------------------
# STEP 1: CONFIGURATION
# -------------------
print_header("OSM FEATURE ENRICHMENT - FINAL VERSION (with Dynamic History)")

INPUT_CSV = r"C:\Users\lucad\OneDrive - HafenCity Universität Hamburg\Studium_HCU\MA06\Thesis\Unfallstudie\Unfallstudy_Maschinelearn\seg base\New\unfaelle_enriched_V4_complete_withProxy.csv"
OUTPUT_CSV = INPUT_CSV.replace('.csv', '_final_enriched_dynamic.csv')

CRS_METRIC = 'EPSG:25832'
JUNCTION_RADIUS = 35       
ACCIDENT_CLUSTER_RADIUS = 15 

# -------------------
# STEP 2: LOAD DATA
# -------------------
print_header("STEP 2: Load Accident Data and OSM Network")

print("  Loading accident data...")
try:
    df = pd.read_csv(INPUT_CSV, sep=';', decimal=',')
    gdf_accidents_all = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.LINREFX, df.LINREFY), crs=CRS_METRIC
    )
    print(f"✓ Successfully loaded: {len(gdf_accidents_all)} accidents.")
except FileNotFoundError:
    print(f"ERROR: File not found! Please check the path:\n{INPUT_CSV}")
    exit()

print("  Loading OSM data for Hamburg...")
G = ox.graph_from_place('Hamburg, Germany', network_type='drive', simplify=True)
G = ox.project_graph(G, to_crs=CRS_METRIC)
nodes, edges = ox.graph_to_gdfs(G)
print(f"✓ OSM network loaded ({len(nodes)} nodes, {len(edges)} edges).")

# -------------------
# STEP 3: CALCULATE INITIAL JUNCTION FEATURES
# -------------------
print_header("STEP 3: Calculate Initial Junction Features")
# KORRIGIERT: Renamed 'complexity' to 'complexity_junction'
nodes['complexity_junction'] = nodes['street_count']
junctions = nodes[nodes['complexity_junction'] > 2].copy()
print(f"✓ {len(junctions)} junctions (complexity_junction > 2) identified.")


# =================================================================================
# --- STEP 4: DYNAMIC & TEMPORALLY-CORRECT FEATURE CALCULATION ---
# =================================================================================
print_header("STEP 4: Dynamic, Temporally-Correct Feature Calculation")

enriched_yearly_dfs = []
all_years = sorted(gdf_accidents_all['UJAHR'].unique())

for year in tqdm(all_years, desc="Processing years"):
    
    current_year_accidents = gdf_accidents_all[gdf_accidents_all['UJAHR'] == year].copy()
    historic_accidents = gdf_accidents_all[gdf_accidents_all['UJAHR'] < year].copy()
    historic_rad_accidents = historic_accidents[historic_accidents['UN_Rad'] > 0].copy()

    if current_year_accidents.empty:
        continue

    # --- Feature 1: KORRIGIERT: Renamed to 'Bike_Buffer_15m' ---
    if historic_rad_accidents.empty:
        current_year_accidents['Bike_Buffer_15m'] = 0
    else:
        buffer_current_year = current_year_accidents.copy()
        buffer_current_year['geometry'] = buffer_current_year.geometry.buffer(ACCIDENT_CLUSTER_RADIUS)
        joined = gpd.sjoin(buffer_current_year, historic_rad_accidents, how='left', predicate='intersects')
        counts = joined.groupby(joined.index)['index_right'].count()
        current_year_accidents['Bike_Buffer_15m'] = current_year_accidents.index.map(counts).fillna(0).astype(int)

    # --- Feature 2: KORRIGIERT: Renamed to 'Acc_hist_junction_bike' ---
    temp_junctions = junctions.copy()
    if historic_rad_accidents.empty:
        temp_junctions['temp_hist_count'] = 0
    else:
        junctions_buffered = temp_junctions.copy()
        junctions_buffered['geometry'] = junctions_buffered.geometry.buffer(JUNCTION_RADIUS)
        joined_junctions = gpd.sjoin(junctions_buffered, historic_rad_accidents, how='left', predicate='intersects')
        junction_counts = joined_junctions.groupby(joined_junctions.index)['index_right'].count()
        temp_junctions['temp_hist_count'] = temp_junctions.index.map(junction_counts).fillna(0).astype(int)
    
    # Link current year's accidents to junctions and their *current* history
    gdf_enriched_year = gpd.sjoin_nearest(
        current_year_accidents, 
        # KORRIGIERT: Use 'complexity_junction'
        temp_junctions[['geometry', 'complexity_junction', 'temp_hist_count']], 
        how="left", 
        # KORRIGIERT: Use 'dist_junction'
        distance_col="dist_junction" 
    )
    gdf_enriched_year = gdf_enriched_year[~gdf_enriched_year.index.duplicated(keep='first')]
    # KORRIGIERT: Rename to 'Acc_hist_junction_bike'
    gdf_enriched_year.rename(columns={'temp_hist_count': 'Acc_hist_junction_bike'}, inplace=True)
    gdf_enriched_year['Acc_hist_junction_bike'].fillna(0, inplace=True)
    
    enriched_yearly_dfs.append(gdf_enriched_year)

# ===================================================================
# --- STEP 5: FINAL MERGE AND ENRICHMENT ---
# ===================================================================
print_header("STEP 5: Concatenate All Years and Finalize Features")

if not enriched_yearly_dfs:
    print("ERROR: No data was processed. Exiting.")
    exit()

gdf_enriched_final = pd.concat(enriched_yearly_dfs).sort_index()

# Add OSM edge features (street type, speed limit)
edge_cols = ['highway', 'geometry']
if 'maxspeed' in edges.columns:
    edge_cols.append('maxspeed')

gdf_enriched_final = gpd.sjoin_nearest(
    gdf_enriched_final, edges[edge_cols], how="left", distance_col="dist_segment"
)
gdf_enriched_final = gdf_enriched_final.drop(columns=['index_right'], errors='ignore')
gdf_enriched_final = gdf_enriched_final[~gdf_enriched_final.index.duplicated(keep='first')]

# KORRIGIERT: Rename columns to 'StrTyp' and 'speedlimit'
gdf_enriched_final['StrTyp'] = gdf_enriched_final['highway'].apply(get_street_type)
if 'maxspeed' in gdf_enriched_final.columns:
    gdf_enriched_final['speedlimit'] = gdf_enriched_final['maxspeed'].apply(get_max_speed)
else:
    gdf_enriched_final['speedlimit'] = None

print("✓ Dynamic features and OSM data successfully merged.")

# -------------------
# STEP 6: SAVE ENRICHED DATA
# -------------------
print_header("STEP 6: Save Enriched Data")

# KORRIGIERT: Use the exact column list you provided
export_cols = list(df.columns) + [
    'dist_junction', 
    'complexity_junction', 
    'Acc_hist_junction_bike',
    'StrTyp', 
    'speedlimit', 
    'Bike_Buffer_15m'
]

# Ensure we only select columns that actually exist
final_cols_to_export = [col for col in export_cols if col in gdf_enriched_final.columns]
missing_cols = set(export_cols) - set(final_cols_to_export)
if missing_cols:
    print(f"Warning: The following new columns could not be found: {missing_cols}")

df_export = gdf_enriched_final[final_cols_to_export]
df_export.to_csv(OUTPUT_CSV, index=False, sep=';', decimal=',')

print(f"\nSUCCESS! File '{OUTPUT_CSV}' created.")
# KORRIGIERT: Update print statement with new names
print(f"Included dynamic features: 'Acc_hist_junction_bike', 'Bike_Buffer_15m'")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*70)