# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Bicycle Exposure Mapping to OSM Segments
# DATE: 20.10.2025
# DESCRIPTION: Loads accident data, the OSM street network, and bicycle GPS/track data.
# It spatially maps the bicycle traffic data (from multiple years) to the
# nearest OSM street segment (within a 20m buffer) to create an
# average exposure score per segment.
# Finally, it attaches this segment-based exposure score to each
# accident point based on proximity.
# ===================================================================

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
import osmnx as ox
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

BASE_DIR = r"C:\Users\lucad\analyse_python\erweiterung_study\erweiterung_Data\Infra und Verkhserweiterung"
os.chdir(BASE_DIR)

ACCIDENT_CSV = "unfaelle_mit_strassennetz.csv"
BIKE_TRAFFIC_CSVS = [
    "de_hh_up_jahr2022_EPSG_25832.csv",
    "de_hh_up_jahr2023_EPSG_25832.csv",
    "de_hh_up_stadtradeln2018_EPSG_25832.csv",
    "de_hh_up_stadtradeln2019_EPSG_25832.csv",
    "de_hh_up_stadtradeln2020_EPSG_25832.csv",
]
MATCH_BUFFER_METERS = 20
CRS_METRIC = "EPSG:25832"
OUTPUT_CSV = "unfaelle_mit_osm_radexp.csv"

print(f"✓ Base directory set to: {BASE_DIR}")
print(f"✓ Accident input: {ACCIDENT_CSV}")
print(f"✓ Output file: {OUTPUT_CSV}")

# -------------------
# STEP 2: LOAD BICYCLE TRAFFIC DATA
# -------------------
print_header("STEP 2: Load and Aggregate Bicycle Traffic Data")

rad_list = []
for f in BIKE_TRAFFIC_CSVS:
    try:
        df = pd.read_csv(f, sep=";")
        if "geom" not in df.columns:
            print(f"{f}: no 'geom' column found, skipping.")
            continue
            
        df["geometry"] = df["geom"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_METRIC)
        
        if "anzahl" in gdf.columns:
            gdf = gdf[["anzahl","geometry"]]
        else:
            gdf["anzahl"] = 1 # Assume count = 1 if no 'anzahl' column
            
        rad_list.append(gdf)
        print(f"{f}: {len(gdf):,} lines loaded.")
    except Exception as e:
        print(f"ERROR loading {f}: {e}")

if not rad_list:
    raise ValueError("No bicycle traffic CSVs were loaded! Check paths.")

rad_all = pd.concat(rad_list, ignore_index=True)
print(f"Total: {len(rad_all):,} bicycle traffic lines aggregated.")

# -------------------
# STEP 3: LOAD ACCIDENTS & OSM NETWORK
# -------------------
print_header("STEP 3: Load Accident Data and OSM Network")

udf = pd.read_csv(ACCIDENT_CSV, sep=";")
# Ensure coordinate columns are numeric
for col in ["LINREFX","LINREFY"]:
    if col in udf.columns:
        udf[col] = pd.to_numeric(udf[col].astype(str).str.replace(",","."), errors="coerce")

if not {"LINREFX","LINREFY"}.issubset(udf.columns):
    raise ValueError("Input CSV requires 'LINREFX' and 'LINREFY' columns.")

udf = udf.dropna(subset=["LINREFX","LINREFY"])
udf = udf[(udf["LINREFX"]!=0) & (udf["LINREFY"]!=0)]

ugdf = gpd.GeoDataFrame(udf, geometry=gpd.points_from_xy(udf["LINREFX"], udf["LINREFY"]), crs=CRS_METRIC)
ugdf = ugdf[ugdf.geometry.is_valid & ~ugdf.geometry.is_empty].copy()

if len(ugdf) == 0:
    raise ValueError("No valid accident coordinates found after cleaning!")

print(f"{len(ugdf):,} valid accidents loaded.")

# Get OSM data for the area covering all accidents
bounds = ugdf.total_bounds
convex_utm = box(*bounds).buffer(500)
convex_wgs84 = gpd.GeoSeries([convex_utm], crs=CRS_METRIC).to_crs("EPSG:4326").iloc[0]

print("  Fetching OSM 'drive' network, please wait...")
G = ox.graph_from_polygon(convex_wgs84, network_type="drive")
edges_osm = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True).to_crs(CRS_METRIC)
edges_osm = edges_osm.reset_index()
edges_osm["edge_id"] = edges_osm.apply(lambda r: f"{r['u']}_{r['v']}_{r['key']}", axis=1)
print(f"{len(edges_osm):,} OSM segments loaded.")

# -------------------
# STEP 4: MATCH BIKE LINES -> OSM SEGMENTS
# -------------------
print_header("STEP 4: Match Bike Lines to OSM Segments")

# Buffer OSM edges to create a search area
edges_buf = edges_osm.copy()
edges_buf["geometry"] = edges_buf.geometry.buffer(MATCH_BUFFER_METERS)

# Join bike lines that intersect the buffered OSM segments
join = gpd.sjoin(rad_all[["anzahl","geometry"]], edges_buf[["edge_id","geometry"]], how="inner", predicate="intersects")

# Calculate the AVERAGE 'anzahl' for all bike lines associated with each edge
rad_per_edge = join.groupby("edge_id")["anzahl"].mean().rename("RAD_ANZAHL_AVG").reset_index()

# Merge the calculated average back to the main OSM edges
edges_osm = edges_osm.merge(rad_per_edge, on="edge_id", how="left")
edges_osm["RAD_ANZAHL_AVG"] = edges_osm["RAD_ANZAHL_AVG"].fillna(0.0)
edges_osm["RAD_ANZAHL_LOG"] = np.log1p(edges_osm["RAD_ANZAHL_AVG"])

print(f"{len(rad_per_edge):,} OSM segments matched with bike data.")

# -------------------
# STEP 5: MAP ACCIDENTS -> ENRICHED OSM SEGMENTS
# -------------------
print_header("STEP 5: Map Accidents to Enriched OSM Segments")

# We need the original index of the accidents to join the data back
ugdf_indexed = ugdf[["geometry"]].copy().reset_index(drop=False)

# Find the NEAREST OSM segment for each accident
unfall_nearest = gpd.sjoin_nearest(
    ugdf_indexed,
    edges_osm[["edge_id","geometry"]].copy(),
    how="left", distance_col="dist_edge_m"
)

# Keep only the single closest match for each accident
unfall_nearest = unfall_nearest.drop_duplicates(subset=["index"], keep="first")

# Now, merge the bike exposure data using the matched 'edge_id'
uexpo = unfall_nearest.merge(
    edges_osm[["edge_id","RAD_ANZAHL_AVG","RAD_ANZAHL_LOG"]],
    on="edge_id", how="left"
)

# Sort by the original index to ensure correct row alignment
uexpo = uexpo.sort_values("index").reset_index(drop=True)

# Assign the new exposure features back to the original accident DataFrame
udf["RAD_ANZAHL_AVG"] = uexpo["RAD_ANZAHL_AVG"].fillna(0.0).values
udf["RAD_ANZAHL_LOG"] = uexpo["RAD_ANZAHL_LOG"].fillna(0.0).values
udf["DIST_EDGE_M"] = uexpo["dist_edge_m"].values

print(f"{len(udf)} accidents successfully enriched with segment exposure data.")

# -------------------
# STEP 6: SAVE FINAL FILE
# -------------------
print_header("STEP 6: Save Final Enriched Accident File")

udf.to_csv(OUTPUT_CSV, index=False, sep=";")
print(f"Complete: {OUTPUT_CSV} saved with {len(udf):,} accidents.")
print(f"  Median Bike_Exp_AVG: {udf['RAD_ANZAHL_AVG'].median():.2f}")
print(f"  Median Bike_Exp_LOG: {udf['RAD_ANZAHL_LOG'].median():.2f}")

print("\n" + "="*70)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*70)