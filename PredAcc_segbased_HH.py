# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SCRIPT: Enhanced Segment-based Model Hamburg
# DATE: 20.10.2025
# DESCRIPTION: Uses optimal threshold and adds infrastructure (source: OSM) 
# and traffic (source: Geoportal Hamburg) data as features 
# on OSM-Streetsegments, not in a Grid.
#
# BASED ON: 
#   AUTHORS: Sarah Di Grande, Mariaelena Berlotti, Salvatore Cavalieri and Daniel G. Costa
#   YEAR: 2025
#   TITLE: Data-Driven Prediction of High-Risk Situations for Cyclists Through Spatiotemporal Patterns and Environmental Conditions 
#   SOURCE: https://www.scitepress.org/Papers/2025/136464/136464.pdf (18.10.2025)
# ===================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
import warnings
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# Global Configuration
print_header('Enhanced Segment-based Model - F1 Optimization (Undersampling)')
CSV_PATH = "Acc_250Grid_HH_enhanced_final.csv"
TARGET_COLUMN = 'AccBike'  # Updated target variable
RANDOM_STATE = 42
TEST_YEAR = 2024

# -------------------
# STEP 1: LOAD DATA
# -------------------
print_header("STEP 1: Load and Prepare Data")
try:
    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    df.columns = df.columns.str.strip()
    print(f"{len(df)} rows loaded from CSV.")
except FileNotFoundError:
    print(f"ERROR: File not found at {CSV_PATH}.")
    exit()

# Ensure data types for coordinates
for col in ['LINREFX', 'LINREFY']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

# Define target variable
df['target'] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int)

# -------------------
# STEP 2: FEATURE ENGINEERING
# -------------------
print_header("STEP 2: Custom Feature Engineering")

# --- 2.1 Helper functions for mapping ---
def streettype_map(x):
    """Maps the road type 'StrTyp' to numerical values."""
    if pd.isna(x):
        return -1  # Missing value
    x_str = str(x).lower()
    if 'Primary' in x_str:
        return 1  # Main road
    elif 'Secondary' in x_str:
        return 0  # Unclassified/Residential/side road
    return -1  # Other

def tempotype_map(x):
    """Maps 'Speedlimit' to numerical categories."""
    if pd.isna(x):
        return 0  # Missing value or default
    if x <= 30:
        return 1  # Low speed limit (e.g., 30 km/h zone)
    elif x >= 50:
        return 2  # High speed limit (e.g., main road)
    return 0  # Other

# --- 2.2 Creation of new features ---

# Feature 1: ON_JUNCTION (binary)
# Defines an area as a junction if the distance to the next junction is < 35m.
if 'Dist_junction' in df.columns:
    df['ON_JUNCTION'] = (df['Dist_junction'] < 35).astype(int)
    print("Feature 'ON_JUNCTION' created.")
else:
    df['ON_JUNCTION'] = 0
    print("Warning: Column 'Dist_junction' not found. 'ON_JUNCTION' set to 0.")

# Feature 2 & 3: STREETTYPE and TEMPOTYPE (categorical)
if 'StrTyp' in df.columns:
    df['STREETTYPE'] = df['StrTyp'].apply(streettype_map)
    print("Feature 'STREETTYPE' created.")
else:
    df['STREETTYPE'] = -1 # KORRIGIERT (war STRASSENTYP)
    print("!! Warning: Column 'StrTyp' not found. 'STREETTYPE' set to -1.")

if 'Speedlimit' in df.columns:
    df['TEMPOTYPE'] = df['Speedlimit'].apply(tempotype_map)
    print("Feature 'TEMPOTYPE' created.")
else:
    df['TEMPOTYPE'] = 0
    print("Warning: Column 'Speedlimit' not found. 'TEMPOTYPE' set to 0.")

# --- 2.3 Creation of interaction features ---

# Feature 4: Interaction Junction and Road Type
df['JUNCTION_STREETTYPE_INTERACTION'] = df['ON_JUNCTION'] + df['STREETTYPE']
print("Interaction feature 'JUNCTION_StrTyp_INTERACTION' created.")

# Feature 5: Interaction Car Traffic and Speed Limit
if 'Car_DTV' in df.columns:
    df['CAR_SPEED_INTERACTION'] = df['Car_DTV'] * df['TEMPOTYPE']
    print("Interaction feature 'CAR_SPEED_INTERACTION' created.")
else:
    df['CAR_SPEED_INTERACTION'] = 0
    print("!! Warning: Column 'Car_DTV' not found. 'CAR_SPEED_INTERACTION' set to 0.")

# Feature 6: Traffic Density Proxy
if 'Car_exposure_proxy' in df.columns and 'Bike_exposure_proxy' in df.columns:
    df['TRAFFICDENSITY_PROXY'] = df['Car_exposure_proxy'] + df['Bike_exposure_proxy']
    print("Feature 'TRAFFICDENSITY_PROXY' created.")
else:
    df['TRAFFICDENSITY_PROXY'] = 0
    print("!! Warning: Proxy columns for traffic density not found.")

# Feature 7: Interaction Bike Traffic and Road Type
if 'Bike_exposure_proxy' in df.columns:
    df['BIKE_STREETTYPE_INTERACTION'] = df['Bike_Count_AVG'] * df['STREETTYPE']
    print("Interaction feature 'BIKE_STREETTYPE_INTERACTION' created.")
else:
    df['BIKE_STREETTYPE_INTERACTION'] = 0
    print("Warning: Column 'Bike_exposure_proxy' not found.")

# 2.3 Remove undesired columns
columns_to_drop = ['Car_DTV_Source','Car_exposure_proxy', 'Bike_exposure_proxy' ]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
print(f"Columns removed (if present): {columns_to_drop}")

print("Feature Engineering completed.")


# -------------------
# STEP 3: FEATURE SELECTION
# -------------------
print_header("STEP 3: Feature Selection")

# New feature list + the features created above
feature_cols = [
    'AMONTH', 'AWEEKDAY', 'AHOUR', 'LINREFX', 'LINREFY',
    'StrCon', 'LightCon', 'District', 'Bike_Count_AVG','Dist_Acc_to_Line',
    'Car_DTV', 'Dist_junction', 'Complexity_junction',
    'AccHist_junction_bike', 'StrTyp', 'BikeAcc_buffer_15m', 'Speedlimit', 

 # ---- Newly created features (from STEP 2) ----
    'TEMPOTYPE',
    'ON_JUNCTION',
    'JUNCTION_StrTyp_INTERACTION',
    'CAR_SPEED_INTERACTION',
    'TRAFFICDENSITY_PROXY',
    'BIKE_StrTyp_INTERACTION'
]

# Ensure only existing columns are used
final_feature_cols = [col for col in feature_cols if col in df.columns]
missing_cols = set(feature_cols) - set(final_feature_cols)

print(f"{len(final_feature_cols)} features selected for the model.")
if missing_cols:
    print(f"Warning: The following features were not found in the CSV: {list(missing_cols)}")


# -------------------
# STEP 4: TRAIN-TEST-SPLIT
# -------------------
print_header("STEP 4: Train-Test Split")
train = df[df['AYEAR'] < TEST_YEAR].copy()
test = df[df['AYEAR'] == TEST_YEAR].copy()

X_train = train[final_feature_cols]
y_train = train['target']
X_test = test[final_feature_cols]
y_test = test['target']

print(f"Training: {len(X_train)} | Test: {len(X_test)}")
print(f"Training set distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# -------------------
# STEP 5: APPLY UNDERSAMPLING
# -------------------
print_header("STEP 5: Apply Random Undersampling to Training Data")

# Initialize the RandomUnderSampler
rus = RandomUnderSampler(random_state=RANDOM_STATE)

print(f"Distribution before undersampling: {y_train.value_counts().to_dict()}")

# Apply undersampling
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print(f"Distribution after undersampling: {y_train_resampled.value_counts().to_dict()}")

# IMPORTANT: Overwrite old training variables with the new, balanced data
X_train = X_train_resampled
y_train = y_train_resampled


# -------------------
# STEP 6: CALCULATE CLASS WEIGHTS
# -------------------
print_header("STEP 6: Calculate Class Weights")
# Note: This is calculated on the *already undersampled* data
p1 = y_train.mean() if len(y_train) > 0 else 0.0
class_weights = [1 - p1, p1] if p1 > 0 else [1.0, 1.0]
print(f"Class weights: w0={class_weights[0]:.4f}, w1={class_weights[1]:.4f}")

# -------------------
# STEP 7: PREPROCESSING
# -------------------
print_header("STEP 7: Preprocessing (Categorical Features)")
categorical_features_potential = [
    'AMONTH', 'AWEEKDAY', 'StrCon', 'LightCon', 'District', 'StrTyp', 
    # ---- Newly added categorical features ----
    'ON_JUNCTION',
    'TEMPOTYPE'
]
categorical_features = [col for col in categorical_features_potential if col in X_train.columns]
print(f"Categorical features identified: {categorical_features}")

for col in categorical_features:
    X_train[col] = X_train[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

# -------------------
# STEP 8: MODEL TRAINING
# -------------------
print_header("STEP 8: CatBoost Training")
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_strength=1.5,
    bagging_temperature=0.8,
    loss_function='Logloss',
    eval_metric='AUC',
    random_state=RANDOM_STATE,
    verbose=200,
    class_weights=class_weights,
    early_stopping_rounds=150
)

model.fit(train_pool, eval_set=test_pool)
print("Training completed.")

# -------------------
# STEP 9: F1-OPTIMIZATION & EVALUATION
# -------------------
print_header("STEP 9: F1-Score Optimization and Evaluation")

# This block only runs if test data is available
if len(X_test) > 0:
    y_proba = model.predict_proba(test_pool)[:, 1]
    
    # Calculate AUC-ROC, if possible
    if len(np.unique(y_test)) > 1:
        auc_val = roc_auc_score(y_test, y_proba)
        print(f"Overall AUC-ROC: {auc_val:.4f}")
    else:
        auc_val = float('nan')
        print("AUC-ROC not calculable (only one class in test set).")

    # Calculate Precision, Recall, and F1-Score across all thresholds
    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    # Calculate F1-Score and avoid NaN values
    f1_scores = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1], 
                          out=np.zeros_like(prec[:-1]), where=(prec[:-1] + rec[:-1])!=0)
    
    # Find optimal threshold
    if len(f1_scores) > 0:
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thr[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
    else:
        optimal_threshold = 0.5
        optimal_f1 = float('nan')

    print(f"\nOptimal Threshold: {optimal_threshold:.3f}")
    print(f"Maximum F1-Score: {optimal_f1:.3f}")

    # Function for detailed evaluation at a specific threshold
    def eval_at(t):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return dict(th=t, rec=recall, prec=precision, f1=f1, cm=cm, tn=tn, fp=fp, fn=fn, tp=tp)

    # Calculate results with the optimal threshold
    result = eval_at(optimal_threshold)
    
    # Print final performance
    print("\n" + "="*40)
    print("Final Model Performance")
    print("="*40)
    print(f"Threshold:   {result['th']:.3f}")
    print(f"Recall:      {result['rec']:.3f} ({result['rec']*100:.1f}%)")
    print(f"Precision:   {result['prec']:.3f} ({result['prec']*100:.1f}%)")
    print(f"F1-Score:    {result['f1']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {result['tn']:6d} | FP: {result['fp']:6d}")
    print(f"  FN: {result['fn']:6d} | TP: {result['tp']:6d}")


# -------------------
# STEP 10: EXPORT PREDICTIONS FOR MAPPING
# -------------------
if len(X_test) > 0:
    print_header("STEP 10: Export Predictions for Mapping")

    # Create predictions DataFrame with all important columns
    predictions_export = test.copy()

    # Add predictions
    predictions_export['pred_proba'] = y_proba
    predictions_export['pred_class'] = (y_proba >= optimal_threshold).astype(int)

    # Select relevant columns for export
    export_columns = [
        'LINREFX', 'LINREFY',       # Coordinates
        'AYEAR', 'AMONTH', 'AHOUR', # Time
        'pred_proba', 'pred_class',  # Predictions
        'AccBike'                    # Actual accident (target variable)
    ]
    
    # Ensure only existing columns are exported
    export_columns = [col for col in export_columns if col in predictions_export.columns]
    predictions_df = predictions_export[export_columns]

    # Sort by probability (highest risk first)
    predictions_df = predictions_df.sort_values('pred_proba', ascending=False)

    # Save to CSV
    output_path = CSV_PATH.replace('.csv', '_PREDICTIONS.csv')
    predictions_df.to_csv(output_path, sep=';', decimal=',', index=False)
    print(f"{len(predictions_df)} predictions exported to: {output_path}")

    # Print statistics
    predicted_accidents = predictions_df['pred_class'].sum()
    actual_accidents = predictions_df['AccBike'].sum()
    print("\nPrediction Statistics:")
    print(f"  Predicted Accidents: {predicted_accidents}")
    print(f"  Actual Accidents:  {actual_accidents}")

    # Separate file with only predicted accidents
    predicted_accidents_only = predictions_df[predictions_df['pred_class'] == 1]
    output_path_accidents = CSV_PATH.replace('.csv', '_PREDICTED_ACCIDENTS_ONLY.csv')
    predicted_accidents_only.to_csv(output_path_accidents, sep=';', decimal=',', index=False)
    print(f"\n{len(predicted_accidents_only)} predicted accidents exported to: {output_path_accidents}")


# -------------------
# STEP 11: VISUALIZATIONS
# -------------------
print_header("STEP 11: Create Visualizations")

# 1. Feature Importance
# The feature names come from the 'final_feature_cols' list, not from the model itself.
importance_df = pd.DataFrame({
    'Feature': final_feature_cols,  # <-- CORRECTED LINE
    'Importance': model.get_feature_importance()
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance - Segment-based Model (CatBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_feature_importance_segmentbased.png', dpi=300, bbox_inches='tight')
plt.close()
print("'01_feature_importance_segmentbased.png' saved")

# Only run if test data is available
if len(X_test) > 0 and 'result' in locals() and 'auc_val' in locals():
    
    # 2. Precision-Recall Curve
    plt.figure(figsize=(12, 7))
    plt.plot(rec, prec, linewidth=2.5, label='Precision-Recall Curve', color='steelblue')
    plt.scatter(result['rec'], result['prec'], color='darkgreen', s=250, zorder=5,
                label=f"Max F1-Score = {result['f1']:.3f}", edgecolors='black', marker='o')
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curve (Segment-based Model)', fontsize=15, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.xlim([0, 1.02]); plt.ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig('02_precision_recall_curve_segmentbased.png', dpi=300)
    plt.close()
    print("'02_precision_recall_curve_segmentbased.png' saved")

    # 3. Confusion Matrix (Enhanced Version)
    cm = result['cm']
    
    fig, ax = plt.subplots(figsize=(8, 8)) # More height for the text box

    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Accident', 'Accident'],
                yticklabels=['No Accident', 'Accident'],
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    ax.set_title('Segment-based: Confusion Matrix (Optimal Threshold)', fontsize=14, fontweight='bold', y=1.08)
    
    metrics_text = (f"Precision: {result['prec']:.3f}  |  "
                    f"Recall: {result['rec']:.3f}  |  "
                    f"F1-Score: {result['f1']:.3f}  |  "
                    f"AUC: {auc_val:.3f}")

    ax.text(0.5, 1.02, metrics_text,
            transform=ax.transAxes,
            fontsize=12, 
            ha='center', 
            va='bottom', # Align vertical to the bottom
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))
    plt.tight_layout()
    plt.savefig('03_confusion_matrix_segmentbased.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("'03_confusion_matrix_segmentbased.png' saved")


    # 4. F1-Score over Thresholds
    plt.figure(figsize=(12, 7))
    plt.plot(thr, f1_scores, linewidth=2, color='purple', label='F1-Score')
    plt.axvline(optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score across Different Thresholds', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('04_f1_threshold_curve_segmentbased.png', dpi=300)
    plt.close()
    print("'04_f1_threshold_curve_segmentbased.png' saved")

else:
    print("Visualization plots skipped: 'result' or 'auc_val' not found.")

print("\n" + "="*70)
print("MODEL EVALUATION AND EXPORT COMPLETED")
print("="*70)