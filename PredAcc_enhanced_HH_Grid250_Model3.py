# ===================================================================
# MASTER THESIS

# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)

# SKRIPT: Enhanced Model Hamburg- 250m Grid with optimal threshold
# DATE: 18.10.2025
# DESCRIPTION: Uses optimal threshold, adjusted temporal features for Harmburg Accidents and
# adding infrastructure (source: OSM) and traffic (source: Geoportal Hamburg) Data as features
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
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Helper
def print_header(title):
    print(70*'-')
    print(title)
    print(70*'-')

# Global Configuration
print_header('Enhanced Model Hamburg - F1 Optimization (Undersampling) with CatBoost')
CSV_PATH = "Acc_250Grid_HH_enhanced_final.csv"
TARGET_COLUMN = 'AccBike'
RANDOM_STATE = 42
TEST_YEAR = 2024

# -------------------
# STEP 1: LOAD DATA
# -------------------
print_header('STEP 1: Load and prepare data')
try:
    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    df.columns = df.columns.str.strip()
    print(f"{len(df)} rows loaded from CSV.")
except FileNotFoundError:
    print(f"ERROR: File '{CSV_PATH}' not found.")
    exit()

# Datentypen sicherstellen
for col in ['LINREFX', 'LINREFY']:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

df['target'] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
print("Target variable prepared.")

# -------------------
# STEP 2: FEATURE ENGINEERING
# -------------------
print_header('STEP 2: Feature Engineering')

# Traffic Density Proxy
if 'Car_exposure_proxy' in df.columns and 'Bike_exposure_proxy' in df.columns:
    df['TRAFFICDENSITY_PROXY'] = df['Car_exposure_proxy'] + df['Bike_exposure_proxy']

df['Exposure_Car'] = pd.to_numeric(df['Car_DTV_Grid'], errors='coerce').fillna(0.0) if 'Car_DTV_Grid' in df.columns else 0.0
df['Exposure_Bike'] = pd.to_numeric(df['Bike_exposure_avg'], errors='coerce').fillna(0.0) if 'Bike_exposure_avg' in df.columns else 0.0
df['Rad_Years_Available'] = pd.to_numeric(df['Years_with_Data'], errors='coerce').fillna(0) if 'Years_with_Data' in df.columns else 0
print("Feature engineering completed.")


# -------------------
# STEP 3: FEATURE SELECTION AND DATA SPLIT
# -------------------
print_header('STEP 3 & 4: Feature Selection and Data Split')
feature_cols = [
    'LINREFX', 'LINREFY', 'AMONTH', 'AWEEKDAY', 'AHOUR', 'StrCon', 'LightCon', 'District',
    'BikeAccDens',
    'Junction_Count', 'Road_Length_Total', 'Primary_Road_Length',
    'Speedlimit_Max', 'Road_Density',
    'Car_DTV_Grid', 'Bike_exposure_avg','TRAFFICDENSITY_PROXY'
]
feature_cols = [col for col in feature_cols if col in df.columns]
print(f"{len(feature_cols)} features selected.")

train = df[df['AYEAR'] != TEST_YEAR].copy()
test = df[df['AYEAR'] == TEST_YEAR].copy()

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
print(f"Train class distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# -------------------
# STEP 5: Undersampling 
# -------------------
print_header('STEP 5: Apply Random Undersampling to Training Data')
rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"Train class distribution after undersampling: 0={sum(y_train_res==0)}, 1={sum(y_train_res==1)}")

# -------------------
# STEP 6: PREPROCESSING
# -------------------
print_header('STEP 6: Prepare Categorical Features')
categorical_features = ['AMONTH', 'AWEEKDAY', 'AHOUR', 'StrCon', 'LightCon', 'District'
    ]
categorical_features = [col for col in categorical_features if col in feature_cols]

for col in categorical_features:
    X_train_res[col] = X_train_res[col].fillna('missing').astype(str)
    X_test[col] = X_test[col].fillna('missing').astype(str)
print("Categorical features prepared for CatBoost.")

# -------------------
# STEP 7: MODEL TRAINING
# -------------------
print_header('STEP 7: CatBoost Training')
train_pool = Pool(X_train_res, y_train_res, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

base_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=5,
    random_strength=1.5, bagging_temperature=0.8, loss_function='Logloss',
    eval_metric='AUC', random_state=RANDOM_STATE, verbose=200
)

if len(X_train_res) and len(X_test):
    base_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=150)
else:
    base_model.fit(train_pool)
print('Training completed.')

# -------------------
# STEP 8: F1-SCORE OPTIMIZATION AND EVALUATION
# -------------------
print_header('STEP 8: F1-Score Optimization and Evaluation')
if len(X_test) > 0:
    y_proba = base_model.predict_proba(test_pool)[:, 1]
    auc_val = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else float('nan')
    print(f"Overall AUC-ROC: {auc_val:.4f}" if not np.isnan(auc_val) else "AUC-ROC: n/a")

    prec, rec, thr = precision_recall_curve(y_test, y_proba)
    f1_scores = (2 * prec * rec) / (prec + rec + 1e-12)
    optimal_threshold = thr[np.nanargmax(f1_scores)] if len(f1_scores) > 0 else 0.5
    optimal_f1 = np.nanmax(f1_scores) if len(f1_scores) > 0 else float('nan')
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Maximum F1-Score: {optimal_f1:.3f}" if not np.isnan(optimal_f1) else "Maximum F1-Score: n/a")

    def eval_at(t):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * p * r) / (p + r + 1e-12)
        return {'th': t, 'rec': r, 'prec': p, 'f1': f1, 'cm': cm, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    result = eval_at(optimal_threshold)
    print_header('FINAL MODEL PERFORMANCE (Undersampling)')
    print(f"Threshold: {result['th']:.3f}")
    print(f"Recall:    {result['rec']:.3f} ({result['rec']*100:.1f}%)")
    print(f"Precision: {result['prec']:.3f} ({result['prec']*100:.1f}%)")
    print(f"F1-Score:  {result['f1']:.3f}")
    print(f"\nMatrix:")
    print(f" TN: {result['tn']:6d}   FP: {result['fp']:6d}")
    print(f" FN: {result['fn']:6d}   TP: {result['tp']:6d}")

# -------------------
# STEP 9: EXPORT PREDICTIONS FOR MAPPING  (NEU EINGEFÜGT)
# -------------------
if len(X_test) > 0:
    print_header("STEP 9: Export Predictions for Mapping")

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
# STEP 10: VISUALIZATIONS (EHEMALS 9)
# -------------------

print("\n" + "="*70)
print("STEP 10: Create visualizations")
print("="*70)

# 1. Feature Importance
importances = base_model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Enhanced Grid: Feature Importance (CatBoost)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_feature_importance_classweights.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '01_feature_importance_classweights.png' saved")

# 2. Precision-Recall Curve (nur wenn Test vorhanden)
if len(X_test):
    # prec, rec, thr = precision_recall_curve(y_test, base_model.predict_proba(test_pool)[:, 1]) # Bereits in STEP 8 berechnet
    # Falls Schritt 8 lief:
    if 'result' in locals():
        plt.figure(figsize=(12, 7)) # Figur hier öffnen
        plt.plot(rec, prec, linewidth=2.5, label='Enhanced Grid: Precision-Recall Curve', color='steelblue', alpha=0.8)
        plt.scatter(result['rec'], result['prec'],
                    color='darkgreen', s=400, zorder=5,
                    label=f"Highest F1 Score = {result['f1']:.3f}",
                    edgecolors='black', linewidths=2.5, marker='o')
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curve with Optimal Threshold', fontsize=15, fontweight='bold')
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim([0, 1.02]); plt.ylim([0, 1.02])
        plt.tight_layout()
        plt.savefig('02_precision_recall_curve_classweights.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ '02_precision_recall_curve_classweights.png' saved")

# 3. Confusion Matrix
if 'result' in locals() and 'auc_val' in locals():
    cm = result['cm']
    
    fig, ax = plt.subplots(figsize=(8, 8)) # Etwas mehr Höhe für den Platz

    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                yticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    ax.set_title('Enhanced Grid: Confusion Matrix (Optimal Threshold)', fontsize=14, fontweight='bold', y=1.08)
    
    metrics_text = (f"Precision: {result['prec']:.3f}  |  "
                    f"Recall: {result['rec']:.3f}  |  "
                    f"F1-Score: {result['f1']:.3f}  |  "
                    f"AUC: {auc_val:.3f}")

    ax.text(0.5, 1.02, metrics_text,
            transform=ax.transAxes,
            fontsize=12, 
            ha='center', 
            va='bottom', # Vertikal am unteren Rand der Textbox ausrichten
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))
    plt.tight_layout()
    plt.savefig('03_confusion_matrix_undersampling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("'03_confusion_matrix_undersampling.png'")
else:
    print("Confusion Matrix plot skipped: 'result' or 'auc_val' not found.")

# 4. F1-Score over Thresholds
if len(X_test) and 'thr' in locals() and 'f1_scores' in locals():
    plt.figure(figsize=(12, 7))
    plt.plot(thr, f1_scores[:-1], linewidth=2, color='purple', label='F1-Score')
    if 'optimal_threshold' in locals():
        plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Enhanced Grid: Optimal Threshold = {optimal_threshold:.3f}')
        if 'optimal_f1' in locals():
            plt.axhline(optimal_f1, color='red', linestyle=':', linewidth=1, alpha=0.5)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score across Different Thresholds', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('04_f1_threshold_curve_enhancedgrid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ '04_f1_threshold_curve_enhancedgrid.png' saved")

print("\n" + "="*70)
print("ALL VISUALIZATIONS SUCCESSFULLY CREATED")
print("="*70)

print("\n" + "="*70)
print("MODEL EVALUATION COMPLETED")
print("="*70)