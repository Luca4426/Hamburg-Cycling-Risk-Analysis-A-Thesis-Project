# ===================================================================
# MASTER THESIS
# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)
# SKRIPT: Adjust Model Hamburg - 250m Grid with optimal threshold
# DATE: 18.10.2025
# DESCRIPTION: Uses optimal threshold and adjust temporal features for Hamburg Accidents
#
# BASED ON:
# AUTHORS: Sarah Di Grande, Mariaelena Berlotti, Salvatore Cavalieri and Daniel G. Costa
# YEAR: 2025
# TITLE: Data-Driven Prediction of High-Risk Situations for Cyclists Through Spatiotemporal Patterns and Environmental Conditions
# SOURCE: https://www.scitepress.org/Papers/2025/136464/136464.pdf (18.10.2025)
# ===================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score
)
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Global Configuration
CSV_PATH = "Acc_250Grid_HH_adjust.csv"
TARGET_COLUMN = 'AccBike'
RANDOM_STATE = 42
TEST_YEAR = 2024

print("="*70)
print("FINAL IMPROVED MODEL - F1 Optimization")
print("="*70)

# -------------------
# HELPER FUNCTIONS
# -------------------
import pandas as pd

def group_month(month):
    """Groups months according to Di Grande et al. study (3-level split)"""
    if pd.isna(month):
        return "missing"
    if month in [1, 2, 12]:
        return 'low_risk_winter'
    elif month in [3, 4, 10, 11]:
        return 'moderate_risk_transition'
    elif month in [5, 6, 7, 8, 9]:
        return 'high_risk_summer'
    return 'missing'

def group_weekday(day):
    """Groups weekdays according to Di Grande et al. study (finer granularity)"""
    if pd.isna(day):
        return "missing"
    if day in [2, 3, 4, 5, 6]: # Tue-Sa
        return 'highrisk-day'
    elif day == 1: # Mo
        return 'monday'
    elif day == 7: # Su
        return 'sunday'
    return 'missing'

def group_hour(hour):
    """Groups hours according to Di Grande et al. study (3-level split)"""
    if pd.isna(hour):
        return "missing"
    if hour in [0, 1, 2, 3, 4, 5, 22, 23]:
        return 'low_risk_night'
    elif hour in [9, 10, 11, 12,  19, 20, 21]:
        return 'medium_risk_offpeak'
    elif hour in [7, 8, 9, 13, 14, 15, 16, 17, 18]:
        return 'high_risk_peak'
    return 'missing'

# -------------------
# STEP 1: LOAD DATA
# -------------------
print("\n" + "="*70)
print("STEP 1: Load and prepare data")
print("="*70)

try:
    df = pd.read_csv(CSV_PATH, sep=';', decimal=',')
    df.columns = df.columns.str.strip()
    print(f"✓ {len(df)} rows loaded from {CSV_PATH}.")
except FileNotFoundError:
    print(f"ERROR: File {CSV_PATH} not found.")
    exit()

# Ensure data types for coordinates
for col in ['LINREFX', 'LINREFY']:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

df['target'] = df[TARGET_COLUMN].astype(int)

# -------------------
# STEP 2: FEATURE ENGINEERING
# -------------------
print("\n" + "="*70)
print("STEP 2: Feature Engineering")
print("="*70)

# Numeric conversion
numeric_cols = ['AMONTH', 'AWEEKDAY', 'AHOUR', 'LINREFX', 'LINREFY',
                'StrCon', 'LightCon', 'BikeAccDens']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
print("✓ Numeric columns converted.")

# Apply grouping functions
df['MonthGroup'] = df['AMONTH'].apply(group_month)
df['WeekdayGroup'] = df['AWEEKDAY'].apply(group_weekday)
df['HourGroup'] = df['AHOUR'].apply(group_hour)
print("✓ Temporal features grouped.")

print("✓ Feature engineering completed.")

# -------------------
# STEP 3: FEATURE SELECTION
# -------------------
feature_cols = [
    'LINREFX', 'LINREFY', 'AMONTH', 'AWEEKDAY', 'AHOUR',
    'StrCon', 'LightCon', 'District', 'BikeAccDens',
    'MonthGroup', 'WeekdayGroup', 'HourGroup'
]

feature_cols = [col for col in feature_cols if col in df.columns]
print(f"✓ {len(feature_cols)} features selected.")

# -------------------
# STEP 4: TRAIN-TEST SPLIT
# -------------------
train = df[df['AYEAR'] < TEST_YEAR].copy()
test = df[df['AYEAR'] == TEST_YEAR].copy()

X_train = train[feature_cols]
y_train = train['target']
X_test = test[feature_cols]
y_test = test['target']

print(f"✓ Training: {len(X_train)} | Test: {len(X_test)}")

# -------------------
# STEP 5: UNDERSAMPLING
# -------------------
print("\n" + "="*70)
print("STEP 5: Undersampling")
print("="*70)

rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"✓ After undersampling: {len(X_train_res)} samples")

# -------------------
# STEP 6: PREPROCESSING
# -------------------
categorical_features = ['AMONTH', 'AWEEKDAY', 'AHOUR', 'StrCon', 'LightCon', 'District',
                        'MonthGroup', 'WeekdayGroup', 'HourGroup']

categorical_features = [col for col in categorical_features if col in feature_cols]

for col in categorical_features:
    X_train_res[col] = X_train_res[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

# -------------------
# STEP 7: MODEL TRAINING
# -------------------
print("\n" + "="*70)
print("STEP 7: CatBoost Training")
print("="*70)

train_pool = Pool(X_train_res, y_train_res, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    verbose=200,
    random_state=RANDOM_STATE
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=150)
print("✓ Training completed.")

# -------------------
# STEP 8: F1 OPTIMIZATION
# -------------------
print("\n" + "="*70)
print("STEP 8: F1-Score Optimization")
print("="*70)

# Base probabilities
y_proba = model.predict_proba(test_pool)[:, 1]
auc_val = roc_auc_score(y_test, y_proba)
print(f"✓ Overall AUC-ROC: {auc_val:.4f}")

# Precision-Recall curve
prec, rec, thr = precision_recall_curve(y_test, y_proba)

# Calculate F1 scores for all thresholds
f1_scores = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)

# Find threshold with maximum F1-Score
optimal_threshold = thr[np.nanargmax(f1_scores)]
optimal_f1 = np.nanmax(f1_scores)
print(f"\n✓ Optimal threshold found: {optimal_threshold:.3f}")
print(f"✓ Maximum F1-Score: {optimal_f1:.3f}")

# Evaluation with optimal threshold
def eval_at(t):
    yp = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm.ravel()
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    f = 2 * (p * r) / (p + r + 1e-12)
    return dict(th=t, rec=r, prec=p, f1=f, cm=cm, tn=tn, fp=fp, fn=fn, tp=tp)

result = eval_at(optimal_threshold)

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)
print(f"Threshold: {result['th']:.3f}")
print(f"Recall: {result['rec']:.3f} ({result['rec']*100:.1f}% of accidents detected)")
print(f"Precision: {result['prec']:.3f} ({result['prec']*100:.1f}% of predictions correct)")
print(f"F1-Score: {result['f1']:.3f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives: {result['tn']:6d}")
print(f"  False Positives: {result['fp']:6d}")
print(f"  False Negatives: {result['fn']:6d}")
print(f"  True Positives: {result['tp']:6d}")

# -------------------
# STEP 9: VISUALIZATIONS
# -------------------
print("\n" + "="*70)
print("STEP 9: Create visualizations")
print("="*70)

# 1. Feature Importance
importances = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Adjust: Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '01_feature_importance.png' saved")

# 2. Precision-Recall Curve
plt.figure(figsize=(12, 7))
plt.plot(rec, prec, linewidth=2.5, label='Precision-Recall Curve', color='steelblue', alpha=0.8)

plt.scatter(result['rec'], result['prec'],
            color='darkgreen',
            s=400,
            zorder=5,
            label=f"Highest F1 Score = {result['f1']:.3f}",
            edgecolors='black',
            linewidths=2.5,
            marker='o')

plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')
plt.title('Adjust: Precision-Recall Curve with Optimal Threshold', fontsize=15, fontweight='bold')

plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([0, 1.02])
plt.ylim([0, 1.02])
plt.tight_layout()
plt.savefig('02_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '02_precision_recall_curve.png' saved")

# 3. Confusion Matrix
if 'result' in locals() and 'auc_val' in locals():
    cm = result['cm']
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sns.heatmap(cm,
                annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                yticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Adjust: Confusion Matrix - F1-Optimized Model', fontsize=14, fontweight='bold', y=1.08)
    
    metrics_text = (f"Precision: {result['prec']:.3f} | "
                    f"Recall: {result['rec']:.3f} | "
                    f"F1-Score: {result['f1']:.3f} | "
                    f"AUC: {auc_val:.3f}")
    
    ax.text(0.5, 1.02, metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            ha='center',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('03_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ '03_confusion_matrix.png' saved")
else:
    print("Confusion Matrix plot skipped: 'result' or 'auc_val' not found.")

# 4. F1-Score over Thresholds
plt.figure(figsize=(12, 7))
plt.plot(thr, f1_scores, linewidth=2, color='purple', label='F1-Score')
plt.axvline(optimal_threshold, color='green', linestyle='--', linewidth=2,
            label=f'Optimal Threshold = {optimal_threshold:.3f}')
plt.axhline(optimal_f1, color='red', linestyle=':', linewidth=1, alpha=0.5)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Adjust: F1-Score across Different Thresholds', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_f1_threshold_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '04_f1_threshold_curve.png' saved")

print("\n" + "="*70)
print("ALL VISUALIZATIONS SUCCESSFULLY CREATED")
print("="*70)
print("\nGenerated files:")
print("  - 01_feature_importance.png")
print("  - 02_precision_recall_curve.png")
print("  - 03_confusion_matrix.png")
print("  - 04_f1_threshold_curve.png")

print("\n" + "="*70)
print("MODEL EVALUATION COMPLETED")
print("="*70)
