# ===================================================================
# MASTER THESIS
#
# AUTHOR: Luca Alexander Davids
# UNIVERSITY: HafenCity Universität Hamburg (HCU)
#
# SKRIPT: Baseline Model Hamburg- 500m Grid with F1-OPTIMIZED threshold
# DATE: 18.10.2025
# DESCRIPTION: Uses F1-OPTIMIZED threshold and exact same features as study methodology
#              (Temporal groupings now 1:1 match Di Grande et al.)
#
# BASED ON: 
#   AUTHORS: Sarah Di Grande, Mariaelena Berlotti, Salvatore Cavalieri and Daniel G. Costa
#   YEAR: 2025
#   TITLE: Data-Driven Prediction of High-Risk Situations for Cyclists Through Spatiotemporal Patterns and 
#   Environmental Conditions 
#   SOURCE: https://www.scitepress.org/Papers/2025/136464/136464.pdf (18.10.2025)
# ===================================================================
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, classification_report
)
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Global Configuration
CSV_PATH = "Acc_HH_Grid500.csv"
TARGET_COLUMN = 'AccBike'
RANDOM_STATE = 42
TEST_YEAR = 2024


print("="*70)
print("FINAL BASELINE MODEL - F1-Optimized Threshold (Di Grande Groupings)")
print("="*70)

# -------------------
# HELPER FUNCTIONS
# -------------------
import pandas as pd

def group_month_studie(month):
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

def group_weekday_hamburg(day):
    """Groups weekdays according to Di Grande et al. study (finer granularity)"""
    if pd.isna(day):
        return "missing"
    if day in [1, 2, 3, 4, 5]: # Mo-Fr
        return 'weekday'
    elif day == 6: # Sa
        return 'saturday'
    elif day == 7: # So
        return 'sunday'
    return 'missing'

def group_hour_hamburg(hour):
    """Groups hours according to Di Grande et al. study (3-level split)"""
    if pd.isna(hour): # <--- KORREKTUR
        return "missing"
    if hour in [0, 1, 2, 3, 4, 5, 22, 23]:
        return 'low_risk_night'
    elif hour in [6, 8, 9, 10, 11, 19, 20, 21]:
        return 'medium_risk_offpeak'
    elif hour in [7, 12, 13, 14, 15, 16, 17, 18]:
        return 'high_risk_peak'
    return 'missing'
# -------------------
# STEP 1: LOAD DATA
# -------------------
print("\n" + "="*70)
print("STEP 1: Load and prepare data")
print("="*70)

try:
    df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8', decimal=',')
    df.columns = df.columns.str.strip()
    print(f"✓ {len(df)} rows loaded from '{CSV_PATH}'.")
except FileNotFoundError:
    print(f"ERROR: File '{CSV_PATH}' not found.")
    exit()

df['target'] = df[TARGET_COLUMN].astype(int)
print(f"✓ Target variable '{TARGET_COLUMN}' set. Positive class: {(df['target']==1).mean():.2%}")

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
df['MonthGroup'] = df['AMONTH'].apply(group_month_studie)
df['WeekdayGroup'] = df['AWEEKDAY'].apply(group_weekday_hamburg)
df['HourGroup'] = df['AHOUR'].apply(group_hour_hamburg)
print("✓ Temporal features grouped (matching Di Grande et al. ).")

# Define feature list
feature_cols = ['AMONTH', 'AWEEKDAY', 'AHOUR', 'LINREFX', 'LINREFY', 'StrCon', 'LightCon', 'BikeAccDens',
                'MonthGroup', 'WeekdayGroup', 'HourGroup', 'District']
X = df[feature_cols]
y = df['target']

# -------------------
# STEP 3: TRAIN-TEST SPLIT
# -------------------
print("\n" + "="*70)
print("STEP 3: Train-Test Split and Balancing")
print("="*70)

X_train = X[df['AYEAR'] < TEST_YEAR]
y_train = y[df['AYEAR'] < TEST_YEAR]
X_test = X[df['AYEAR'] == TEST_YEAR]
y_test = y[df['AYEAR'] == TEST_YEAR]

print(f"✓ Train/Test split. Train: {len(X_train)}, Test: {len(X_test)}")

rus = RandomUnderSampler(random_state=RANDOM_STATE)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"✓ Undersampling completed. New training size: {len(X_train_res)}")

# -------------------
# STEP 4: PREPROCESSING
# -------------------
print("\n" + "="*70)
print("STEP 4: NaN handling for categorical features")
print("="*70)

categorical_features = ['MonthGroup', 'WeekdayGroup', 'HourGroup', 'District',
                       'LightCon', 'StrCon']

for col in categorical_features:
    X_train_res[col] = X_train_res[col].fillna("missing").astype(str)
    X_test[col] = X_test[col].fillna("missing").astype(str)

print("✓ Missing values in categorical features converted to 'missing'.")

# -------------------
# STEP 5: MODEL TRAINING
# -------------------
print("\n" + "="*70)
print("STEP 5: Model Training")
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

model.fit(train_pool)
print("✓ Training completed.")


# -------------------
# STEP 5.5: F1-THRESHOLD OPTIMIZATION (on Training Set) <--- MODIFIKATION: NEUER STEP
# -------------------
print("\n" + "="*70)
print("STEP 5.5: F1-Threshold Optimization (on Resampled Training Set)")
print("="*70)

# Hole Wahrscheinlichkeiten für die *resampelten* Trainingsdaten
y_proba_train_res = model.predict_proba(train_pool)[:, 1]

# Berechne P-R-Kurve auf Trainingsdaten
prec_train, rec_train, thr_train = precision_recall_curve(y_train_res, y_proba_train_res)

# Berechne F1-Scores für alle Thresholds
f1_scores_train = 2 * (prec_train[:-1] * rec_train[:-1]) / (prec_train[:-1] + rec_train[:-1] + 1e-12)
f1_scores_train = np.nan_to_num(f1_scores_train) # NaNs entfernen

# Finde den besten Threshold
optimal_f1_train = np.max(f1_scores_train)
best_threshold = thr_train[np.argmax(f1_scores_train)]

print(f"✓ Optimal F1-Score on (resampled) Train Set: {optimal_f1_train:.4f}")
print(f"✓ Best Threshold found: {best_threshold:.4f}")


# -------------------
# STEP 6: EVALUATION WITH F1-OPTIMIZED THRESHOLD <--- MODIFIKATION: Titel geändert
# -------------------
print("\n" + "="*70)
print(f"STEP 6: Evaluation with F1-OPTIMIZED Threshold: {best_threshold:.4f}") # <--- MODIFIKATION
print("="*70)

# Base probabilities (auf Test-Set)
y_proba = model.predict_proba(test_pool)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"✓ Overall AUC-ROC: {auc:.4f}")

# Precision-Recall curve (auf Test-Set)
prec, rec, thr = precision_recall_curve(y_test, y_proba)

# Calculate F1 scores for all thresholds (auf Test-Set, zur Referenz)
f1_scores_test = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
optimal_f1_test = np.nanmax(f1_scores_test)

print(f"\n✓ USING F1-OPTIMIZED THRESHOLD: {best_threshold:.4f}") # <--- MODIFIKATION
print(f"  (Note: Maximum possible F1-Score on *Test Set* would be: {optimal_f1_test:.3f})")

# Evaluation with OPTIMIZED threshold
def eval_at(t):
    yp = (y_proba >= t).astype(int)
    cm = confusion_matrix(y_test, yp)
    tn, fp, fn, tp = cm.ravel()
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    f = 2 * (p * r) / (p + r + 1e-12)
    return dict(th=t, rec=r, prec=p, f1=f, cm=cm, tn=tn, fp=fp, fn=fn, tp=tp)

result = eval_at(best_threshold) # <--- MODIFIKATION

print("\n" + "="*70)
print(f"FINAL MODEL PERFORMANCE (Threshold = {best_threshold:.4f})") # <--- MODIFIKATION
print("="*70)
# (Dieser Teil wurde in Ihrem Skript gefehlt, ist aber nützlich für das Kopieren von Ergebnissen)
print(classification_report(y_test, (y_proba >= best_threshold).astype(int), target_names=['Class 0 (No Bike)', 'Class 1 (Bike)']))
# (Originaler Print-Block)
print(f"Threshold: {result['th']:.3f}")
print(f"Recall: {result['rec']:.3f} ({result['rec']*100:.1f}% of accidents detected)")
print(f"Precision: {result['prec']:.3f} ({result['prec']*100:.1f}% of predictions correct)")
print(f"F1-Score: {result['f1']:.3f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {result['tn']:6d}")
print(f"  False Positives: {result['fp']:6d}")
print(f"  False Negatives: {result['fn']:6d}")
print(f"  True Positives:  {result['tp']:6d}")

# -------------------
# STEP 7: VISUALIZATIONS
# -------------------
print("\n" + "="*70)
print("STEP 7: Create visualizations")
print("="*70)

# 1. Feature Importance Base
importances = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Base: Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_feature_importance_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '01_feature_importance_baseline.png' saved")

# 2. Precision-Recall Curve
plt.figure(figsize=(12, 7))
plt.plot(rec, prec, linewidth=2.5, label='Base: Precision-Recall Curve', color='steelblue', alpha=0.8)
plt.scatter(result['rec'], result['prec'],
            color='darkgreen',
            s=400,
            zorder=5,
            label=f"F1-Optimized Threshold: F1 = {result['f1']:.3f}", # <--- MODIFIKATION
            edgecolors='black',
            linewidths=2.5,
            marker='o')
plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')
plt.title('Base: Precision-Recall Curve with F1-Optimized Threshold', fontsize=15, fontweight='bold') # <--- MODIFIKATION
plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([0, 1.02])
plt.ylim([0, 1.02])
plt.tight_layout()
plt.savefig('02_precision_recall_curve_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '02_precision_recall_curve_baseline.png' saved")

# 3. Confusion Matrix
# Sicherheitsabfrage, um sicherzustellen, dass die benötigten Daten existieren
if 'result' in locals() and 'auc' in locals(): # <--- MODIFIKATION: 'auc_val' zu 'auc' korrigiert (Bugfix)
    # Hole die Konfusionsmatrix aus dem 'result'-Dictionary
    cm = result['cm']
    
    # Erstelle eine Figur und ein Achsen-Objekt für bessere Kontrolle
    fig, ax = plt.subplots(figsize=(8, 8)) # Etwas mehr Höhe für den Platz

    # Erstelle die Heatmap auf dem Achsen-Objekt 'ax'
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                yticklabels=['No Bicycle Accident', 'Bicycle Accident'],
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)

    # Setze die Achsenbeschriftungen
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Setze den Titel und verschiebe ihn leicht nach oben (y=1.08), um Platz zu schaffen
    ax.set_title('Base: Confusion Matrix - F1-Optimized Model', fontsize=14, fontweight='bold', y=1.08) # <--- MODIFIKATION
    
    # Erstelle einen einzelnen String mit den Metriken, formatiert für die horizontale Anzeige
    metrics_text = (f"Precision: {result['prec']:.3f}  |  "
                    f"Recall: {result['rec']:.3f}  |  "
                    f"F1-Score: {result['f1']:.3f}  |  "
                    f"AUC: {auc:.3f}") # <--- MODIFIKATION: 'auc_val' zu 'auc' korrigiert (Bugfix)

    # Platziere die Textbox unter dem Titel mithilfe relativer Koordinaten
    ax.text(0.5, 1.02, metrics_text,
            transform=ax.transAxes,
            fontsize=12, 
            ha='center', 
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))

    # Passe das Layout an, um ein Abschneiden zu verhindern
    plt.tight_layout()

    # Speichere die Figur
    plt.savefig('03_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Schließe die Figur, um Speicher freizugeben
    
    # Gib eine Erfolgsmeldung aus
    print("✓ '03_confusion_matrix.png' saved")
else:
    # Gib eine Warnung aus, falls die Daten für den Plot fehlen
    print("Confusion Matrix plot skipped: 'result' or 'auc' not found.") # <--- MODIFIKATION

# 4. F1-Score over Thresholds (showing optimized threshold)
plt.figure(figsize=(12, 7))
plt.plot(thr, f1_scores_test, linewidth=2, color='purple', label='F1-Score (Test Set)')
plt.axvline(best_threshold, color='darkgreen', linestyle='--', linewidth=2, # <--- MODIFIKATION
            label=f'Optimized Threshold = {best_threshold:.3f} (from Training Set)') # <--- MODIFIKATION
plt.axhline(result['f1'], color='darkgreen', linestyle=':', linewidth=1, alpha=0.5)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Base: F1-Score across Different Thresholds (Optimized)', fontsize=14, fontweight='bold') # <--- MODIFIKATION
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_f1_threshold_curve_baseline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ '04_f1_threshold_curve_baseline.png' saved")

print("\n" + "="*70)
print("ALL VISUALIZATIONS SUCCESSFULLY CREATED")
print("="*70)
print("\nGenerated files:")
print("  - 01_feature_importance_baseline.png")
print("  - 02_precision_recall_curve_baseline.png")
print("  - 03_confusion_matrix.png")
print("  - 04_f1_threshold_curve_baseline.png")

print("\n" + "="*70)
print("SCRIPT SUCCESSFULLY COMPLETED")
print("="*70)