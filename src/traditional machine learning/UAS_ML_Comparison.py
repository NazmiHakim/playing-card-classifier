import cv2
import numpy as np
import os
import shutil
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# Seed untuk konsistensi
SEED = 42
np.random.seed(SEED)

# Deteksi Lingkungan Kerja
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("--- MENJALANKAN DI GOOGLE COLAB ---")
    from google.colab import drive
    drive.mount('/content/drive')

    possible_paths = [
        '/content/drive/MyDrive/CARD DATASET 2',
        '/content/drive/MyDrive/CARD_DATASET_SMALL',
        '/content/drive/MyDrive/CARD DATASET',
        '/content/drive/My Drive/CARD DATASET 2',
    ]
    DATA_DIR = possible_paths[0]
    for path in possible_paths:
        if os.path.exists(path):
            DATA_DIR = path
            print(f"Dataset ditemukan: {DATA_DIR}")
            break

    # Salin ke SSD Colab untuk kecepatan I/O
    LOCAL_DIR = '/content/dataset_kartu_ml'
    if not os.path.exists(LOCAL_DIR):
        print("Menyalin dataset ke SSD lokal...")
        try:
            shutil.copytree(DATA_DIR, LOCAL_DIR)
            LOCAL_DIR = DATA_DIR
        except Exception as e:
            print(f"Gagal salin: {e}")
            LOCAL_DIR = DATA_DIR
    DATA_DIR = LOCAL_DIR
else:
    for candidate in [
        r'd:\Tugas Semester 6\Machine Learning\Tugas Individu\CARD DATASET 2',
        './CARD DATASET 2',
        r'd:\Tugas Semester 6\Machine Learning\Tugas Individu\CARD_DATASET_SMALL',
        './CARD_DATASET_SMALL',
    ]:
        if os.path.exists(candidate):
            DATA_DIR = candidate
            break

CLASS_NAMES = ['CLUB', 'DIAMOND', 'HEART', 'SPADE']
IMG_SIZE    = (224, 224)
TEST_SIZE   = 0.2

# Ekstraksi Fitur HOG + Warna
def extract_features(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, img_size)

    # Rata-rata warna BGR
    color_features = np.mean(img_resized, axis=(0, 1))

    # HOG (Grayscale)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    return np.hstack((color_features, hog_features))

print("\n--- 1. MEMUAT DATASET & EKSTRAKSI FITUR HOG ---")
X, y = [], []
class_counts = {}

for label_idx, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_path):
        print(f"[WARNING] Folder tidak ditemukan: {class_path}")
        continue

    image_files = sorted([
        f for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    class_counts[class_name] = len(image_files)
    print(f"  Memproses {class_name}: {len(image_files)} gambar", end="")

    for img_file in image_files:
        feat = extract_features(os.path.join(class_path, img_file), IMG_SIZE)
        if feat is not None:
            X.append(feat)
            y.append(label_idx)
    print(f" → {len([v for v in y if v == label_idx])} fitur diekstrak")

X = np.array(X)
y = np.array(y)

# Visualisasi Distribusi Dataset
plt.figure(figsize=(7, 4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', alpha=0.85)
plt.title('Distribusi Dataset CARD DATASET 2 (Total: 800 Gambar)', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Suit', fontsize=10)
plt.ylabel('Jumlah Gambar', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f'{int(yval)}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('distribusi_kelas_ml.png', dpi=300)
plt.show()

# Pembagian Dataset (80/20) & Standardisasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Definisi 5 Model Machine Learning Tradisional
models_config = [
    {
        "name": "SVM — Kernel Linear",
        "short": "SVM\nLinear",
        "model": SVC(kernel='linear', C=1.0, max_iter=2000, random_state=SEED, probability=True),
        "color": "#4e79a7"
    },
    {
        "name": "SVM — Kernel RBF (C=1)",
        "short": "SVM\nRBF C=1",
        "model": SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=2000, random_state=SEED, probability=True),
        "color": "#59a14f"
    },
    {
        "name": "SVM — Kernel RBF (C=10)",
        "short": "SVM\nRBF C=10",
        "model": SVC(kernel='rbf', C=10.0, gamma=0.01, max_iter=2000, random_state=SEED, probability=True),
        "color": "#f28e2b"
    },
    {
        "name": "K-Nearest Neighbors (k=7)",
        "short": "KNN\nk=7",
        "model": KNeighborsClassifier(n_neighbors=7, metric='minkowski', n_jobs=-1),
        "color": "#e15759"
    },
    {
        "name": "Random Forest (n=200)",
        "short": "Random\nForest",
        "model": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "color": "#b07aa1"
    },
]

print("\n--- 2. TRAINING & EVALUASI 5 MODEL ML ---")
results = []

for cfg in models_config:
    print(f"\n  ▶ {cfg['name']}...")
    clf = cfg["model"]
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )

    results.append({
        "name":      cfg["name"],
        "short":     cfg["short"],
        "color":     cfg["color"],
        "model":     clf,
        "y_pred":    y_pred,
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    })
    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

# Ringkasan Evaluasi
print("\n--- 3. RINGKASAN PERBANDINGAN 5 MODEL ML ---")
summary_rows = []
for r in results:
    summary_rows.append({
        "Model": r["name"],
        "Accuracy": f"{r['accuracy']*100:.2f}%",
        "Precision": f"{r['precision']:.4f}",
        "Recall": f"{r['recall']:.4f}",
        "F1-Score": f"{r['f1']:.4f}",
    })
df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

best_ml = max(results, key=lambda r: r['accuracy'])
print(f"\n[INFO] Model ML Terbaik: {best_ml['name']} ({best_ml['accuracy']*100:.2f}%)")

print("\n--- 4. CLASSIFICATION REPORT DETAIL ---")
for r in results:
    print(f"\n{'='*55}\n  {r['name']}\n{'='*55}")
    print(classification_report(y_test, r['y_pred'], target_names=CLASS_NAMES, zero_division=0))

# Visualisasi Confusion Matrix (5 Model)
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle("Confusion Matrix — 5 Model Machine Learning Tradisional\n(Dataset: CARD DATASET 2, 800 Gambar, Split 80/20)", fontsize=13, fontweight='bold', y=1.02)

cmaps = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
for i, (r, cmap) in enumerate(zip(results, cmaps)):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[i], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, annot_kws={"size": 11, "weight": "bold"}, cbar=False)
    axes[i].set_title(f"{r['name']}\nAcc: {r['accuracy']*100:.1f}%", fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Predicted', fontsize=8)
    axes[i].set_ylabel('True', fontsize=8)
    axes[i].tick_params(axis='both', labelsize=7)

plt.tight_layout()
plt.savefig('confusion_matrix_all_ml.png', dpi=200, bbox_inches='tight')
plt.show()

# Kurva ROC untuk Model ML Terbaik
if hasattr(best_ml['model'], 'predict_proba'):
    y_true_bin = label_binarize(y_test, classes=range(len(CLASS_NAMES)))
    y_prob = best_ml['model'].predict_proba(X_test_sc)
    fpr_all, tpr_all, auc_all = {}, {}, {}
    for i in range(len(CLASS_NAMES)):
        fpr_all[i], tpr_all[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_all[i] = auc(fpr_all[i], tpr_all[i])

    plt.figure(figsize=(7, 5))
    roc_colors = ['blue', 'orange', 'green', 'red']
    for i, color in zip(range(len(CLASS_NAMES)), roc_colors):
        plt.plot(fpr_all[i], tpr_all[i], color=color, lw=2, label=f'{CLASS_NAMES[i]} (AUC = {auc_all[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(f'ROC Curve — {best_ml["name"]}', fontsize=11, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('roc_curve_best_ml.png', dpi=300)
    plt.show()

# Perbandingan Akhir (5 ML + 2 DL)
DL_RESULTS = [
    {
        "name": "MobileNetV2\n(Baseline)",
        "short": "MobileNetV2\nBaseline",
        "color": "#76b7b2",
        "accuracy":  0.7875,
        "f1":        0.8300,
    },
    {
        "name": "MobileNetV2\n+ TTA",
        "short": "MobileNetV2\n+ TTA",
        "color": "#ff9da7",
        "accuracy":  0.8250,
        "f1":        0.8300,
    },
]

all_models = results + DL_RESULTS
model_labels   = [r["short"] for r in all_models]
acc_values     = [r["accuracy"] * 100 for r in all_models]
f1_values      = [r["f1"] * 100 for r in all_models]
bar_colors     = [r["color"] for r in all_models]

x = np.arange(len(all_models))
separator = len(results) - 0.5

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Perbandingan Performa 7 Model: 5 ML Tradisional vs 2 Deep Learning\n(Dataset: CARD DATASET 2, 800 Gambar, Split 80/20, SEED=42/123)", fontsize=13, fontweight='bold')

for ax, values, metric_label, ylim_min in [
    (axes[0], acc_values, "Accuracy (%)", 50),
    (axes[1], f1_values,  "F1-Score Macro (%)", 50)
]:
    bars = ax.bar(x, values, color=bar_colors, edgecolor='black', alpha=0.88, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8.5)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_ylim(ylim_min, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.axvline(x=separator, color='gray', linestyle='--', linewidth=1.5)
    ax.text(separator - 1.3, ylim_min + 2, '← ML Tradisional', fontsize=8, color='gray', fontstyle='italic')
    ax.text(separator + 0.2, ylim_min + 2, 'Deep Learning →', fontsize=8, color='gray', fontstyle='italic')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

axes[0].set_title('Accuracy', fontsize=11, fontweight='bold')
axes[1].set_title('F1-Score (Macro Average)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('perbandingan_7_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Analisis Kesenjangan Performa
print("\n" + "=" * 60)
print("ANALISIS KESENJANGAN: ML vs DL")
print("=" * 60)
best_ml_acc  = best_ml['accuracy'] * 100
best_dl_acc  = max(r['accuracy'] for r in DL_RESULTS) * 100
gap          = best_dl_acc - best_ml_acc

print(f"  Akurasi ML terbaik : {best_ml_acc:.2f}% ({best_ml['name']})")
print(f"  Akurasi DL terbaik : {best_dl_acc:.2f}% (MobileNetV2 + TTA)")
print(f"  Selisih (gap)       : +{gap:.2f}% (DL lebih unggul)")

print("""
  Alasan MobileNetV2 mengungguli HOG + SVM:
  1. Fitur Hierarkis Otomatis: CNN mengekstraksi pola secara otomatis dibanding HOG (gradien manual).
  2. Ketahanan Latar Belakang: Transfer learning berfokus pada suit kartu remi.
  3. Pengurangan Overfitting: Data augmentation & Dropout menstabilkan model pada dataset terbatas.
""")
print("=" * 60)
