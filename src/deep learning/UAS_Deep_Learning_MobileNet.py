import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomBrightness, RandomContrast
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random

# Seed untuk konsistensi
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Deteksi Lingkungan Kerja
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    print("\n--- MENJALANKAN DI GOOGLE COLAB ---")
    from google.colab import drive
    drive.mount('/content/drive')
    
    DATA_DIR = '/content/drive/MyDrive/CARD DATASET 2'
    possible_paths = [
        '/content/drive/MyDrive/CARD DATASET 2',
        '/content/drive/MyDrive/CARD_DATASET_SMALL',
        '/content/drive/MyDrive/CARD DATASET',
        '/content/drive/MyDrive/CARD DATASET-20260613T040036Z-3-001/CARD DATASET',
        '/content/drive/My Drive/CARD DATASET 2',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            DATA_DIR = path
            print(f"Dataset ditemukan: {DATA_DIR}")
            break
            
    # Salin ke SSD lokal Colab untuk mempercepat training
    import shutil
    LOCAL_DIR = '/content/dataset_kartu'
    if not os.path.exists(LOCAL_DIR):
        print(f"Menyalin dataset ke SSD lokal Colab...")
        try:
            shutil.copytree(DATA_DIR, LOCAL_DIR)
        except Exception as e:
            print(f"Gagal salin: {e}")
            LOCAL_DIR = DATA_DIR
    DATA_DIR = LOCAL_DIR
else:
    DATA_DIR = r'd:\Tugas Semester 6\Machine Learning\Tugas Individu\CARD DATASET 2'
    for candidate in [
        r'd:\Tugas Semester 6\Machine Learning\Tugas Individu\CARD DATASET 2',
        './CARD DATASET 2',
        r'd:\Tugas Semester 6\Machine Learning\Tugas Individu\CARD_DATASET_SMALL',
        './CARD_DATASET_SMALL',
    ]:
        if os.path.exists(candidate):
            DATA_DIR = candidate
            break

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("\n--- 1. MEMUAT DATASET UNTUK DEEP LEARNING ---")
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataset = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print(f"Kelas yang dideteksi: {class_names}")

# Optimasi Pipeline Data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(buffer_size=640, seed=SEED, reshuffle_each_iteration=True).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Visualisasi Distribusi Dataset
class_counts = {}
for c in class_names:
    class_path = os.path.join(DATA_DIR, c)
    class_counts[c] = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(class_path) else 200

plt.figure(figsize=(7, 4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', alpha=0.8)
plt.title('Distribusi Jumlah Gambar per Kelas Kartu Remi', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Suit', fontsize=10)
plt.ylabel('Jumlah File Gambar', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('distribusi_kelas.png', dpi=300)
plt.show()

# Layer Augmentasi Data (On-the-fly)
data_augmentation = Sequential([
  RandomFlip('horizontal'),
  RandomRotation(0.1),
  RandomZoom(0.15),
  RandomTranslation(0.1, 0.1),
  RandomBrightness(0.15),
  RandomContrast(0.1)
], name="Data_Augmentation_Layer")

# Visualisasi Contoh Hasil Augmentasi
plt.figure(figsize=(8, 8))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.suptitle("Visualisasi Data Augmentation", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualisasi_augmentasi.png', dpi=300)
plt.show()

# Membangun Model MobileNetV2 (Fase 1: Feature Extraction)
print("\n--- 2. MEMBANGUN ARSITEKTUR MODEL ---")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Training Fase 1
EPOCHS = 30
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model_kartu.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

print(f"\n--- 3. MEMULAI TRAINING (Fase 1: Feature Extraction) ---")
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks)

# Training Fase 2: Fine-Tuning
print("\n--- 3b. MEMULAI FASE 2: FINE-TUNING ---")
FINE_TUNE_AT = 100
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

FINE_TUNE_EPOCHS = 20
INITIAL_EPOCH = len(history.history['accuracy'])
callbacks_finetune = [
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model_kartu.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

history_fine = model.fit(
    train_dataset,
    epochs=INITIAL_EPOCH + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCH,
    validation_data=val_dataset,
    callbacks=callbacks_finetune
)

# Menggabungkan History Pelatihan
acc_total     = history.history['accuracy']     + history_fine.history['accuracy']
val_acc_total = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss_total    = history.history['loss']         + history_fine.history['loss']
val_loss_total = history.history['val_loss']    + history_fine.history['val_loss']

# Visualisasi Kurva Pelatihan
epochs_range = range(len(acc_total))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_total, label='Training Accuracy', marker='o', color='#1f77b4')
plt.plot(epochs_range, val_acc_total, label='Validation Accuracy', marker='o', color='#2ca02c')
plt.axvline(x=INITIAL_EPOCH - 1, color='gray', linestyle='--', label=f'Fine-Tune Start (Epoch {INITIAL_EPOCH})')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_total, label='Training Loss', marker='o', color='#d62728')
plt.plot(epochs_range, val_loss_total, label='Validation Loss', marker='o', color='#ff7f0e')
plt.axvline(x=INITIAL_EPOCH - 1, color='gray', linestyle='--', label=f'Fine-Tune Start (Epoch {INITIAL_EPOCH})')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.suptitle("Kurva Pelatihan Model - CNN (MobileNetV2)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kurva_pelatihan.png', dpi=300)
plt.show()

# Evaluasi Model
print("\n--- 5. MEMBUAT CONFUSION MATRIX DEEP LEARNING ---")
y_true, y_pred_probs = [], []
for images, labels in val_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs.extend(preds)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report (MobileNetV2 Baseline):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12, "weight": "bold"}, cbar=True)
plt.title("Confusion Matrix - CNN (MobileNetV2)", fontsize=12, fontweight='bold')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=300)
plt.show()

# Visualisasi Bar Chart Metrik per Kelas
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(class_names)))
x = np.arange(len(class_names))
width = 0.25
plt.figure(figsize=(8, 5))
plt.bar(x - width, precision, width, label='Precision', color='#1f77b4', edgecolor='black', alpha=0.8)
plt.bar(x, recall, width, label='Recall', color='#2ca02c', edgecolor='black', alpha=0.8)
plt.bar(x + width, f1, width, label='F1-Score', color='#ff7f0e', edgecolor='black', alpha=0.8)
plt.title('Metrik Evaluasi per Kelas Kartu Remi', fontsize=12, fontweight='bold')
plt.xticks(x, class_names)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for i in range(len(class_names)):
    plt.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
    plt.text(i, recall[i] + 0.02, f'{recall[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
    plt.text(i + width, f1[i] + 0.02, f'{f1[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('metrik_evaluasi_cnn.png', dpi=300)
plt.show()

# Visualisasi ROC Curve
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red']
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Multi-Class', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('kurva_roc_cnn.png', dpi=300)
plt.show()

# Visualisasi Sampel Hasil Prediksi
plt.figure(figsize=(10, 10))
for images, labels in val_dataset.take(1):
    preds = model.predict(images, verbose=0)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_label = class_names[labels[i]]
        pred_label = class_names[np.argmax(preds[i])]
        confidence = np.max(preds[i]) * 100
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"Asli: {true_label}\nTebak: {pred_label} ({confidence:.1f}%)", color=color, fontsize=10)
        plt.axis("off")
plt.suptitle("Sampel Hasil Prediksi Model MobileNetV2", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hasil_prediksi_sampel.png', dpi=300)
plt.show()

# Simpan Model Final
model.save('model_kartu_mobilenet.keras')
print("\n[SUKSES] Model berhasil disimpan.")
