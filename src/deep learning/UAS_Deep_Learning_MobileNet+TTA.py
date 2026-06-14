import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     RandomFlip, RandomRotation, RandomZoom,
                                     RandomTranslation, RandomBrightness, RandomContrast)
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_curve, auc, accuracy_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import seaborn as sns
import numpy as np
import os
import random

# Seed untuk konsistensi
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"[CONFIG] SEED = {SEED}")

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
    possible_paths = [
        '/content/drive/MyDrive/CARD DATASET 2',
        '/content/drive/MyDrive/CARD_DATASET_SMALL',
        '/content/drive/MyDrive/CARD DATASET',
        '/content/drive/MyDrive/CARD DATASET-20260613T040036Z-3-001/CARD DATASET',
        '/content/drive/My Drive/CARD DATASET 2',
    ]
    DATA_DIR = possible_paths[0]
    for path in possible_paths:
        if os.path.exists(path):
            DATA_DIR = path
            print(f"Dataset ditemukan: {DATA_DIR}")
            break
    import shutil
    LOCAL_DIR = '/content/dataset_kartu'
    if not os.path.exists(LOCAL_DIR):
        print("Menyalin dataset ke SSD lokal Colab...")
        try:
            shutil.copytree(DATA_DIR, LOCAL_DIR)
            print("Penyalinan berhasil!")
        except Exception as e:
            print(f"Gagal salin: {e}. Menggunakan path Drive langsung.")
            LOCAL_DIR = DATA_DIR
    else:
        print("Dataset lokal sudah ada.")
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

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

print("\n--- 1. MEMUAT DATASET UNTUK DEEP LEARNING ---")
train_dataset = image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_dataset = image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Kelas yang dideteksi: {class_names}")

# Optimasi Pipeline Data
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (train_dataset
                 .cache()
                 .shuffle(buffer_size=640, seed=SEED, reshuffle_each_iteration=True)
                 .prefetch(buffer_size=AUTOTUNE))
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
print("[INFO] Pipeline data dioptimasi dengan cache(), shuffle(), dan prefetch(AUTOTUNE).")

# Visualisasi Distribusi Dataset
print("\n--- MENAMPILKAN DISTRIBUSI DATASET ---")
class_counts = {}
for c in class_names:
    class_path = os.path.join(DATA_DIR, c)
    if os.path.exists(class_path):
        class_counts[c] = len([f for f in os.listdir(class_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    else:
        class_counts[c] = 200

plt.figure(figsize=(7, 4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.bar(class_counts.keys(), class_counts.values(),
               color=colors, edgecolor='black', alpha=0.8)
plt.title('Distribusi Jumlah Gambar per Kelas Kartu Remi', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Suit', fontsize=10)
plt.ylabel('Jumlah File Gambar', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 5,
             f'{yval}', ha='center', va='bottom', fontweight='bold')
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
print("\n--- MENAMPILKAN CONTOH HASIL AUGMENTASI ---")
plt.figure(figsize=(8, 8))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.suptitle("Visualisasi Data Augmentation (6 Jenis)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualisasi_augmentasi.png', dpi=300)
plt.show()

# Membangun Model MobileNetV2 (Fase 1: Feature Extraction)
print("\n--- 2. MEMBANGUN ARSITEKTUR MODEL ---")
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Training Fase 1
EPOCHS = 30
callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_kartu.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5,
        min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    )
]

print(f"\n--- 3. MEMULAI TRAINING (Maks {EPOCHS} EPOCHS dengan Early Stopping) ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=callbacks_phase1
)
best_phase1_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
best_phase1_acc   = max(history.history['val_accuracy'])
print(f"[SUMMARY] Phase 1 selesai. Best val_accuracy = {best_phase1_acc:.4f} "
      f"(epoch {best_phase1_epoch}/{len(history.history['accuracy'])})")

# Training Fase 2: Fine-Tuning
print("\n--- 3b. MEMULAI FASE 2: FINE-TUNING ---")
FINE_TUNE_AT = 100
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"Layer yang dilatih ulang: dari layer {FINE_TUNE_AT} ke atas")
print(f"Total trainable variables: {len(model.trainable_variables)}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5,
        clipnorm=1.0
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

FINE_TUNE_EPOCHS = 20
INITIAL_EPOCH    = len(history.history['accuracy'])

callbacks_phase2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model_kartu.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8,
        restore_best_weights=True, verbose=1
    )
]

history_fine = model.fit(
    train_dataset,
    epochs=INITIAL_EPOCH + FINE_TUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCH,
    validation_data=val_dataset,
    callbacks=callbacks_phase2
)
best_phase2_epoch = int(np.argmax(history_fine.history['val_accuracy'])) + 1 + INITIAL_EPOCH
best_phase2_acc   = max(history_fine.history['val_accuracy'])
print(f"[SUMMARY] Phase 2 selesai. Best val_accuracy = {best_phase2_acc:.4f} "
      f"(epoch {best_phase2_epoch})")

# Menggabungkan History Pelatihan
acc_total      = history.history['accuracy']     + history_fine.history['accuracy']
val_acc_total  = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss_total     = history.history['loss']         + history_fine.history['loss']
val_loss_total = history.history['val_loss']     + history_fine.history['val_loss']
phase1_end     = len(history.history['accuracy'])

# Visualisasi Kurva Pelatihan
print("\n--- 4. MENAMPILKAN GRAFIK EVALUASI ---")
epochs_range = range(len(acc_total))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc_total,     label='Training Accuracy',   marker='o', color='#1f77b4')
plt.plot(epochs_range, val_acc_total, label='Validation Accuracy', marker='o', color='#2ca02c')
plt.axvline(x=phase1_end - 1, color='gray', linestyle='--', linewidth=1.2,
            label=f'Fine-Tune Start (Epoch {phase1_end})')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_total,     label='Training Loss',   marker='o', color='#d62728')
plt.plot(epochs_range, val_loss_total, label='Validation Loss', marker='o', color='#ff7f0e')
plt.axvline(x=phase1_end - 1, color='gray', linestyle='--', linewidth=1.2,
            label=f'Fine-Tune Start (Epoch {phase1_end})')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

plt.suptitle("Kurva Pelatihan Model - CNN (MobileNetV2)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kurva_pelatihan.png', dpi=300)
plt.show()

# Evaluasi Model Baseline (Tanpa TTA)
print("\n--- 5. EVALUASI MODEL (BASELINE, TANPA TTA) ---")
y_true            = []
y_pred_probs_base = []
for images, labels in val_dataset:
    y_true.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    y_pred_probs_base.extend(preds)

y_true            = np.array(y_true)
y_pred_probs_base = np.array(y_pred_probs_base)
y_pred_base       = np.argmax(y_pred_probs_base, axis=1)
acc_base          = accuracy_score(y_true, y_pred_base)
print(f"[INFO] Akurasi BASELINE (tanpa TTA): {acc_base:.4f}")

# Evaluasi dengan Test-Time Augmentation (TTA)
print("\n--- 6. EVALUASI DENGAN TEST-TIME AUGMENTATION (TTA) ---")
y_pred_probs_tta = []
for images, labels in val_dataset:
    preds_orig = model.predict(images, verbose=0)
    images_flipped = tf.image.flip_left_right(images)
    preds_flip = model.predict(images_flipped, verbose=0)
    preds_avg = (preds_orig + preds_flip) / 2.0
    y_pred_probs_tta.extend(preds_avg)

y_pred_probs_tta = np.array(y_pred_probs_tta)
y_pred_tta       = np.argmax(y_pred_probs_tta, axis=1)
acc_tta          = accuracy_score(y_true, y_pred_tta)
print(f"[INFO] Akurasi DENGAN TTA (flip averaging): {acc_tta:.4f}")
print(f"[INFO] Selisih TTA vs Baseline: {acc_tta - acc_base:+.4f}")

# Memilih mode prediksi terbaik untuk visualisasi detail
if acc_tta >= acc_base:
    print("[INFO] TTA >= Baseline -> menggunakan hasil TTA untuk evaluasi lanjutan.")
    y_pred_probs = y_pred_probs_tta
    y_pred       = y_pred_tta
    EVAL_MODE    = "TTA (flip averaging)"
else:
    print("[INFO] Baseline > TTA -> menggunakan hasil baseline untuk evaluasi lanjutan.")
    y_pred_probs = y_pred_probs_base
    y_pred       = y_pred_base
    EVAL_MODE    = "Baseline (tanpa TTA)"

print(f"\nClassification Report ({EVAL_MODE}):")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualisasi Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 12, "weight": "bold"}, cbar=True)
plt.title(f"Confusion Matrix - CNN (MobileNetV2, {EVAL_MODE})", fontsize=12, fontweight='bold')
plt.ylabel('True Class', fontsize=10)
plt.xlabel('Predicted Class', fontsize=10)
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=300)
plt.show()

# Visualisasi Bar Chart Metrik per Kelas
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=range(num_classes)
)
xpos  = np.arange(num_classes)
width = 0.25

plt.figure(figsize=(8, 5))
plt.bar(xpos - width, precision, width, label='Precision', color='#1f77b4', edgecolor='black', alpha=0.8)
plt.bar(xpos,         recall,    width, label='Recall',    color='#2ca02c', edgecolor='black', alpha=0.8)
plt.bar(xpos + width, f1,        width, label='F1-Score',  color='#ff7f0e', edgecolor='black', alpha=0.8)
plt.title(f'Metrik Evaluasi per Kelas ({EVAL_MODE})', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Suit', fontsize=10)
plt.ylabel('Score (0.0 - 1.0)', fontsize=10)
plt.xticks(xpos, class_names, fontsize=10)
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
for i in range(num_classes):
    plt.text(i - width, precision[i] + 0.02, f'{precision[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
    plt.text(i,         recall[i]    + 0.02, f'{recall[i]:.2f}',    ha='center', fontsize=8, fontweight='bold')
    plt.text(i + width, f1[i]        + 0.02, f'{f1[i]:.2f}',        ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('metrik_evaluasi_cnn.png', dpi=300)
plt.show()

# Visualisasi ROC Curve
y_true_bin = label_binarize(y_true, classes=range(num_classes))
fpr, tpr, roc_auc = {}, {}, {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc[i]        = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors_roc = ['blue', 'orange', 'green', 'red']
for i, color in zip(range(num_classes), colors_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=10)
plt.ylabel('True Positive Rate (TPR)', fontsize=10)
plt.title(f'ROC Curve - Multi-Class ({EVAL_MODE})', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('kurva_roc_cnn.png', dpi=300)
plt.show()

# Visualisasi Grad-CAM
print("\n--- 7. MENAMPILKAN GRAD-CAM VISUALIZATION ---")
classifier_layers = model.layers[-3:]
head_input = tf.keras.Input(shape=base_model.output.shape[1:])
h = head_input
for layer in classifier_layers:
    h = layer(h)
classifier_head = Model(head_input, h, name="classifier_head")

def make_gradcam_heatmap(img_array_0_255, base_model, classifier_head, pred_index=None):
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        tf.cast(img_array_0_255, tf.float32)
    )
    with tf.GradientTape() as tape:
        conv_output = base_model(preprocessed, training=False)
        tape.watch(conv_output)
        preds = classifier_head(conv_output, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads        = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_output[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0)
    max_val      = tf.reduce_max(heatmap)
    heatmap      = heatmap / (max_val + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_np, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet           = mpl_cm.get_cmap("jet")
    jet_colors    = jet(np.arange(256))[:, :3]
    jet_heatmap   = jet_colors[heatmap_uint8]
    jet_img       = tf.keras.utils.array_to_img(jet_heatmap)
    jet_img       = jet_img.resize((img_np.shape[1], img_np.shape[0]))
    jet_arr       = tf.keras.utils.img_to_array(jet_img)
    superimposed  = jet_arr * alpha + img_np
    return tf.keras.utils.array_to_img(superimposed)

plt.figure(figsize=(12, 8))
plot_count = 0
for images, labels in val_dataset.take(1):
    for i in range(min(6, len(images))):
        img_array  = tf.expand_dims(images[i], axis=0)
        pred_probs = model.predict(img_array, verbose=0)[0]
        pred_idx   = int(np.argmax(pred_probs))
        true_idx   = int(labels[i].numpy())
        try:
            heatmap  = make_gradcam_heatmap(img_array, base_model, classifier_head, pred_idx)
            img_np   = images[i].numpy().astype("uint8")
            overlaid = overlay_gradcam(img_np, heatmap)
            color    = 'green' if pred_idx == true_idx else 'red'

            plt.subplot(2, 6, plot_count + 1)
            plt.imshow(img_np)
            plt.title(f"Asli: {class_names[true_idx]}", fontsize=8)
            plt.axis("off")

            plt.subplot(2, 6, plot_count + 7)
            plt.imshow(overlaid)
            plt.title(f"Pred: {class_names[pred_idx]} ({pred_probs[pred_idx]*100:.0f}%)",
                      fontsize=8, color=color)
            plt.axis("off")
            plot_count += 1
        except Exception as e:
            print(f"  Grad-CAM gambar {i} gagal: {e}")

if plot_count > 0:
    plt.suptitle("Grad-CAM: Area Fokus Model\n(Atas = gambar asli | Bawah = heatmap fokus)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradcam_visualisasi.png', dpi=300)
    plt.show()

# Visualisasi Sampel Hasil Prediksi
print("\n--- 8. MENAMPILKAN CONTOH HASIL PREDIKSI ---")
plt.figure(figsize=(10, 10))
for images, labels in val_dataset.take(1):
    preds = model.predict(images, verbose=0)
    for i in range(9):
        true_label = class_names[labels[i]]
        pred_label = class_names[np.argmax(preds[i])]
        confidence = np.max(preds[i]) * 100
        color      = 'green' if true_label == pred_label else 'red'
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Asli: {true_label}\nTebak: {pred_label} ({confidence:.1f}%)",
                  color=color, fontsize=10)
        plt.axis("off")
plt.suptitle("Sampel Hasil Prediksi Model MobileNetV2", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('hasil_prediksi_sampel.png', dpi=300)
plt.show()

# Simpan Model Final
model.save('model_kartu_mobilenet.keras')
print("\n[SUKSES] Model berhasil disimpan dengan nama 'model_kartu_mobilenet.keras'")

# Ringkasan Akhir
print("\n" + "=" * 55)
print("RINGKASAN HASIL RUN INI")
print("=" * 55)
print(f"  SEED                   : {SEED}")
print(f"  FINE_TUNE_AT (layer)   : {FINE_TUNE_AT}")
print(f"  Best val_acc Phase 1   : {best_phase1_acc:.4f} (epoch {best_phase1_epoch})")
print(f"  Best val_acc Phase 2   : {best_phase2_acc:.4f} (epoch {best_phase2_epoch})")
print(f"  Akurasi Baseline       : {acc_base:.4f}")
print(f"  Akurasi dengan TTA     : {acc_tta:.4f}  (selisih {acc_tta - acc_base:+.4f})")
print(f"  Mode evaluasi dipakai  : {EVAL_MODE}")
print("=" * 55)