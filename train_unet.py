"""
train_unet.py
Segmentacja defektów orzechów włoskich (TensorFlow / Keras)
Założenia:
- struktura danych (root):
    train/good
    validation/good
    test_public/good
    test_public/bad
    test_public/ground_truth
    test_private
    test_private_mixed
- obrazy: dowolne rozmiary, tło czarne
- maski: w folderze test_public/ground_truth (jeśli brak maski -> traktujemy jako "no defect")
Uruchomienie:
    pip install -U tensorflow opencv-python tqdm
    python train_unet.py
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2  # dla czytania/zapisu obrazów
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ---------------------------
# Ustawienia (dostosuj)
# ---------------------------
DATA_ROOT = "walnuts/walnuts"  # <- ustaw tutaj ścieżkę do korzenia datasetu MVTec
TRAIN_DIR = os.path.join(DATA_ROOT, "train/good")
VAL_DIR = os.path.join(DATA_ROOT, "validation/good")
TEST_PUBLIC_DIR = os.path.join(DATA_ROOT, "test_public")
GT_DIR = os.path.join(TEST_PUBLIC_DIR, "ground_truth/bad")  # maski
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif")
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)  # możesz zmniejszyć do (256,256) jeśli GPU ma mało pamięci
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 50
MODEL_SAVE = "unet_effb0_segmentation.h5"
SEED = 42
# ---------------------------
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# Helper: znajdź pliki obrazów
# ---------------------------
def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files = sorted(files)
    return files

# ---------------------------
# Helper: znajdź maskę odpowiadającą obrazowi
# Zakładamy: struktura ground_truth może zawierać podfoldery, maski mogą mieć podobną nazwę
# Strategia: dla obrazu "xxx.png" szukamy "*xxx*.png" w GT_DIR lub podkatalogach,
# jeśli nie znajdziemy -> zwróć None (będziemy generować maskę zerową).
# ---------------------------
def find_mask_for_image(image_path, gt_root=GT_DIR):
    image_name = os.path.basename(image_path)
    name_no_ext = os.path.splitext(image_name)[0]
    # przeszukaj wszystkie pliki png w gt_root
    candidates = glob.glob(os.path.join(gt_root, "**", f"*{name_no_ext}*.png"), recursive=True)
    if len(candidates) > 0:
        return candidates[0]
    # czasami nazwa maski: image_000.png -> mask_000.png; spróbuj dopasować końcówkę numerową
    # fallback: nic nie znaleziono
    return None

# ---------------------------
# Wczytywanie obrazu i maski, preprocessing
# ---------------------------
def read_image(path, size=IMAGE_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nie można wczytać obrazu: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img

def read_mask(path, size=IMAGE_SIZE):
    if path is None:
        # brak maski -> zwróć puste (zero) maski
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    # normalizuj do 0/1 (maska może mieć wartości 0 i 255)
    m = (m > 127).astype(np.float32)
    return np.expand_dims(m, axis=-1)

# ---------------------------
# Generator tworzący listę par (image_path, mask_path_or_None)
# ---------------------------
def build_file_pairs_from_dir(img_dirs, gt_root=GT_DIR):
    pairs = []
    for d in img_dirs:
        imgs = list_images(d)
        for img_path in imgs:
            # spróbuj dopasować maskę
            mask_path = find_mask_for_image(img_path, gt_root=gt_root)
            pairs.append((img_path, mask_path))
    return pairs

# ---------------------------
# tf.data pipeline: z plików na batch
# używamy map z funkcjami numpy->tf, a augmentacje wykonujemy w TF
# ---------------------------
def numpy_loader(image_path, mask_path):
    img = read_image(image_path.decode('utf-8'))
    mask = read_mask(mask_path.decode('utf-8')) if mask_path.decode('utf-8') != 'None' else np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32)
    return img, mask

def tf_parse(image_path, mask_path):
    img, mask = tf.numpy_function(numpy_loader, [image_path, mask_path], [tf.float32, tf.float32])
    img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    mask.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
    return img, mask

# ---------------------------
# Augmentacje w TF
# ---------------------------
def augment(img, mask):
    # losowe odbicia
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    # losowe obroty 0/90/180/270
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    if k > 0:
        img = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)
    # jasność / kontrast
    img = tf.image.random_brightness(img, max_delta=0.08)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # drobne przesunięcie i skalowanie (crop/pad)
    # tu zostawimy prostsze augmentacje — można rozszerzyć
    return img, mask

# ---------------------------
# Dataset builder
# ---------------------------
def build_dataset(pairs, batch=BATCH_SIZE, shuffle=True, augment_prob=0.8):
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] if p[1] is not None else 'None' for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=SEED)
    ds = ds.map(tf_parse, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.map(lambda i, m: (i, m), num_parallel_calls=AUTOTUNE)
    # augmentacje z prawdopodobieństwem
    def maybe_augment(i, m):
        cond = tf.less(tf.random.uniform([], 0, 1.0), augment_prob)
        i2, m2 = tf.cond(cond, lambda: augment(i, m), lambda: (i, m))
        return i2, m2
    ds = ds.map(maybe_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# ---------------------------
# Model: U-Net z EfficientNetB0 jako enkoder
# ---------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def upsample_concat(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return outputs

def build_unet_effb0(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), encoder_trainable=False):
    """
    Budowa U-Net z EfficientNetB0 jako enkoderem.
    Wersja poprawiona - wyjście zgodne z input 256x256.
    """
    def find_skip_layers_by_size(base_model, target_sizes):
        """
        Znajdź warstwy enkodera, których output spatial size odpowiada target_sizes.
        Zwraca listę warstw (od najmniejszej target_size do największej).
        """
        found = {}
        for layer in base_model.layers:
            out_shape = getattr(layer, 'output_shape', None)
            if not out_shape or len(out_shape) < 3:
                continue
            try:
                h = out_shape[1]
                w = out_shape[2]
            except Exception:
                continue
            for t in target_sizes:
                if h == t and w == t:
                    found[t] = layer
        skips = []
        for t in sorted(target_sizes, reverse=True):
            if t in found:
                skips.append(found[t].output)
            else:
                skips.append(None)
        return skips, found

    # ---------------------------
    # Pretrained encoder
    # ---------------------------
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = encoder_trainable

    # oczekiwane spatial sizes dla skipów (input 256 -> [128,64,32,16])
    H = input_shape[0]
    target_sizes = [H // 2, H // 4, H // 8, H // 16]

    skips, found_map = find_skip_layers_by_size(base, target_sizes)

    # debug: pokaż znalezione warstwy
    print("Requested skip sizes:", target_sizes)
    print("Found skip layers (size -> layer name):")
    for size, layer in found_map.items():
        print(f"  {size} -> {layer.name}")

    x = base.output  # bottleneck (np. 8x8 przy input 256)
    x = conv_block(x, 512)

    # Dekoder: używamy skips od najmniejszego (16) do największego (128)
    skips_reversed = list(reversed(skips))
    filters = [256, 128, 64, 32]

    for skip_tensor, f in zip(skips_reversed, filters):
        if skip_tensor is None:
            x = layers.UpSampling2D((2,2))(x)
            x = conv_block(x, f)
        else:
            x = upsample_concat(x, skip_tensor, f)

    # ostatni upsample do rozmiaru wejścia 256x256
    x = layers.UpSampling2D((2,2))(x)  # 128->256
    x = conv_block(x, 16)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model


# ---------------------------
# Losses i metryki: Dice + BCE
# ---------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    d = dice_loss(y_true, y_pred)
    return bce + d

# ---------------------------
# Przygotowanie danych
# ---------------------------
# przykładowo bierzemy wszystkie obrazy z train/good jako train (tu bez masek -> zero mask),
# oraz validation/good jako val; jeśli masz specjalny split z maskami użyj go.
train_pairs = build_file_pairs_from_dir([TRAIN_DIR])
val_pairs = build_file_pairs_from_dir([VAL_DIR])

print(f"Train samples: {len(train_pairs)}, Val samples: {len(val_pairs)}")

train_ds = build_dataset(train_pairs, batch=BATCH_SIZE, shuffle=True, augment_prob=0.9)
val_ds = build_dataset(val_pairs, batch=BATCH_SIZE, shuffle=False, augment_prob=0.0)

# ---------------------------
# Build model
# ---------------------------
model = build_unet_effb0(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), encoder_trainable=False)
model.summary()

# compile
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)])

# ---------------------------
# Callbacks
# ---------------------------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_cb = ModelCheckpoint("checkpoints/best_model.h5", save_best_only=True, monitor='val_loss', mode='min')
reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
#early_cb = EarlyStopping(monitor='val_loss', patience=12, verbose=1, restore_best_weights=True)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, reduce_cb]
    #callbacks=[checkpoint_cb, reduce_cb, early_cb]
)

# zapisz finalny model
model.save(MODEL_SAVE)

# ---------------------------
# Inference helper: zapis progowanych masek
# ---------------------------
def predict_and_save(model, image_paths, out_dir="predictions", threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    for p in tqdm(image_paths):
        img = read_image(p, size=IMAGE_SIZE)
        inp = np.expand_dims(img, axis=0)
        pred = model.predict(inp)[0,...,0]
        mask = (pred >= threshold).astype(np.uint8) * 255
        # zapisz maskę z tą samą nazwą jak obraz
        name = os.path.basename(p)
        out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_mask.png")
        cv2.imwrite(out_path, mask)

# przykładowe użycie na test_public/bad
test_bad = list_images(os.path.join(TEST_PUBLIC_DIR, "bad"))
if len(test_bad) > 0:
    predict_and_save(model, test_bad, out_dir="predictions/test_public_bad", threshold=0.45)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.legend(['train_dice', 'val_dice'])
plt.show()

test_images = list_images(os.path.join(TEST_PUBLIC_DIR, "bad"))
sample_paths = random.sample(test_images, min(5, len(test_images)))

for path in sample_paths:
    # Wczytaj obraz
    img = read_image(path, size=IMAGE_SIZE)

    # Predykcja modelu
    pred = model.predict(np.expand_dims(img, axis=0))[0, ..., 0]
    mask_pred = (pred >= 0.5).astype(np.uint8)

    # Wczytaj ground truth (jeśli dostępna)
    mask_gt_path = find_mask_for_image(path)
    mask_gt = read_mask(mask_gt_path, size=IMAGE_SIZE)

    # Wyświetlanie
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Obraz")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask_pred, cmap='gray')
    plt.title("Maska – predykcja")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(mask_gt[...,0], cmap='gray')
    plt.title("Maska – ground truth")
    plt.axis('off')

    plt.show()


print("Koniec skryptu. Modele i predykcje zapisane.")
