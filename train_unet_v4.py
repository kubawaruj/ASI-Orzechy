import os
import glob
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---- Config ----
DATA_ROOT = "walnuts/walnuts"
TRAIN_DIR = os.path.join(DATA_ROOT, "train/good")
VAL_DIR = os.path.join(DATA_ROOT, "validation/good")
TEST_PUBLIC_DIR = os.path.join(DATA_ROOT, "test_public")
GT_DIR = os.path.join(TEST_PUBLIC_DIR, "ground_truth/bad")
UNKNOWN_TEST_DIR = os.path.join(DATA_ROOT, "test_private_mixed")

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif")

BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
SEED = 69

EPOCHS_PHASE1 = 60    # encoder frozen
EPOCHS_PHASE2 = 90    # fine-tune encoder
TOTAL_EPOCHS = EPOCHS_PHASE1 + EPOCHS_PHASE2

LR_PHASE1 = 3e-4
LR_PHASE2 = 1e-5

THRESHOLD = 0.2
MODEL_SAVE = "unet_effb0_2phase.keras"

random.seed(SEED)
tf.random.set_seed(SEED)


# ---- Dataset functions ----
def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def find_mask_for_image(image_path, gt_root=GT_DIR):
    name = os.path.splitext(os.path.basename(image_path))[0]
    candidates = glob.glob(os.path.join(gt_root, "**", f"*{name}*.png"), recursive=True)
    return candidates[0] if len(candidates) > 0 else None

def read_image(path, size=IMAGE_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0

def read_mask(path, size=IMAGE_SIZE):
    if path is None:
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.float32)
    return np.expand_dims(m, axis=-1)

def build_pairs(img_dirs, gt_root=GT_DIR):
    pairs = []
    for d in img_dirs:
        for p in list_images(d):
            pairs.append((p, find_mask_for_image(p, gt_root)))
    return pairs

def numpy_loader(image_path, mask_path):
    img = read_image(image_path.decode())
    if mask_path.decode() != "None":
        mask = read_mask(mask_path.decode())
    else:
        mask = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), np.float32)
    return img, mask

def tf_parse(image_path, mask_path):
    img, mask = tf.numpy_function(numpy_loader, [image_path, mask_path], [tf.float32, tf.float32])
    img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    mask.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
    return img, mask

def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)

    noise = tf.random.normal(tf.shape(img), 0.0, 0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, mask

def build_dataset(pairs, batch=BATCH_SIZE, shuffle=True, augment_prob=0.9):
    imgs = [p[0] for p in pairs]
    masks = [p[1] if p[1] is not None else "None" for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    if shuffle:
        ds = ds.shuffle(len(imgs), seed=SEED)

    ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

    def maybe_aug(i, m):
        return tf.cond(
            tf.random.uniform([]) < augment_prob,
            lambda: augment(i, m),
            lambda: (i, m)
        )

    if shuffle:
        ds = ds.map(maybe_aug, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---- Model ----
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet_effb0(input_shape):
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)

    skip1 = base.get_layer("block2a_expand_activation").output
    skip2 = base.get_layer("block3a_expand_activation").output
    skip3 = base.get_layer("block4a_expand_activation").output
    skip4 = base.get_layer("block6a_expand_activation").output

    x = base.get_layer("top_activation").output
    x = conv_block(x, 512)

    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, skip4]); x = conv_block(x, 256)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, skip3]); x = conv_block(x, 128)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, skip2]); x = conv_block(x, 64)
    x = layers.UpSampling2D()(x); x = layers.Concatenate()([x, skip1]); x = conv_block(x, 32)
    x = layers.UpSampling2D()(x); x = conv_block(x, 16)

    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return models.Model(base.input, out)


# ---- Losses, metrics ----
def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=1.5):
    return tf.pow((1 - tversky(y_true, y_pred)), gamma)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    ft = focal_tversky_loss(y_true, y_pred)
    return 0.3 * bce + 0.7 * ft

def dice_non_empty(y_true, y_pred, smooth=1e-6):
    y_sum = tf.reduce_sum(y_true, axis=[1,2,3])
    mask = tf.cast(y_sum > 0, tf.float32)

    yt = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    yp = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter = tf.reduce_sum(yt * yp, axis=1)
    dice = (2*inter + smooth)/(tf.reduce_sum(yt,1)+tf.reduce_sum(yp,1)+smooth)
    return tf.reduce_sum(dice * mask)/(tf.reduce_sum(mask)+smooth)

def precision_non_empty(y_true, y_pred, threshold=THRESHOLD, smooth=1e-6):
    mask = tf.reduce_sum(y_true, axis=[1,2,3]) > 0
    mask = tf.cast(mask, tf.float32)

    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1,2,3])
    fp = tf.reduce_sum((1 - y_true) * y_pred_bin, axis=[1,2,3])

    precision = (tp + smooth) / (tp + fp + smooth)
    return tf.reduce_sum(precision * mask) / (tf.reduce_sum(mask) + smooth)

def recall_non_empty(y_true, y_pred, threshold=THRESHOLD, smooth=1e-6):
    mask = tf.reduce_sum(y_true, axis=[1,2,3]) > 0
    mask = tf.cast(mask, tf.float32)

    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin, axis=[1,2,3])
    fn = tf.reduce_sum(y_true * (1 - y_pred_bin), axis=[1,2,3])

    recall = (tp + smooth) / (tp + fn + smooth)
    return tf.reduce_sum(recall * mask) / (tf.reduce_sum(mask) + smooth)

# ---- Callbacks ----
# ---- Visualization ----
class VisualizePredictions(Callback):
    def __init__(self, sample_paths, interval=20):
        super().__init__()
        self.sample_paths = sample_paths
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        for p in self.sample_paths:
            img = read_image(p)
            pred = self.model.predict(img[None], verbose=0)[0,...,0]
            mask = (pred >= THRESHOLD).astype(np.uint8)
            gt = read_mask(find_mask_for_image(p))[...,0]

            plt.figure(figsize=(10,3))
            plt.subplot(1,3,1); plt.imshow(img); plt.title("Img"); plt.axis("off")
            plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Pred"); plt.axis("off")
            plt.subplot(1,3,3); plt.imshow(gt, cmap="gray"); plt.title("GT"); plt.axis("off")
            plt.show()


# ---- History ----
class HistoryTracker(Callback):
    def __init__(self):
        super().__init__()
        self.history = {
            "loss": [], "val_loss": [],
            "dice_non_empty": [], "val_dice_non_empty": [],
            "precision_non_empty": [], "val_precision_non_empty": [],
            "recall_non_empty": [], "val_recall_non_empty": [],
            "lr": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.history:
            if k == "lr":
                opt = self.model.optimizer
                try:
                    lr = float(tf.keras.backend.get_value(opt.lr))
                except AttributeError:
                    lr = float(tf.keras.backend.get_value(opt.learning_rate))
                self.history["lr"].append(lr)
            else:
                self.history[k].append(logs.get(k))


# ---- History plot ----
def plot_training_curves(hist, save_path="training_curves.png"):
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(18,5))

    # ---- Loss ----
    plt.subplot(1,4,1)
    plt.plot(epochs, hist["loss"], label="train")
    plt.plot(epochs, hist["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Dice ----
    plt.subplot(1,4,2)
    plt.plot(epochs, hist["dice_non_empty"], label="train")
    plt.plot(epochs, hist["val_dice_non_empty"], label="val")
    plt.title("Dice (non-empty)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Precision ----
    plt.subplot(1,4,3)
    plt.plot(epochs, hist["precision_non_empty"], label="train")
    plt.plot(epochs, hist["val_precision_non_empty"], label="val")
    plt.title("Precision (non-empty)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    # ---- Recall ----
    plt.subplot(1,4,4)
    plt.plot(epochs, hist["recall_non_empty"], label="train")
    plt.plot(epochs, hist["val_recall_non_empty"], label="val")
    plt.title("Recall (non-empty)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print("Saved curves to:", save_path)

# ---- Final visualization ----
def visualize_final_predictions(model, sample_paths, title="Sample predictions", threshold=THRESHOLD, max_samples=5):
    for p in sample_paths[:max_samples]:
        img = read_image(p, size=IMAGE_SIZE)
        pred = model.predict(img[None], verbose=0)[0,...,0]
        mask_pred = (pred >= threshold).astype(np.uint8)
        mask_gt_path = find_mask_for_image(p)
        mask_gt = read_mask(mask_gt_path, size=IMAGE_SIZE)[...,0] if mask_gt_path else np.zeros_like(mask_pred)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(mask_pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(mask_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# ---- Unkown samples visualization ----
def predict_unknown_samples(model, folder_path, sample_count=10, threshold=THRESHOLD):
    files = list_images(folder_path)
    if not files:
        print(f"\n[UWAGA] Folder {folder_path} jest pusty lub nie istnieje!")
        return

    count = min(len(files), sample_count)
    sampled_files = random.sample(files, count)
    print(f"\n--- TESTOWANIE NA NIEZNANYCH ORZECHACH ({folder_path}) ---")
    print(f"Wylosowano {count} zdjęć do podglądu.")

    for path in sampled_files:
        img = read_image(path, size=IMAGE_SIZE)
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0, ..., 0]
        mask_pred = (pred >= threshold).astype(np.uint8)

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Plik: {os.path.basename(path)}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask_pred, cmap='gray')
        plt.title("Maska (Model)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ---- Predict, save to file ----
def predict_and_save(model, image_paths, out_dir="predictions", threshold=THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerowanie masek dla {out_dir}...")
    for p in tqdm(image_paths):
        img = read_image(p, size=IMAGE_SIZE)
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0, ..., 0]
        mask = (pred >= threshold).astype(np.uint8) * 255
        name = os.path.basename(p)
        out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_mask.png")
        cv2.imwrite(out_path, mask)


# ---- Main ----
# ---- Data prep ----
train_good = build_pairs([TRAIN_DIR])
bad_dir = os.path.join(TEST_PUBLIC_DIR, "bad")
train_bad = build_pairs([bad_dir], gt_root=GT_DIR)

mult = max(1, len(train_good)//max(1,len(train_bad)))
train_pairs = train_good + train_bad*mult
random.shuffle(train_pairs)

val_pairs = build_pairs([VAL_DIR])

def has_defect(p): return p[1] is not None

defects = [p for p in train_pairs if has_defect(p)]
empty = [p for p in train_pairs if not has_defect(p)]

balanced = []
for _ in range(len(train_pairs)):
    balanced.append(random.choice(defects if random.random()<0.6 else empty))

train_ds = build_dataset(balanced, augment_prob=0.9)
val_ds = build_dataset(val_pairs, shuffle=False, augment_prob=0.0)


# ---- Training ----
model = build_unet_effb0((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

for l in model.layers:
    if isinstance(l, tf.keras.Model):
        l.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
    loss=combined_loss,
    metrics=[dice_non_empty, precision_non_empty, recall_non_empty]
)

hist_cb = HistoryTracker()
sample_vis = random.sample(list_images(bad_dir), min(3, len(train_bad)))
vis_cb1 = VisualizePredictions(sample_vis, interval=60)
ckpt1 = ModelCheckpoint("checkpoints/best_1phase.keras", save_best_only=True)

print("\n=== PHASE 1: TRAIN DECODER ===")
model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=val_ds,
    callbacks=[ckpt1, vis_cb1, hist_cb]
)


# ---- Training - phase 2 ----
print("\n=== PHASE 2: FINE TUNE ENCODER ===")

model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
    loss=combined_loss,
    metrics=[dice_non_empty, precision_non_empty, recall_non_empty]
)

early = EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)
ckpt = ModelCheckpoint("checkpoints/best_2phase.keras", save_best_only=True)
vis_cb = VisualizePredictions(sample_vis, interval=200)

model.fit(
    train_ds,
    epochs=TOTAL_EPOCHS,
    initial_epoch=EPOCHS_PHASE1,
    validation_data=val_ds,
    callbacks=[early, ckpt, vis_cb, hist_cb]
)

model.save(MODEL_SAVE)
print("Saved:", MODEL_SAVE)

plot_training_curves(hist_cb.history)

# ---- Results ----
# ---- Test_public/bad - save to file ----
test_bad = list_images(os.path.join(TEST_PUBLIC_DIR, "bad"))
if len(test_bad) > 0:
    predict_and_save(model, test_bad, out_dir="predictions/test_public_bad", threshold=THRESHOLD)

# ---- Test_public/bad - visualization with mask ground truth ----
sample_paths = random.sample(test_bad, min(5, len(test_bad)))
print("\nWyświetlam przykładowe predykcje z zestawu testowego (z Ground Truth):")
for path in sample_paths:
    img = read_image(path, size=IMAGE_SIZE)
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0, ..., 0]
    mask_pred = (pred >= THRESHOLD).astype(np.uint8)
    mask_gt_path = find_mask_for_image(path)
    mask_gt = read_mask(mask_gt_path, size=IMAGE_SIZE)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Obraz"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(mask_pred, cmap='gray'); plt.title("Predykcja"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(mask_gt[..., 0], cmap='gray'); plt.title("Ground Truth"); plt.axis('off')
    plt.show()


# ---- Test_private_mix - visualization and save to file walnuts without mask ground truth ----
if os.path.exists(UNKNOWN_TEST_DIR):
    predict_unknown_samples(model, UNKNOWN_TEST_DIR, sample_count=10, threshold=THRESHOLD)
    test_unknown = list_images(UNKNOWN_TEST_DIR)
    predict_and_save(model, test_unknown, out_dir="predictions/test_private_mix", threshold=THRESHOLD)
else:
    print(f"\n[INFO] Nie znaleziono folderu '{UNKNOWN_TEST_DIR}'. Pomięto dodatkowy test.")

