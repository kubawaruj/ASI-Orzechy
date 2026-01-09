import os
import glob
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# ---------------------------
DATA_ROOT = "walnuts/walnuts" 
TRAIN_DIR = os.path.join(DATA_ROOT, "train/good")
VAL_DIR = os.path.join(DATA_ROOT, "validation/good")
TEST_PUBLIC_DIR = os.path.join(DATA_ROOT, "test_public")
GT_DIR = os.path.join(TEST_PUBLIC_DIR, "ground_truth/bad") 
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif")
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)  #(256,256), (512,512), 
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 40
MODEL_SAVE = "unet_effb0_segmentation.keras"
SEED = 0
THRESHOLD = 0.45

# ---------------------------
random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    files = sorted(files)
    return files

# ---------------------------
def find_mask_for_image(image_path, gt_root=GT_DIR):
    image_name = os.path.basename(image_path)
    name_no_ext = os.path.splitext(image_name)[0]
    candidates = glob.glob(os.path.join(gt_root, "**", f"*{name_no_ext}*.png"), recursive=True)
    if len(candidates) > 0:
        return candidates[0]
    return None

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
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((size[0], size[1], 1), dtype=np.float32)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.float32)
    return np.expand_dims(m, axis=-1)


# ---------------------------
def build_file_pairs_from_dir(img_dirs, gt_root=GT_DIR):
    pairs = []
    for d in img_dirs:
        imgs = list_images(d)
        for img_path in imgs:
            mask_path = find_mask_for_image(img_path, gt_root=gt_root)
            pairs.append((img_path, mask_path))
    return pairs

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
    # delikatny blur
    img_uint8 = tf.cast(img * 255.0, tf.uint8)
    img_uint8 = tf.image.random_jpeg_quality(img_uint8, 80, 100)
    img = tf.cast(img_uint8, tf.float32) / 255.0
    # losowy gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, mask

# ---------------------------
def build_dataset(pairs, batch=BATCH_SIZE, shuffle=True, augment_prob=0.9):
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] if p[1] is not None else 'None' for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(img_paths), seed=SEED)
    ds = ds.map(tf_parse, num_parallel_calls=1)
    if shuffle:
        ds = ds.map(lambda i, m: (i, m), num_parallel_calls=1)
    def maybe_augment(i, m):
        cond = tf.less(tf.random.uniform([], 0, 1.0), augment_prob)
        i2, m2 = tf.cond(cond, lambda: augment(i, m), lambda: (i, m))
        return i2, m2
    ds = ds.map(maybe_augment, num_parallel_calls=1)
    ds = ds.batch(batch).prefetch(1)
    return ds


# ---------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def upsample_concat(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return outputs

def build_unet_effb0(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    encoder_trainable=False
):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base.trainable = encoder_trainable

    skip1 = base.get_layer("block2a_expand_activation").output  # 256×256
    skip2 = base.get_layer("block3a_expand_activation").output  # 128×128
    skip3 = base.get_layer("block4a_expand_activation").output  # 64×64
    skip4 = base.get_layer("block6a_expand_activation").output  # 32×32

    x = base.get_layer("top_activation").output  # 16×16
    x = conv_block(x, 512)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip4])
    x = conv_block(x, 256)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip3])
    x = conv_block(x, 128)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip2])
    x = conv_block(x, 64)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Concatenate()([x, skip1])
    x = conv_block(x, 32)

    x = layers.UpSampling2D((2, 2))(x)
    x = conv_block(x, 16)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model


# ---------------------------
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def weighted_bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)

    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=1.5):
    tv = tversky(y_true, y_pred)
    return tf.pow((1 - tv), gamma)

def dice_non_empty(y_true, y_pred, smooth=1e-6):
    y_true_sum = tf.reduce_sum(y_true, axis=[1,2,3])
    mask = tf.cast(y_true_sum > 0, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=1) +
        tf.reduce_sum(y_pred_f, axis=1) +
        smooth
    )

    return tf.reduce_sum(dice * mask) / (tf.reduce_sum(mask) + smooth)

def combined_loss(y_true, y_pred):
    return 0.3 * tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + 0.7 * focal_tversky_loss(y_true, y_pred)

# ---------------------------
class VisualizePredictions(tf.keras.callbacks.Callback):
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        pass

    def __init__(self, sample_paths, interval=5, threshold=THRESHOLD):
        super().__init__()
        self.sample_paths = sample_paths
        self.interval = interval
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return

        for path in self.sample_paths:
            img = read_image(path, size=IMAGE_SIZE)
            pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0, ..., 0]
            mask_pred = (pred >= self.threshold).astype(np.uint8)

            mask_gt_path = find_mask_for_image(path)
            mask_gt = read_mask(mask_gt_path, size=IMAGE_SIZE)

            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,4))
            plt.subplot(1,3,1)
            plt.imshow(img)
            plt.title("Obraz")
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(mask_pred, cmap='gray')
            plt.title("Maska - predykcja")
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(mask_gt[...,0], cmap='gray')
            plt.title("Maska - ground truth")
            plt.axis('off')
            plt.show()



# ---------------------------
train_good_pairs = build_file_pairs_from_dir([TRAIN_DIR])
BAD_TRAIN_DIR = os.path.join(TEST_PUBLIC_DIR, "bad")
train_bad_pairs = build_file_pairs_from_dir([BAD_TRAIN_DIR], gt_root=GT_DIR)

mult = max(1, len(train_good_pairs) // max(1, len(train_bad_pairs)))
train_bad_pairs_oversampled = train_bad_pairs * mult

train_pairs = train_good_pairs + train_bad_pairs_oversampled
random.shuffle(train_pairs)

val_pairs = build_file_pairs_from_dir([VAL_DIR])

def has_defect(pair):
    return pair[1] is not None

pairs_defect = [p for p in train_pairs if has_defect(p)]
pairs_empty = [p for p in train_pairs if not has_defect(p)]

train_pairs_balanced = []
for _ in range(len(train_pairs)):
    if random.random() < 0.6:
        train_pairs_balanced.append(random.choice(pairs_defect))
    else:
        train_pairs_balanced.append(random.choice(pairs_empty))


print(f"Train samples: {len(train_pairs_balanced)}, Val samples: {len(val_pairs)}")
train_ds = build_dataset(train_pairs_balanced, batch=BATCH_SIZE, shuffle=True, augment_prob=0.9)
val_ds = build_dataset(val_pairs, batch=BATCH_SIZE, shuffle=False, augment_prob=0.0)

# ---------------------------
if os.path.exists(MODEL_SAVE):
    print(f"Wczytuję istniejący model z pliku: {MODEL_SAVE}")
    model = tf.keras.models.load_model(
        MODEL_SAVE,
        custom_objects={'combined_loss': combined_loss, 'dice_non_empty': dice_non_empty}
    )
else:
    print("Tworzę nowy model U-Net z EfficientNetB0")
    model = build_unet_effb0(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), encoder_trainable=False)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
              loss=combined_loss, 
              metrics=[dice_non_empty])

model.summary()

# ---------------------------
img_batch, mask_batch = next(iter(train_ds))
print("Image batch shape:", img_batch.shape)
print("Mask batch shape:", mask_batch.shape)
print("Mask mean value:", tf.reduce_mean(mask_batch).numpy())


import matplotlib.pyplot as plt
for i in range(min(3, mask_batch.shape[0])):
    plt.imshow(mask_batch[i, ..., 0], cmap='gray')
    plt.title(f"Mask {i}")
    plt.show()

# ---------------------------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_cb = ModelCheckpoint("checkpoints/best_model.keras", save_best_only=True, monitor='val_loss', mode='min')
reduce_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

sample_paths = random.sample(list_images(BAD_TRAIN_DIR), min(3, len(train_bad_pairs)))
visual_cb = VisualizePredictions(sample_paths, interval=20, threshold=THRESHOLD)

# ---------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, early_cb, visual_cb]
    #callbacks=[checkpoint_cb, reduce_cb, early_cb, visual_cb]
    #callbacks=[checkpoint_cb, reduce_cb]
    #callbacks=[checkpoint_cb, reduce_cb, early_cb]
)

model.save(MODEL_SAVE)

# ---------------------------
def predict_and_save(model, image_paths, out_dir="predictions", threshold=THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)
    for p in tqdm(image_paths):
        img = read_image(p, size=IMAGE_SIZE)
        inp = np.expand_dims(img, axis=0)
        pred = model.predict(inp)[0,...,0]
        mask = (pred >= threshold).astype(np.uint8) * 255
        name = os.path.basename(p)
        out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_mask.png")
        cv2.imwrite(out_path, mask)

test_bad = list_images(os.path.join(TEST_PUBLIC_DIR, "bad"))
if len(test_bad) > 0:
    predict_and_save(model, test_bad, out_dir="predictions/test_public_bad", threshold=THRESHOLD)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['dice_non_empty'])
plt.plot(history.history['val_dice_non_empty'])
plt.legend(['train_dice', 'val_dice'])
plt.show()

test_images = list_images(os.path.join(TEST_PUBLIC_DIR, "bad"))
sample_paths = random.sample(test_images, min(5, len(test_images)))

for path in sample_paths:
    img = read_image(path, size=IMAGE_SIZE)

    pred = model.predict(np.expand_dims(img, axis=0))[0, ..., 0]
    mask_pred = (pred >= 0.5).astype(np.uint8)

    mask_gt_path = find_mask_for_image(path)
    mask_gt = read_mask(mask_gt_path, size=IMAGE_SIZE)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Obraz")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(mask_pred, cmap='gray')
    plt.title("Maska - predykcja")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(mask_gt[...,0], cmap='gray')
    plt.title("Maska - ground truth")
    plt.axis('off')

    plt.show()

pred.mean()
print("Koniec skryptu. Modele i predykcje zapisane.")
