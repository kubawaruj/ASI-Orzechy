import os, glob, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------
DATA_ROOT = "walnuts/walnuts"
TRAIN_DIR = os.path.join(DATA_ROOT, "train/good")
VAL_DIR   = os.path.join(DATA_ROOT, "validation/good")
TEST_DIR  = os.path.join(DATA_ROOT, "test_public")
GT_DIR    = os.path.join(TEST_DIR, "ground_truth/bad")

IMAGE_SIZE = (256, 256) #256, 256 ; 512, 512
BATCH_SIZE = 8
EPOCHS = 5
SEED = 0
MODEL_SAVE = "unet_effb0_segmentation.keras"

TRAIN_PHASE = 2   # 1 = frozen encoder, 2 = fine-tuning

random.seed(SEED)
tf.random.set_seed(SEED)

# DATA ---------------------------
def list_images(folder):
    files = []
    for ext in (".png", ".jpg", ".jpeg", ".tif"):
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)

def find_mask_for_image(img_path):
    name = os.path.splitext(os.path.basename(img_path))[0]
    masks = glob.glob(os.path.join(GT_DIR, "**", f"*{name}*.png"), recursive=True)
    return masks[0] if masks else None

def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img.astype(np.float32) / 255.0

def read_mask(path):
    if path is None:
        return np.zeros((*IMAGE_SIZE, 1), np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    return np.expand_dims((m > 127).astype(np.float32), -1)

def tf_parse(img_p, mask_p):
    img, mask = tf.numpy_function(
        lambda i, m: (read_image(i.decode()), read_mask(m.decode()) if m.decode() != "None" else read_mask(None)),
        [img_p, mask_p],
        [tf.float32, tf.float32]
    )
    img.set_shape((*IMAGE_SIZE, 3))
    mask.set_shape((*IMAGE_SIZE, 1))
    return img, mask

def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    img += tf.random.normal(tf.shape(img), 0, 0.02)
    return tf.clip_by_value(img, 0, 1), mask

def build_dataset(pairs, augment_prob=0.9, shuffle=True):
    imgs = [p[0] for p in pairs]
    masks = [p[1] if p[1] else "None" for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((imgs, masks))
    if shuffle:
        ds = ds.shuffle(len(imgs), seed=SEED)
    ds = ds.map(tf_parse, num_parallel_calls=1)
    ds = ds.map(lambda i, m:
        tf.cond(tf.random.uniform(()) < augment_prob,
                lambda: augment(i, m),
                lambda: (i, m)))
    return ds.batch(BATCH_SIZE).prefetch(1)

# MODEL ---------------------------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def build_unet_effb0(train_encoder=False):
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = train_encoder

    skips = [
        base.get_layer("block2a_expand_activation").output,
        base.get_layer("block3a_expand_activation").output,
        base.get_layer("block4a_expand_activation").output,
        base.get_layer("block6a_expand_activation").output,
    ]

    x = conv_block(base.get_layer("top_activation").output, 512)

    for s, f in zip(reversed(skips), [256, 128, 64, 32]):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, s])
        x = conv_block(x, f)

    x = layers.UpSampling2D()(x)
    x = conv_block(x, 16)

    out = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return models.Model(base.input, out)

# LOSS, METRICS ---------------------------
def dice_non_empty(y_true, y_pred, smooth=1e-6):
    mask = tf.reduce_sum(y_true, axis=[1,2,3]) > 0
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    inter = tf.reduce_sum(y_true * y_pred)
    return (2*inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def tversky(y_true, y_pred, alpha=0.8, beta=0.2, smooth=1e-6):
    tp = tf.reduce_sum(y_true * y_pred)
    fn = tf.reduce_sum(y_true * (1-y_pred))
    fp = tf.reduce_sum((1-y_true) * y_pred)
    return (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=1.2):
    return tf.pow(1 - tversky(y_true, y_pred), gamma)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return 0.3 * bce + 0.7 * focal_tversky_loss(y_true, y_pred)

# TRAIN ---------------------------
def build_pairs(folder):
    return [(p, find_mask_for_image(p)) for p in list_images(folder)]

train_pairs = build_pairs(TRAIN_DIR) + build_pairs(os.path.join(TEST_DIR, "bad"))
val_pairs   = build_pairs(VAL_DIR)

train_ds = build_dataset(train_pairs)
val_ds   = build_dataset(val_pairs, augment_prob=0.0, shuffle=False)

if os.path.exists(MODEL_SAVE):
    model = tf.keras.models.load_model(MODEL_SAVE,
        custom_objects={"combined_loss": combined_loss, "dice_non_empty": dice_non_empty})
else:
    model = build_unet_effb0(train_encoder=(TRAIN_PHASE == 2))

lr = 1e-4 if TRAIN_PHASE == 1 else 1e-5

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=combined_loss,
    metrics=[dice_non_empty]
)

callbacks = [
    ModelCheckpoint("checkpoints/best.keras", save_best_only=True),
    ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
    EarlyStopping(patience=10, restore_best_weights=True)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save(MODEL_SAVE)

print("TRENING ZAKOŃCZONY")
