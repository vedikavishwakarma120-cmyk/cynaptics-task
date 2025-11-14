import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

SR = 22050
DURATION = 4.0
N_MELS = 128
HOP_LENGTH = 512

TRAIN_DIR = 'train/train'
TEST_DIR = 'test/test'
CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))

def load_audio(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def make_log_mel(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def file_to_mel(path):
    y = load_audio(path)
    return make_log_mel(y)

X = []
y = []

for cls in CLASS_NAMES:
    cls_folder = os.path.join(TRAIN_DIR, cls)
    for fn in tqdm(os.listdir(cls_folder), desc=f'Loading {cls}'):
        if not fn.lower().endswith(('.wav', '.mp3', '.flac')):
            continue
        path = os.path.join(cls_folder, fn)
        mel = file_to_mel(path)
        X.append(mel)
        y.append(cls)

X = np.array(X)
X = X[..., np.newaxis]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

input_shape = X_train.shape[1:]

def make_model(input_shape, n_classes=len(CLASS_NAMES)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

model = make_model(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[checkpoint, reduce_lr, early]
)

try:
    df_sample = pd.read_csv('sample_submission.csv')
    test_files = df_sample['ID'].values
except FileNotFoundError:
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.wav','.mp3','.flac'))])

X_test = []
for fn in tqdm(test_files, desc='Loading test'):
    path = os.path.join(TEST_DIR, fn)
    mel = file_to_mel(path)
    X_test.append(mel)

X_test = np.array(X_test)[..., np.newaxis]

preds = model.predict(X_test)
pred_idx = np.argmax(preds, axis=1)
pred_labels = le.inverse_transform(pred_idx)

submission = pd.DataFrame({
    'ID': test_files,
    'Class': pred_labels
})
submission.to_csv('submission.csv', index=False)

best_model = tf.keras.models.load_model('best_model.h5')
val_preds_prob = best_model.predict(X_val)
val_preds_idx = np.argmax(val_preds_prob, axis=1)

accuracy = accuracy_score(y_val, val_preds_idx)
print(f"Validation Accuracy: {accuracy:.4f}")
print(classification_report(y_val, val_preds_idx, target_names=le.classes_))

