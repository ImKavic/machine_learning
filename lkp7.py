# Langkah 1 - Siapkan Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_kelulusan.csv")
X = df.drop("Lulus", axis=1)
y = df["Lulus"]

sc = StandardScaler()
Xs = sc.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)


# Langkah 2 — Bangun Model ANN
import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # klasifikasi biner
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])
model.summary()


# Langkah 3 — Training dengan Early Stopping
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32,
    callbacks=[es], verbose=1
)


# Langkah 4 — Evaluasi di Test Set
from sklearn.metrics import classification_report, confusion_matrix

loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print("Test Acc:", acc, "AUC:", auc)

y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))


# Langkah 5 — Visualisasi Learning Curve
import matplotlib.pyplot as plt

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Learning Curve")
plt.tight_layout(); plt.savefig("learning_curve.png", dpi=120)


# Langkah 6 — Eksperimen
# Ubah jumlah neuron (32/64/128) dan catat efeknya.
# Bandingkan Adam vs SGD+momentum (learning rate berbeda).
# Tambahkan regulasi lain: L2, Dropout lebih besar, atau Batch Normalization.
# Laporkan metrik F1 dan AUC selain akurasi.


# =====================
# Langkah 6 — Eksperimen
# 1. Ubah jumlah neuron (32/64/128) dan catat efeknya.
# =====================

# Fungsi untuk membangun dan melatih model dengan jumlah neuron tertentu
def train_model(neurons):
    from keras import layers, Sequential
    from keras.callbacks import EarlyStopping

    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0  # supress output
    )

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neurons: {neurons} | Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}")
    return history

# Eksperimen dengan jumlah neuron: 32, 64, 128
hist_32 = train_model(32)
hist_64 = train_model(64)
hist_128 = train_model(128)

# ==========================================

# =====================
# Langkah 6 — Eksperimen
# 2. Bandingkan Adam vs SGD+momentum (learning rate berbeda).
# =====================

from keras.optimizers import Adam, SGD

# Fungsi untuk membangun dan melatih model dengan optimizer tertentu
def train_model_optimizer(optimizer, neurons=32):
    from keras import layers, Sequential
    from keras.callbacks import EarlyStopping

    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0  # supress output
    )

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Optimizer: {optimizer.get_config()['name']} | Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}")
    return history

# Definisikan beberapa optimizer untuk eksperimen
optimizers = [
    Adam(learning_rate=1e-3),
    Adam(learning_rate=1e-4),
    SGD(learning_rate=1e-2, momentum=0.9),
    SGD(learning_rate=1e-3, momentum=0.9)
]

# Jalankan eksperimen
for opt in optimizers:
    train_model_optimizer(opt, neurons=32)  # tetap gunakan 32 neuron pertama

# ==========================================

# =====================
# Langkah 6 — Eksperimen
# 3. Tambahkan regulasi lain: L2, Dropout lebih besar, atau Batch Normalization.
# =====================

from keras.regularizers import l2
from keras.layers import BatchNormalization

# Fungsi untuk membangun dan melatih model dengan regulasi tambahan
def train_model_regularized(neurons=32, dropout_rate=0.5, l2_lambda=0.01, use_batchnorm=True):
    from keras import layers, Sequential
    from keras.callbacks import EarlyStopping

    model = Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    # Hidden layer pertama dengan L2 dan optional BatchNorm
    model.add(layers.Dense(neurons, activation=None, kernel_regularizer=l2(l2_lambda)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))

    # Hidden layer kedua
    model.add(layers.Dense(16, activation=None, kernel_regularizer=l2(l2_lambda)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(layers.Activation("relu"))

    # Output layer
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )

    # Early stopping
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Regularized Model | Neurons: {neurons}, Dropout: {dropout_rate}, L2: {l2_lambda}, BatchNorm: {use_batchnorm}")
    print(f"Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}\n")
    return history

# Jalankan eksperimen regulasi
hist_reg = train_model_regularized(neurons=32, dropout_rate=0.5, l2_lambda=0.01, use_batchnorm=True)

# ==========================================

# =====================
# Langkah 6 — Eksperimen
# 4. Laporkan metrik F1 dan AUC selain akurasi.
# =====================

from sklearn.metrics import f1_score, roc_auc_score

# Fungsi untuk menghitung metrik tambahan
def report_metrics(model, X_test, y_test, threshold=0.5):
    # Prediksi probabilitas
    y_proba = model.predict(X_test).ravel()
    # Konversi ke kelas 0/1
    y_pred = (y_proba >= threshold).astype(int)
    
    # Hitung F1-score dan AUC
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Test Accuracy (built-in) : {model.evaluate(X_test, y_test, verbose=0)[1]:.3f}")
    print(f"F1-score                 : {f1:.3f}")
    print(f"AUC                      : {auc:.3f}\n")
    return f1, auc

# Contoh penggunaan untuk model original
report_metrics(model, X_test, y_test)

# Bisa juga dipakai untuk model eksperimen lain, misal:
# report_metrics(model_neuron32, X_test, y_test)
# report_metrics(model_optimizer_adam, X_test, y_test)
# report_metrics(model_regulasi, X_test, y_test)

# ==========================================

# =====================
# LANGKAH 6 — EKSPERIMEN LENGKAP
# =====================

print(f"LANGKAH 6 — EKSPERIMEN LENGKAP")

from keras import layers, Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from sklearn.metrics import f1_score, roc_auc_score
from keras.layers import BatchNormalization

# Fungsi untuk report metrik tambahan
def report_metrics(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Test Accuracy (built-in) : {model.evaluate(X_test, y_test, verbose=0)[1]:.3f}")
    print(f"F1-score                 : {f1:.3f}")
    print(f"AUC                      : {auc:.3f}\n")
    return f1, auc

# =====================
# 6.1 Eksperimen Neuron
# =====================
def train_model_neuron(neurons):
    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=32,
                        callbacks=[es], verbose=0)
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neurons: {neurons} | Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}")
    return model, history

# Jalankan eksperimen neuron
model_neuron32, hist_32 = train_model_neuron(32)
model_neuron64, hist_64 = train_model_neuron(64)
model_neuron128, hist_128 = train_model_neuron(128)

# =====================
# 6.2 Eksperimen Optimizer
# =====================
def train_model_optimizer(optimizer, neurons=32):
    model = Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=32,
                        callbacks=[es], verbose=0)
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Optimizer: {optimizer.get_config()['name']} | Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}")
    return model, history

# Definisikan optimizer
optimizers = [
    Adam(learning_rate=1e-3),
    Adam(learning_rate=1e-4),
    SGD(learning_rate=1e-2, momentum=0.9),
    SGD(learning_rate=1e-3, momentum=0.9)
]

# Jalankan eksperimen optimizer
model_optimizer_adam1, hist_opt1 = train_model_optimizer(optimizers[0])
model_optimizer_adam2, hist_opt2 = train_model_optimizer(optimizers[1])
model_optimizer_sgd1, hist_sgd1 = train_model_optimizer(optimizers[2])
model_optimizer_sgd2, hist_sgd2 = train_model_optimizer(optimizers[3])

# =====================
# 6.3 Eksperimen Regulasi (Dropout, L2, BatchNorm)
# =====================
def train_model_regularized(neurons=32, dropout_rate=0.5, l2_lambda=0.01, use_batchnorm=True):
    model = Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    # Hidden layer pertama
    model.add(layers.Dense(neurons, activation=None, kernel_regularizer=l2(l2_lambda)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))

    # Hidden layer kedua
    model.add(layers.Dense(16, activation=None, kernel_regularizer=l2(l2_lambda)))
    if use_batchnorm:
        model.add(BatchNormalization())
    model.add(layers.Activation("relu"))

    # Output
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=32,
                        callbacks=[es], verbose=0)
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Regularized Model | Neurons: {neurons}, Dropout: {dropout_rate}, L2: {l2_lambda}, BatchNorm: {use_batchnorm}")
    print(f"Test Accuracy: {acc:.3f} | Test AUC: {auc:.3f}\n")
    return model, history

# Jalankan eksperimen regulasi
model_regulasi, hist_reg = train_model_regularized(neurons=32, dropout_rate=0.5, l2_lambda=0.01, use_batchnorm=True)

# =====================
# 6.4 Laporan metrik F1 & AUC
# =====================
# Contoh penggunaan untuk semua model
report_metrics(model_neuron32, X_test, y_test)
report_metrics(model_optimizer_adam1, X_test, y_test)
report_metrics(model_regulasi, X_test, y_test)
