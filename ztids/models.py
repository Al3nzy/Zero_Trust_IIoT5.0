"""Model zoo. The proposed detector keeps the architecture of the original manuscript (CNN-BiLSTM); the normalisation layers are LayerNorm by
default (BatchNorm collapsed on outlier-heavy rare-event features; set ZTIDS_NORM=bn to reproduce the original)."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from . import config as C

if C.THREADS > 0:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(C.THREADS)
        tf.config.threading.set_inter_op_parallelism_threads(C.THREADS)
    except RuntimeError:
        pass


@tf.keras.utils.register_keras_serializable(package="ztids")
class TemporalAttention(L.Layer):
    """Additive attention over the sequence axis (one half of a 'dual attention' block)."""
    def build(self, s):
        self.w = self.add_weight(name="w", shape=(s[-1], 1), initializer="glorot_uniform")
    def call(self, x):
        a = tf.nn.softmax(tf.matmul(x, self.w), axis=1)
        return x * a * tf.cast(tf.shape(x)[1], x.dtype)


@tf.keras.utils.register_keras_serializable(package="ztids")
class ChannelAttention(L.Layer):
    """Squeeze-and-excitation style attention over the channel axis (other half)."""
    def __init__(self, r=4, **kw):
        super().__init__(**kw); self.r = r
    def build(self, s):
        self.d1 = L.Dense(max(s[-1] // self.r, 4), activation="relu"); self.d2 = L.Dense(s[-1], activation="sigmoid")
    def call(self, x):
        return x * self.d2(self.d1(tf.reduce_mean(x, axis=1)))[:, None, :]
    def get_config(self):
        c = super().get_config(); c.update(r=self.r); return c


def _norm(kind):
    return L.BatchNormalization() if kind == "bn" else L.LayerNormalization()


def cnn_bilstm(n_feat=25, n_cls=5, norm=None, attention=False, width=1.0):
    norm = norm or C.NORM
    f1, f2, u = int(256 * width), int(128 * width), int(128 * width)
    inp = keras.Input((n_feat, 1))
    x = L.Conv1D(f1, 3, activation="relu")(inp); x = _norm(norm)(x); x = L.Dropout(0.25)(x)
    x = L.Conv1D(f2, 3, activation="relu")(x);  x = _norm(norm)(x); x = L.Dropout(0.30)(x)
    if attention:                                    # reimplemented dual-attention variant (channel + temporal)
        x = ChannelAttention()(x); x = TemporalAttention()(x)
    x = L.Bidirectional(L.LSTM(u))(x)
    x = L.Dense(256, activation="relu")(x); x = L.Dropout(0.30)(x)
    x = L.Dense(128, activation="relu")(x); x = L.Dropout(0.25)(x)
    out = L.Dense(n_cls, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_bilstm" + ("_attn" if attention else ""))


def mlp(n_feat=25, n_cls=5):
    inp = keras.Input((n_feat, 1)); x = L.Flatten()(inp)
    for u, d in ((256, .3), (128, .25)):
        x = L.Dense(u, activation="relu")(x); x = L.Dropout(d)(x)
    return keras.Model(inp, L.Dense(n_cls, activation="softmax")(x), name="mlp")


def cnn_only(n_feat=25, n_cls=5):
    inp = keras.Input((n_feat, 1))
    x = L.Conv1D(256, 3, activation="relu")(inp); x = _norm(C.NORM)(x); x = L.Dropout(.25)(x)
    x = L.Conv1D(128, 3, activation="relu")(x);  x = _norm(C.NORM)(x); x = L.Dropout(.3)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dense(256, activation="relu")(x); x = L.Dropout(.3)(x)
    x = L.Dense(128, activation="relu")(x); x = L.Dropout(.25)(x)
    return keras.Model(inp, L.Dense(n_cls, activation="softmax")(x), name="cnn_only")


def lstm_only(n_feat=25, n_cls=5):
    inp = keras.Input((n_feat, 1))
    x = L.LSTM(128)(inp)
    x = L.Dense(256, activation="relu")(x); x = L.Dropout(.3)(x)
    x = L.Dense(128, activation="relu")(x); x = L.Dropout(.25)(x)
    return keras.Model(inp, L.Dense(n_cls, activation="softmax")(x), name="lstm_only")


BUILDERS = {"cnn_bilstm": cnn_bilstm, "cnn_bilstm_attn": lambda **k: cnn_bilstm(attention=True, **k),
            "mlp": mlp, "cnn": cnn_only, "lstm": lstm_only}
CUSTOM_OBJECTS = {"TemporalAttention": TemporalAttention, "ChannelAttention": ChannelAttention}


def compile_model(m, lr=5e-4):
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def fit_model(m, Xtr, ytr, Xval, yval, epochs=14, batch=256, patience=3, verbose=0, seed=0, val_weighting=None):
    val_weighting = val_weighting or C.VAL_WEIGHTING
    tf.keras.utils.set_random_seed(seed)
    vd = (Xval[..., None], yval)
    if val_weighting != "none":                      # class-aware validation loss so that rare classes influence early stopping
        cnt = np.bincount(yval, minlength=int(max(yval.max(), ytr.max())) + 1).astype(float); pw = {"sqrt": 0.5, "balanced": 1.0}[val_weighting]
        w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1) ** pw, 0.0)[yval]; vd = (vd[0], vd[1], (w / w.mean()).astype("float32"))
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    return m.fit(Xtr[..., None], ytr, validation_data=vd, epochs=epochs,
                 batch_size=batch, callbacks=cb, verbose=verbose)


def predict(m, X, batch=1024):
    return m.predict(X[..., None], batch_size=batch, verbose=0)


def load(path):
    return keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
