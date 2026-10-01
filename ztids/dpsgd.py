import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import time, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from .dp import epsilon, calibrate_sigma

from . import models as _models  # noqa: F401  (applies the thread configuration)


def dp_cnn_bilstm(n_feat=25, n_cls=5, width=0.25):
    """Same topology as the proposed detector, but DP-compatible: LayerNorm, no dropout, reduced width."""
    f1, f2, u = int(256 * width), int(128 * width), int(128 * width)
    inp = keras.Input((n_feat, 1))
    x = L.Conv1D(f1, 3, activation="relu")(inp); x = L.LayerNormalization()(x)
    x = L.Conv1D(f2, 3, activation="relu")(x); x = L.LayerNormalization()(x)
    x = L.Bidirectional(L.LSTM(u, unroll=True))(x)
    x = L.Dense(int(256 * width), activation="relu")(x)
    x = L.Dense(int(128 * width), activation="relu")(x)
    return keras.Model(inp, L.Dense(n_cls, activation="softmax")(x))


def train_dpsgd(Xfit, yfit, Xval, yval, target_eps, delta=None, epochs=6, batch=256, clip=1.0, lr=2e-3,
                width=0.25, seed=0, max_class_w=20.0, verbose=True):
    """Poisson-subsampled DP-SGD. Returns model, achieved eps, sigma, steps."""
    tf.keras.utils.set_random_seed(seed); rng = np.random.RandomState(seed)
    N = len(Xfit); q = batch / N; steps = int(epochs * N / batch); delta = delta or 1.0 / N
    if target_eps is None:                                         # non-private control (same model, no clipping/noise)
        sigma, eps, clip = 0.0, float("inf"), 1e6
    else:
        sigma = calibrate_sigma(q, steps, target_eps, delta); eps = epsilon(q, sigma, steps, delta)
    cnt = np.bincount(yfit, minlength=5).astype(float)            # class weights from PUBLIC priors (declared assumption)
    cw = np.minimum(cnt.sum() / (5 * np.maximum(cnt, 1)), max_class_w).astype(np.float32)
    model = dp_cnn_bilstm(width=width); opt = keras.optimizers.Adam(lr)
    pad = int(batch + 6 * np.sqrt(batch))                          # fixed padded batch; Poisson size ~ Bin(N, q)
    tv = model.trainable_variables

    @tf.function
    def step(x, y, w):
        def one(args):
            xi, yi, wi = args
            with tf.GradientTape() as t:
                p = model(xi[None], training=False)
                loss = wi * tf.keras.losses.sparse_categorical_crossentropy(yi[None], p)[0]
            g = t.gradient(loss, tv)
            return g
        grads = tf.vectorized_map(one, (x, y, w), fallback_to_while_loop=True)
        sq = tf.add_n([tf.reduce_sum(tf.reshape(g, [pad, -1]) ** 2, axis=1) for g in grads])
        scale = tf.minimum(1.0, clip / (tf.sqrt(sq) + 1e-12)) * tf.cast(w > 0, tf.float32)
        out = []
        for g in grads:
            gs = tf.reduce_sum(g * tf.reshape(scale, [-1] + [1] * (len(g.shape) - 1)), axis=0)
            out.append((gs + tf.random.normal(tf.shape(gs), stddev=sigma * clip)) / batch)
        opt.apply_gradients(zip(out, tv))

    t0 = time.time()
    for s in range(steps):
        idx = np.where(rng.rand(N) < q)[0]
        if len(idx) > pad: idx = idx[:pad]
        x = np.zeros((pad, Xfit.shape[1], 1), np.float32); y = np.zeros(pad, np.int32); w = np.zeros(pad, np.float32)
        x[:len(idx), :, 0] = Xfit[idx]; y[:len(idx)] = yfit[idx]; w[:len(idx)] = cw[yfit[idx]]
        step(tf.constant(x), tf.constant(y), tf.constant(w))
        if verbose and (s % 100 == 0 or s == steps - 1):
            print(f"  dp step {s+1}/{steps} {time.time()-t0:.0f}s", flush=True)
    return model, eps, sigma, steps, delta
