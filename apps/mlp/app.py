import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap

# stlite ships a stub pyarrow without the classes sklearn's input validation
# probes with isinstance(); give the stub harmless placeholders
import pyarrow
for _n in ("Table", "RecordBatch", "ChunkedArray", "Array"):
    if not hasattr(pyarrow, _n):
        setattr(pyarrow, _n, type(_n, (), {}))

st.set_page_config(page_title="MLP from scratch", layout="wide")

INK, MOSS, RUST, GOLD = "#2D2D2D", "#C17A3E", "#A8392A", "#2F7069"
STRONG = [MOSS, RUST, GOLD]
DIM = ListedColormap(["#F0DCC6", "#F3DBD5", "#D3E5E2"])

st.title("A multilayer perceptron, from scratch")
st.caption("No framework — ~60 lines of NumPy. Forward pass, backprop, and the same "
           "optimizers from the previous app, applied to a real (tiny) network. "
           "Train once, then **scrub through the epochs** to watch it learn.")


def quiz(key, question, options, correct, explain):
    with st.expander("🧠 Check yourself"):
        ans = st.radio(question, options, index=None, key=f"quiz_{key}")
        if ans is not None:
            if options.index(ans) == correct:
                st.success("Correct — " + explain)
            else:
                st.error("Not quite — " + explain)


# ---------------------------------------------------------------- data
def make_data(kind, n, noise, seed):
    rng = np.random.default_rng(seed)
    if kind == "moons":
        from sklearn.datasets import make_moons
        return make_moons(n_samples=n, noise=noise, random_state=seed)
    if kind == "circles":
        from sklearn.datasets import make_circles
        return make_circles(n_samples=n, noise=noise, factor=0.45, random_state=seed)
    if kind == "xor":
        X = rng.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += rng.normal(0, noise, X.shape)
        return X, y
    # 3-class spiral
    per = n // 3
    X, y = [], []
    for c in range(3):
        t = np.linspace(0, 2.2 * np.pi, per) + rng.normal(0, noise * 2.2, per)
        r = np.linspace(0.08, 1.0, per)
        X.append(np.c_[r * np.cos(t + c * 2 * np.pi / 3), r * np.sin(t + c * 2 * np.pi / 3)])
        y.append(np.full(per, c))
    return np.vstack(X), np.concatenate(y)


# ---------------------------------------------------------------- model
ACTS = {
    "tanh": (np.tanh, lambda a: 1 - a ** 2),
    "relu": (lambda z: np.maximum(0, z), lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),
}


def init_net(sizes, act, seed):
    rng = np.random.default_rng(seed)
    params = []
    for i in range(len(sizes) - 1):
        scale = np.sqrt(2 / sizes[i]) if act == "relu" else np.sqrt(1 / sizes[i])
        params.append({"W": rng.normal(0, scale, (sizes[i], sizes[i + 1])),
                       "b": np.zeros(sizes[i + 1])})
    return params


def forward(params, X, act):
    f, _ = ACTS[act]
    a = X; cache = [a]
    for i, p in enumerate(params):
        z = a @ p["W"] + p["b"]
        a = f(z) if i < len(params) - 1 else z   # last layer: logits
        cache.append(a)
    z = cache[-1] - cache[-1].max(axis=1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
    return probs, cache


def backward(params, cache, probs, y_onehot, act):
    _, df = ACTS[act]
    n = len(y_onehot)
    grads = []
    delta = (probs - y_onehot) / n                      # dL/dlogits (softmax + CE)
    for i in reversed(range(len(params))):
        a_prev = cache[i]
        grads.insert(0, {"W": a_prev.T @ delta, "b": delta.sum(0)})
        if i > 0:
            delta = (delta @ params[i]["W"].T) * df(cache[i])
    return grads


def make_optimizer(name, lr):
    state = {}
    def step(params, grads, t):
        for i, (p, g) in enumerate(zip(params, grads)):
            for k in ("W", "b"):
                key = (i, k)
                if name == "SGD":
                    p[k] -= lr * g[k]
                elif name == "Momentum":
                    v = state.get(key, 0)
                    v = 0.9 * v - lr * g[k]; state[key] = v
                    p[k] += v
                else:  # Adam
                    m, v = state.get(key, (0, 0))
                    m = 0.9 * m + 0.1 * g[k]
                    v = 0.999 * v + 0.001 * g[k] ** 2
                    state[key] = (m, v)
                    mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
                    p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return step


# ---------------------------------------------------------------- controls
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("Data")
    ds = st.selectbox("Dataset", ["moons", "circles", "xor", "spiral (3 classes)"])
    ds_key = "spiral" if ds.startswith("spiral") else ds
    n = st.slider("Points", 60, 400, 200, 20)
    noise = st.slider("Noise", 0.0, 0.4, 0.15, 0.05)

    st.subheader("Network")
    hidden = st.text_input("Hidden layers (units, comma-separated)", "8,8")
    act = st.selectbox("Activation", ["tanh", "relu", "sigmoid"])

    st.subheader("Training")
    optname = st.selectbox("Optimizer", ["Adam", "SGD", "Momentum"])
    lr = st.select_slider("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], 0.03)
    epochs = st.slider("Epochs", 100, 3000, 800, 100)
    batch = st.select_slider("Batch size", [16, 32, 64, "full"], "full")
    seed = st.number_input("Seed", 0, 999, 0)
    go = st.button("Train", type="primary", width="stretch")

try:
    hidden_sizes = [int(s) for s in hidden.replace(" ", "").split(",") if s]
    assert all(1 <= h <= 128 for h in hidden_sizes) and 1 <= len(hidden_sizes) <= 4
except Exception:
    st.error("Hidden layers must be 1–4 comma-separated integers between 1 and 128, e.g. `8,8`.")
    st.stop()

X, y = make_data(ds_key, n, noise, int(seed))
n_classes = len(np.unique(y))
sizes = [2] + hidden_sizes + [n_classes]
n_params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
cfg_now = (ds_key, n, noise, tuple(hidden_sizes), act, optname, lr, epochs, str(batch), int(seed))

with c2:
    st.markdown(f"**Architecture:** `2 → {' → '.join(map(str, hidden_sizes))} → {n_classes}`"
                f" &nbsp;·&nbsp; {n_params} trainable parameters &nbsp;·&nbsp; "
                f"softmax output, cross-entropy loss")

    if go:
        params = init_net(sizes, act, int(seed))
        opt = make_optimizer(optname, lr)
        Y = np.eye(n_classes)[y]
        rng = np.random.default_rng(int(seed))
        losses, accs, snaps = [], [], []
        every = max(1, epochs // 30)
        prog = st.progress(0.0, "training…")
        t = 0
        for ep in range(1, epochs + 1):
            if batch == "full":
                idx_batches = [np.arange(len(X))]
            else:
                perm = rng.permutation(len(X))
                idx_batches = np.array_split(perm, max(1, len(X) // int(batch)))
            for idx in idx_batches:
                probs, cache = forward(params, X[idx], act)
                grads = backward(params, cache, probs, Y[idx], act)
                t += 1
                opt(params, grads, t)
            probs_all, _ = forward(params, X, act)
            loss = -np.mean(np.log(probs_all[np.arange(len(y)), y] + 1e-12))
            losses.append(loss)
            accs.append((probs_all.argmax(1) == y).mean())
            if ep == 1 or ep == epochs or ep % every == 0:
                snaps.append((ep, [{k: v.copy() for k, v in p.items()} for p in params]))
            if ep % max(1, epochs // 50) == 0:
                prog.progress(ep / epochs, f"epoch {ep}/{epochs} — loss {loss:.3f}")
        prog.empty()
        st.session_state["run"] = {"X": X, "y": y, "act": act, "n_classes": n_classes,
                                   "losses": losses, "accs": accs, "snaps": snaps,
                                   "cfg": cfg_now}

    run = st.session_state.get("run")
    if run is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for c in range(n_classes):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], s=16, c=STRONG[c], edgecolors="white", linewidths=.5)
        ax.set_title("The data — press Train", fontsize=10)
        st.pyplot(fig, width="stretch")
    else:
        if run["cfg"] != cfg_now:
            st.warning("Settings changed since this run — press **Train** to retrain. "
                       "Showing the previous run below.")
        rX, ry = run["X"], run["y"]
        ract, rnc = run["act"], run["n_classes"]
        losses, accs, snaps = run["losses"], run["accs"], run["snaps"]

        a, b = st.columns(2)
        a.metric("Final loss", f"{losses[-1]:.4f}")
        b.metric("Train accuracy", f"{accs[-1]*100:.1f}%")

        ep_options = [e for e, _ in snaps]
        sel = st.select_slider("⏪ Scrub through training — epoch", options=ep_options,
                               value=ep_options[-1])
        ps = dict(snaps)[sel]

        pad = .4
        gx = np.linspace(rX[:, 0].min() - pad, rX[:, 0].max() + pad, 160)
        gy = np.linspace(rX[:, 1].min() - pad, rX[:, 1].max() + pad, 160)
        XX, YY = np.meshgrid(gx, gy)
        grid = np.c_[XX.ravel(), YY.ravel()]
        P, _ = forward(ps, grid, ract)

        colA, colB = st.columns([1.1, 1])
        with colA:
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.contourf(XX, YY, P.argmax(1).reshape(XX.shape),
                        cmap=DIM, levels=[-.5, .5, 1.5, 2.5])
            for c in range(rnc):
                m = ry == c
                ax.scatter(rX[m, 0], rX[m, 1], s=10, c=STRONG[c],
                           edgecolors="white", linewidths=.4)
            ax.set_title(f"decision boundary at epoch {sel}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            st.pyplot(fig, width="stretch")
        with colB:
            fig2, (axl, axa) = plt.subplots(2, 1, figsize=(4.2, 4.2), sharex=True)
            eps_axis = np.arange(1, len(losses) + 1)
            axl.plot(eps_axis, losses, c=RUST, lw=1.2)
            axl.axvline(sel, c=GOLD, lw=1.2)
            axl.set_title("cross-entropy loss", fontsize=9)
            axa.plot(eps_axis, accs, c=MOSS, lw=1.2)
            axa.axvline(sel, c=GOLD, lw=1.2)
            axa.set_title("train accuracy", fontsize=9)
            axa.set_ylim(0, 1.02); axa.set_xlabel("epoch", fontsize=8)
            for a_ in (axl, axa):
                a_.set_facecolor("white")
            st.pyplot(fig2, width="stretch")

        # -------- exact numbers 1: gradient magnitude per layer
        st.subheader("The exact numbers — how strong is the learning signal in each layer?")
        rY = np.eye(rnc)[ry]
        probs_s, cache_s = forward(ps, rX, ract)
        grads_s = backward(ps, cache_s, probs_s, rY, ract)
        norms = [float(np.linalg.norm(g["W"])) for g in grads_s]
        figg, axg = plt.subplots(figsize=(7, 2.4))
        bars = axg.bar([f"layer {i + 1}" for i in range(len(norms))], norms, color=MOSS)
        axg.bar_label(bars, fmt="%.2e", fontsize=7)
        axg.set_title(f"gradient magnitude ‖∂L/∂W‖ per layer at epoch {sel} "
                      "(layer 1 = closest to the input)", fontsize=9)
        axg.set_facecolor("white")
        st.pyplot(figg, width="stretch")
        st.caption("Backprop passes the error signal from right to left, multiplying by the "
                   "activation's derivative at every layer. Try **sigmoid** with hidden layers "
                   "`8,8,8,8`: the left bars collapse toward zero — the *vanishing gradient* "
                   "problem. Then switch to **relu** and watch them recover.")

        # -------- exact numbers 2: what each hidden neuron sees
        if len(ps) > 1:
            st.subheader("The exact numbers — what each first-layer neuron sees")
            f_act = ACTS[ract][0]
            H = f_act(grid @ ps[0]["W"] + ps[0]["b"])
            h1 = min(H.shape[1], 12)
            ncols = 4
            nrows = int(np.ceil(h1 / ncols))
            fign, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.9 * nrows))
            for j, axn in enumerate(np.array(axes).ravel()):
                if j < h1:
                    axn.contourf(XX, YY, H[:, j].reshape(XX.shape), levels=12, cmap="Greys")
                    axn.set_title(f"unit {j + 1}", fontsize=7)
                else:
                    axn.axis("off")
                axn.set_xticks([]); axn.set_yticks([])
            fign.suptitle(f"first-layer activations at epoch {sel} (dark = active)", fontsize=9)
            st.pyplot(fign, width="stretch")
            if H.shape[1] > 12:
                st.caption(f"showing the first 12 of {H.shape[1]} units")
            st.caption("Each hidden unit is a logistic-regression-like cell: one straight "
                       "ridge. The layers above mix these simple shapes into the curved "
                       "boundary on the left — scrub the epochs to watch the ridges rotate "
                       "into place.")

st.info("Things worth trying in class:  **(1)** spiral with a single tiny layer like `1` — "
        "it can't do it: capacity matters.  **(2)** `relu` vs `tanh` — relu makes visibly "
        "*piecewise-linear* boundaries.  **(3)** SGD vs Adam at the same learning rate — the "
        "loss curves tell the optimizer story from the previous app, now on a real network.  "
        "**(4)** sigmoid with `8,8,8,8` — watch the layer-1 gradient bar vanish.")
quiz("mlp", "With sigmoid activations and 4 hidden layers, the layer-1 gradient bar is ~100× "
            "smaller than the last layer's. What is this?",
     ["The vanishing gradient problem — sigmoid's derivative ≤ 0.25 shrinks the signal at "
      "every layer it passes through",
      "Overfitting",
      "A learning rate that is too small"],
     0, "backprop multiplies by the activation derivative once per layer, so the error signal "
        "decays geometrically on its way back — one big reason modern networks use relu.")

with st.expander("The entire backprop, verbatim from this app"):
    st.code('''def backward(params, cache, probs, y_onehot, act):
    _, df = ACTS[act]
    n = len(y_onehot)
    grads = []
    delta = (probs - y_onehot) / n              # dL/dlogits (softmax + CE)
    for i in reversed(range(len(params))):
        a_prev = cache[i]
        grads.insert(0, {"W": a_prev.T @ delta, "b": delta.sum(0)})
        if i > 0:
            delta = (delta @ params[i]["W"].T) * df(cache[i])
    return grads''', language="python")
