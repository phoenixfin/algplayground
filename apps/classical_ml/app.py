import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse

# stlite ships a stub pyarrow without the classes sklearn's input validation
# probes with isinstance(); give the stub harmless placeholders
import pyarrow
for _n in ("Table", "RecordBatch", "ChunkedArray", "Array"):
    if not hasattr(pyarrow, _n):
        setattr(pyarrow, _n, type(_n, (), {}))

st.set_page_config(page_title="Classical ML — interactive", layout="wide")

INK, MOSS, RUST, GOLD = "#2D2D2D", "#C17A3E", "#A8392A", "#2F7069"
STRONG = [MOSS, RUST, GOLD]
DIM = ListedColormap(["#F0DCC6", "#F3DBD5", "#D3E5E2"])

st.title("Classical machine learning, interactively")
st.caption("Every model here is scikit-learn — the same library you'll use in the labs. "
           "Filled points are the **train set** (70%), hollow points the held-out **test set** (30%); "
           "the gap between the two accuracies is the story to watch. Expand **Show the code** "
           "under each demo to see the 5 lines doing the work.")

model = st.sidebar.radio("Model", [
    "Polynomial regression", "Logistic regression", "k-Nearest Neighbors",
    "Support Vector Machine", "Decision tree", "Gaussian Naive Bayes"])
seed = st.sidebar.number_input("Random seed", 0, 999, 42)
rng = np.random.default_rng(int(seed))


def scatter(ax, X, y, svmask=None, hollow=False):
    for c in np.unique(y):
        m = y == c
        if hollow:
            ax.scatter(X[m, 0], X[m, 1], s=30, facecolors="none",
                       edgecolors=STRONG[int(c)], linewidths=1.2, zorder=3)
        else:
            ax.scatter(X[m, 0], X[m, 1], s=32, c=STRONG[int(c)],
                       edgecolors="white", linewidths=1, zorder=3, label=f"class {int(c)}")
    if svmask is not None:
        ax.scatter(X[svmask, 0], X[svmask, 1], s=110, facecolors="none",
                   edgecolors=INK, linewidths=1.6, zorder=4, label="support vector")


def regions(ax, clf, X):
    x0, x1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    y0, y1 = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 220), np.linspace(y0, y1, 220))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=1.0, cmap=DIM, levels=[-.5, .5, 1.5, 2.5], zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def style(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color("#E4DACB")
    ax.tick_params(colors="#6E6259", labelsize=8)


def probe_sliders(X):
    st.markdown("**Probe point ⭐**")
    qx = st.slider("x₀", float(X[:, 0].min() - .5), float(X[:, 0].max() + .5),
                   float(X[:, 0].mean()))
    qy = st.slider("y₀", float(X[:, 1].min() - .5), float(X[:, 1].max() + .5),
                   float(X[:, 1].mean()))
    return np.array([qx, qy])


def mark(ax, q):
    ax.plot(q[0], q[1], marker="*", ms=17, c=INK, mec="white", mew=1.3, zorder=6)


def use_defaults(d):
    for k, v in d.items():
        st.session_state.setdefault(k, v)


def story_row(model_key, stories):
    # buttons must run before the keyed widgets so a click can set their state
    cols = st.columns(len(stories))
    for col, (label, spec) in zip(cols, stories.items()):
        if col.button(label, key=f"btn_{model_key}_{label}", width="stretch"):
            for k, v in spec["set"].items():
                st.session_state[k] = v
            st.session_state["story"] = (model_key, label, spec["note"])
    story = st.session_state.get("story")
    if story and story[0] == model_key:
        st.success(f"**{story[1]}** — {story[2]}")


def split(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.3, random_state=int(seed), stratify=y)


def quiz(key, question, options, correct, explain):
    with st.expander("🧠 Check yourself"):
        ans = st.radio(question, options, index=None, key=f"quiz_{key}")
        if ans is not None:
            if options.index(ans) == correct:
                st.success("Correct — " + explain)
            else:
                st.error("Not quite — " + explain)


def make_2d(kind, n, noise, n_classes=2, rng=rng):
    from sklearn.datasets import make_moons, make_circles, make_blobs
    rs = int(seed)
    if kind == "moons":
        return make_moons(n_samples=n, noise=noise, random_state=rs)
    if kind == "circles":
        return make_circles(n_samples=n, noise=noise, factor=0.45, random_state=rs)
    if kind == "blobs":
        return make_blobs(n_samples=n, centers=n_classes, cluster_std=1.0 + 4 * noise, random_state=rs)
    # spiral
    per = n // n_classes
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0, 2.4 * np.pi, per) + rng.normal(0, noise * 2, per)
        r = np.linspace(0.1, 1.0, per)
        X.append(np.c_[r * np.cos(t + c * 2 * np.pi / n_classes),
                       r * np.sin(t + c * 2 * np.pi / n_classes)])
        y.append(np.full(per, c))
    return np.vstack(X), np.concatenate(y)


# ---------------------------------------------------------------- regression
if model == "Polynomial regression":
    st.header("Polynomial regression — and the overfitting picture")
    use_defaults({"pr_pat": "linear", "pr_n": 30, "pr_noise": 0.15, "pr_deg": 1})
    story_row("poly", {
        "Underfit": {
            "set": {"pr_pat": "sine", "pr_n": 40, "pr_noise": 0.1, "pr_deg": 1},
            "note": "a straight line through a sine wave. Train AND test MSE are both high — "
                    "the model family is too simple, and no amount of data fixes that."},
        "Sweet spot": {
            "set": {"pr_pat": "sine", "pr_n": 40, "pr_noise": 0.1, "pr_deg": 5},
            "note": "degree 5 is enough to follow the wave without chasing the noise. "
                    "Train and test MSE are both low and close together."},
        "Overfit": {
            "set": {"pr_pat": "sine", "pr_n": 20, "pr_noise": 0.25, "pr_deg": 12},
            "note": "12 degrees for 20 noisy points: train MSE collapses, test MSE explodes. "
                    "Scroll to the coefficient table — the cᵢ have blown up to absurd values."},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        pattern = st.selectbox("True pattern", ["linear", "quadratic", "sine"], key="pr_pat")
        n = st.slider("Points", 10, 120, key="pr_n")
        noise = st.slider("Noise σ", 0.0, 0.5, step=0.01, key="pr_noise")
        deg = st.slider("Polynomial degree", 1, 12, key="pr_deg")
        x0 = st.slider("Probe x₀", 0.0, 1.0, 0.5, 0.01)

    x = rng.uniform(0, 1, n)
    f = {"linear": lambda x: 0.2 + 0.6 * x,
         "quadratic": lambda x: 0.25 + 2.2 * (x - .5) ** 2,
         "sine": lambda x: 0.5 + 0.3 * np.sin(3 * np.pi * x)}[pattern]
    y = f(x) + rng.normal(0, noise, n)
    tr = rng.random(n) < 0.7

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    def fit(d):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression())
        m.fit(x[tr, None], y[tr]); return m

    m = fit(deg)
    xs = np.linspace(0, 1, 300)
    tr_mse = mean_squared_error(y[tr], m.predict(x[tr, None]))
    te_mse = mean_squared_error(y[~tr], m.predict(x[~tr, None])) if (~tr).sum() else np.nan

    with c2:
        a, b = st.columns(2)
        a.metric("Train MSE", f"{tr_mse:.4f}")
        b.metric("Test MSE", f"{te_mse:.4f}")
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
        ax.scatter(x[tr], y[tr], c=RUST, s=26, label="train")
        ax.scatter(x[~tr], y[~tr], facecolors="none", edgecolors=RUST, s=26, label="test")
        ax.plot(xs, m.predict(xs[:, None]), c=MOSS, lw=2.2, label=f"degree {deg}")
        ax.plot(xs, f(xs), c="#AAA", ls="--", lw=1, label="true pattern")
        ax.axvline(x0, c=GOLD, lw=1.2, ls=":")
        ax.plot(x0, m.predict([[x0]])[0], marker="*", ms=15, c=INK,
                mec="white", mew=1.2, zorder=6)
        ax.set_ylim(min(y) - .2, max(y) + .2); ax.legend(fontsize=7); style(ax)
        degs = range(1, 13)
        trs, tes = zip(*[(mean_squared_error(y[tr], fit(d).predict(x[tr, None])),
                          mean_squared_error(y[~tr], fit(d).predict(x[~tr, None])))
                         for d in degs])
        ax2.semilogy(degs, trs, c=MOSS, marker="o", ms=3, label="train MSE")
        ax2.semilogy(degs, tes, c=RUST, marker="o", ms=3, label="test MSE")
        ax2.axvline(deg, c=GOLD, lw=1.5)
        ax2.set_xlabel("degree"); ax2.legend(fontsize=7); style(ax2)
        st.pyplot(fig, width="stretch")
    st.info("Raise the degree: train error only falls, test error turns back up. "
            "The gap between the two curves *is* overfitting.")

    st.subheader("The exact numbers")
    lin = m.named_steps["linearregression"]
    coef = lin.coef_.copy()
    coef[0] += lin.intercept_
    eq = f"{coef[0]:.3f}" + "".join(f" {coef[i]:+.3f}·x^{i}" for i in range(1, deg + 1))
    st.markdown(f"The whole fitted model is these {deg + 1} numbers:  **ŷ = {eq}**")
    pw = np.arange(deg + 1)
    terms = coef * x0 ** pw
    st.dataframe(pd.DataFrame({"power i": pw,
                               "coefficient cᵢ": coef.round(4),
                               "x₀^i": (x0 ** pw).round(4),
                               "term cᵢ·x₀^i": terms.round(4)}), hide_index=True)
    st.markdown(f"Sum of the last column → **ŷ({x0:.2f}) = {terms.sum():.4f}** — "
                f"`model.predict` returns {m.predict([[x0]])[0]:.4f}. Move the probe "
                f"and the degree: high-degree coefficients explode, a symptom of overfitting.")
    quiz("poly", "You raise the polynomial degree; train MSE keeps falling but test MSE turns "
                 "upward. What is happening?",
         ["Overfitting — the model is memorizing noise in the training points",
          "Underfitting — the model is too simple for the data",
          "The data has too few features"],
         0, "past the sweet spot, extra flexibility fits the noise, which helps on the training "
            "points and hurts on unseen ones.")
    with st.expander("Show the code"):
        st.code("""from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)
y_hat = model.predict(x_test.reshape(-1, 1))""")

# ---------------------------------------------------------------- logreg
elif model == "Logistic regression":
    st.header("Logistic regression — a line plus a squash (the bridge to neural nets)")
    use_defaults({"lg_ds": "blobs", "lg_n": 120, "lg_noise": 0.15})
    story_row("logreg", {
        "Clean cut": {
            "set": {"lg_ds": "blobs", "lg_n": 120, "lg_noise": 0.1},
            "note": "linearly separable blobs: THREE numbers (w₁, w₂, b) draw the line, and "
                    "the sigmoid turns signed distance from it into a probability. Slide ⭐ "
                    "across the boundary and watch P pass through 0.5."},
        "The line's limit": {
            "set": {"lg_ds": "moons", "lg_n": 120, "lg_noise": 0.15},
            "note": "moons curl around any straight line — this is the best 3 numbers can do. "
                    "Stack several of these units and you get the MLP playground's network."},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["blobs", "moons", "circles"], key="lg_ds")
        n = st.slider("Points", 30, 300, key="lg_n")
        noise = st.slider("Noise", 0.0, 0.4, step=0.05, key="lg_noise")
    X, y = make_2d(ds, n, noise)
    Xtr, Xte, ytr, yte = split(X, y)
    with c1:
        q = probe_sliders(X)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(Xtr, ytr)
    w1, w2 = clf.coef_[0]
    b0 = clf.intercept_[0]
    z = w1 * q[0] + w2 * q[1] + b0
    p1 = 1 / (1 + np.exp(-z))

    with c2:
        a, b = st.columns(2)
        a.metric("Train accuracy", f"{clf.score(Xtr, ytr)*100:.0f}%")
        b.metric("Test accuracy", f"{clf.score(Xte, yte)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 220), np.linspace(*ax.get_ylim(), 220))
        P = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        cs = ax.contour(xx, yy, P, levels=[0.1, 0.5, 0.9], colors=[INK],
                        linewidths=[.8, 1.8, .8], linestyles=["--", "-", "--"])
        ax.clabel(cs, fmt="P=%.1f", fontsize=7)
        scatter(ax, Xtr, ytr)
        scatter(ax, Xte, yte, hollow=True)
        mark(ax, q); style(ax)
        st.pyplot(fig, width="stretch")
        st.caption("filled = train (the model saw these) · hollow = test (it did not)")
    st.info("The solid line is P = 0.5 — the decision boundary, always straight because "
            "z = w₁x + w₂y + b is linear. The dashed lines are where the model is 90% sure: "
            "the further from the line, the more confident.")

    st.subheader("The exact numbers — the whole model is 3 numbers")
    st.latex(r"P(\text{class }1 \mid x, y) = \sigma(w_1 x + w_2 y + b), \qquad "
             r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.markdown(f"Learned parameters: **w₁ = {w1:+.4f}, w₂ = {w2:+.4f}, b = {b0:+.4f}**")
    st.markdown(f"At ⭐: z = ({w1:+.4f})·({q[0]:.2f}) + ({w2:+.4f})·({q[1]:.2f}) + ({b0:+.4f}) "
                f"= **{z:+.4f}**  \n"
                f"σ({z:+.4f}) = 1 / (1 + e^(−z)) = **{p1:.4f}** → prediction "
                f"**class {int(p1 > 0.5)}** "
                f"(sklearn predict_proba: {np.round(clf.predict_proba(q[None])[0], 4).tolist()})")
    st.markdown("One weighted sum plus one squash is exactly a **neuron**. The MLP playground "
                "stacks layers of these so the combined boundary can bend.")
    quiz("logreg", "Why can logistic regression only draw straight decision boundaries?",
         ["Because the sigmoid is a linear function",
          "Because the boundary is where w₁x + w₂y + b = 0, which is a line",
          "Because it only works on 2 classes"],
         1, "σ is curved but monotone, so P = 0.5 exactly where z = 0 — and z = 0 is a straight line.")
    with st.expander("Show the code"):
        st.code("""from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train, y_train)
clf.coef_, clf.intercept_   # the whole model: w1, w2, b
clf.predict_proba(X_new)""")

# ---------------------------------------------------------------- knn
elif model == "k-Nearest Neighbors":
    st.header("k-NN — the neighbors vote")
    use_defaults({"knn_ds": "moons", "knn_n": 120, "knn_noise": 0.2,
                  "knn_k": 3, "knn_w": "uniform"})
    story_row("knn", {
        "Memorize the noise": {
            "set": {"knn_ds": "moons", "knn_n": 120, "knn_noise": 0.35,
                    "knn_k": 1, "knn_w": "uniform"},
            "note": "k=1 on noisy data: every training point gets its own little island, so "
                    "train accuracy is 100% *by construction* — and test accuracy is clearly "
                    "lower. Memorizing is not learning."},
        "Let neighbors vote": {
            "set": {"knn_ds": "moons", "knn_n": 120, "knn_noise": 0.35,
                    "knn_k": 15, "knn_w": "uniform"},
            "note": "same noisy data, k=15: the islands melt away and the train/test gap "
                    "closes. Averaging over neighbors averages the noise out."},
        "Too smooth": {
            "set": {"knn_ds": "spiral", "knn_n": 120, "knn_noise": 0.1,
                    "knn_k": 31, "knn_w": "uniform"},
            "note": "k=31 on a spiral: the vote now reaches across arms and washes out real "
                    "structure — test accuracy drops again. k is a dial between two failure modes."},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["moons", "circles", "blobs", "spiral"], key="knn_ds")
        n = st.slider("Points", 30, 300, key="knn_n")
        noise = st.slider("Noise", 0.0, 0.4, step=0.05, key="knn_noise")
        k = st.slider("k (neighbors)", 1, 31, step=2, key="knn_k")
        weights = st.selectbox("Vote weighting", ["uniform", "distance"], key="knn_w")
    X, y = make_2d(ds, n, noise, n_classes=3 if ds in ("blobs", "spiral") else 2)
    Xtr, Xte, ytr, yte = split(X, y)
    with c1:
        q = probe_sliders(X)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    k_eff = min(k, max(1, int(len(Xtr) * 0.8) - 1))
    clf = KNeighborsClassifier(n_neighbors=k_eff, weights=weights).fit(Xtr, ytr)
    cv = cross_val_score(clf, Xtr, ytr, cv=5).mean()
    if k_eff < k:
        st.caption(f"k capped at {k_eff}: only {len(Xtr)} training points after the 70/30 split.")
    dist, idx = clf.kneighbors(q[None])
    dist, idx = dist[0], idx[0]
    w = np.ones_like(dist) if weights == "uniform" else 1.0 / np.maximum(dist, 1e-9)

    with c2:
        a, b, c3 = st.columns(3)
        a.metric("Train accuracy", f"{clf.score(Xtr, ytr)*100:.1f}%")
        b.metric("Test accuracy", f"{clf.score(Xte, yte)*100:.1f}%")
        c3.metric("5-fold CV (on train)", f"{cv*100:.1f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        scatter(ax, Xtr, ytr)
        scatter(ax, Xte, yte, hollow=True)
        style(ax)
        for i in idx:
            ax.plot([q[0], Xtr[i, 0]], [q[1], Xtr[i, 1]], c=INK, lw=.8, alpha=.55, zorder=5)
        mark(ax, q)
        ax.set_title(f"k={k_eff}, weights={weights}", fontsize=9, color=INK)
        st.pyplot(fig, width="stretch")
        st.caption("filled = train (the model saw these) · hollow = test (it did not)")
    st.info("k=1 memorises every noise point (jagged islands). Large k smooths the boundary "
            "but can wash out real structure. Cross-validation is how we pick k honestly.")

    st.subheader("The exact numbers — the vote at ⭐")
    st.dataframe(pd.DataFrame({"rank": np.arange(1, len(idx) + 1),
                               "neighbor #": idx,
                               "x": Xtr[idx, 0].round(3), "y": Xtr[idx, 1].round(3),
                               "distance to ⭐": dist.round(4),
                               "class": ytr[idx].astype(int),
                               "vote weight": w.round(4)}), hide_index=True)
    tally = {int(c): float(w[ytr[idx] == c].sum()) for c in np.unique(ytr[idx])}
    st.markdown("distance = √((x−x₀)² + (y−y₀)²) ; weight = 1 if uniform, 1/distance otherwise.  \n"
                "Vote totals — " + " · ".join(f"class {c}: **{v:.4f}**" for c, v in tally.items())
                + f" → prediction **class {int(clf.predict(q[None])[0])}** "
                f"(predict_proba = {np.round(clf.predict_proba(q[None])[0], 3).tolist()})")
    quiz("knn", "With k = 1, train accuracy is always 100%. Why?",
         ["Because each training point's nearest neighbor is itself",
          "Because k-NN always generalizes perfectly",
          "Because the classes are balanced"],
         0, "when predicting a training point, its own copy sits at distance 0 — so k=1 'learns' "
            "every point, noise included. The test accuracy tells the real story.")
    with st.expander("Show the code"):
        st.code("""from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=k, weights="uniform")
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))""")

# ---------------------------------------------------------------- svm
elif model == "Support Vector Machine":
    st.header("SVM — the widest street")
    use_defaults({"svm_ds": "moons", "svm_n": 90, "svm_noise": 0.15,
                  "svm_kern": "rbf", "svm_C": 1.0, "svm_g": 1.0})
    story_row("svm", {
        "Widest street": {
            "set": {"svm_ds": "blobs", "svm_n": 90, "svm_noise": 0.15,
                    "svm_kern": "linear", "svm_C": 0.01},
            "note": "a soft linear SVM on separable blobs: the street is as wide as the data "
                    "allows, and lots of points become support vectors. Check w and the street "
                    "width in the numbers below."},
        "Memorize the noise": {
            "set": {"svm_ds": "moons", "svm_n": 90, "svm_noise": 0.3,
                    "svm_kern": "rbf", "svm_C": 100.0, "svm_g": 20.0},
            "note": "huge C (no forgiveness) and huge γ (hyper-local influence): the boundary "
                    "draws an island around every noise point. Train accuracy soars, test "
                    "accuracy tells the truth."},
        "Wrong tool": {
            "set": {"svm_ds": "circles", "svm_n": 90, "svm_noise": 0.15,
                    "svm_kern": "linear", "svm_C": 1.0},
            "note": "a straight line cannot separate concentric rings — train AND test accuracy "
                    "sit near chance, and no C will fix it. Switch the kernel to rbf and watch "
                    "the same data become easy."},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["moons", "circles", "blobs"], key="svm_ds")
        n = st.slider("Points", 30, 200, key="svm_n")
        noise = st.slider("Noise", 0.0, 0.4, step=0.05, key="svm_noise")
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"], key="svm_kern")
        C = st.select_slider("C (margin softness)", [0.01, 0.1, 1.0, 10.0, 100.0], key="svm_C")
        gamma = st.select_slider("γ (RBF width)", [0.1, 0.5, 1.0, 5.0, 20.0], key="svm_g") \
            if kernel == "rbf" else "scale"
    X, y = make_2d(ds, n, noise)
    Xtr, Xte, ytr, yte = split(X, y)
    with c1:
        q = probe_sliders(X)

    from sklearn.svm import SVC
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=3).fit(Xtr, ytr)
    svmask = np.zeros(len(Xtr), bool); svmask[clf.support_] = True
    g = getattr(clf, "_gamma",
                1.0 / (Xtr.shape[1] * Xtr.var()) if gamma == "scale" else float(gamma))
    sv = clf.support_vectors_
    if kernel == "linear":
        kv = sv @ q
    elif kernel == "poly":
        kv = (g * (sv @ q) + clf.coef0) ** 3
    else:
        kv = np.exp(-g * ((sv - q) ** 2).sum(1))
    alpha_y = clf.dual_coef_[0]
    contrib = alpha_y * kv
    f_q = contrib.sum() + clf.intercept_[0]

    with c2:
        a, b, c3 = st.columns(3)
        a.metric("Support vectors", f"{len(clf.support_)} / {len(Xtr)}")
        b.metric("Train accuracy", f"{clf.score(Xtr, ytr)*100:.0f}%")
        c3.metric("Test accuracy", f"{clf.score(Xte, yte)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        # margin contours
        xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 220), np.linspace(*ax.get_ylim(), 220))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[INK], linewidths=[.8, 1.8, .8],
                   linestyles=["--", "-", "--"])
        scatter(ax, Xtr, ytr, svmask)
        scatter(ax, Xte, yte, hollow=True)
        mark(ax, q); style(ax)
        st.pyplot(fig, width="stretch")
        st.caption("filled = train (the model saw these) · hollow = test (it did not)")
    st.info("Solid line = boundary, dashed = the margin. Only ringed points (support vectors) "
            "define it. Lower C → wider, more forgiving street; higher γ → each support "
            "vector's influence becomes more local, boundary gets wigglier.")

    st.subheader("The exact numbers — f(⭐) is a sum over support vectors")
    st.latex(r"f(x)=\sum_{i\in SV}\alpha_i y_i\,K(x_i,x)+b")
    order = np.argsort(-np.abs(contrib))[:8]
    st.dataframe(pd.DataFrame({"support vector #": clf.support_[order],
                               "x": sv[order, 0].round(3), "y": sv[order, 1].round(3),
                               "class": ytr[clf.support_][order].astype(int),
                               "αᵢyᵢ": alpha_y[order].round(4),
                               "K(xᵢ, ⭐)": kv[order].round(4),
                               "αᵢyᵢ·K": contrib[order].round(4)}), hide_index=True)
    if len(contrib) > 8:
        st.caption(f"the 8 largest of {len(contrib)} support-vector terms — γ used: {g:.4f}")
    st.markdown(f"Σ terms = {contrib.sum():+.4f}, plus b = {clf.intercept_[0]:+.4f} → "
                f"**f(⭐) = {f_q:+.4f}** → sign says **class {int(f_q > 0)}** "
                f"(sklearn decision_function: {clf.decision_function(q[None])[0]:+.4f}). "
                f"|f| < 1 means ⭐ sits inside the margin street.")
    if kernel == "linear":
        w_vec = alpha_y @ sv
        st.markdown(f"Linear kernel → an explicit weight vector exists: "
                    f"**w = ({w_vec[0]:+.3f}, {w_vec[1]:+.3f})**, "
                    f"street width 2/‖w‖ = {2 / np.linalg.norm(w_vec):.3f}")
    quiz("svm", "Which training points actually define the SVM's decision boundary?",
         ["All of them equally",
          "Only the support vectors — the points on or inside the margin",
          "Only the misclassified points"],
         1, "every other point could be deleted and the boundary would not move — that is what "
            "the α table above shows: everyone else has α = 0.")
    with st.expander("Show the code"):
        st.code("""from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=1.0, gamma=1.0)
clf.fit(X_train, y_train)
print(clf.support_)   # indices of the support vectors""")

# ---------------------------------------------------------------- tree
elif model == "Decision tree":
    st.header("Decision tree — split the plane with yes/no questions")
    use_defaults({"tr_ds": "blobs", "tr_n": 120, "tr_noise": 0.15,
                  "tr_depth": 3, "tr_crit": "gini"})
    story_row("tree", {
        "One question": {
            "set": {"tr_ds": "blobs", "tr_n": 120, "tr_noise": 0.15, "tr_depth": 1},
            "note": "depth 1 = a single yes/no question = one straight cut through the plane. "
                    "The whole model is one threshold — read it in the tree diagram."},
        "Just right": {
            "set": {"tr_ds": "moons", "tr_n": 120, "tr_noise": 0.2, "tr_depth": 3},
            "note": "a few questions approximate the curve with a coarse staircase. Train and "
                    "test accuracy are close — the tree learned structure, not noise."},
        "Memorize the noise": {
            "set": {"tr_ds": "moons", "tr_n": 120, "tr_noise": 0.3, "tr_depth": 12},
            "note": "unlimited depth on noisy data: the staircase grows a rectangle around "
                    "every stray point, leaves ≈ points, train accuracy 100% — and the test "
                    "accuracy gap is the price."},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["blobs", "moons", "circles", "spiral"], key="tr_ds")
        n = st.slider("Points", 30, 300, key="tr_n")
        noise = st.slider("Noise", 0.0, 0.4, step=0.05, key="tr_noise")
        depth = st.slider("Max depth", 1, 12, key="tr_depth")
        crit = st.selectbox("Criterion", ["gini", "entropy"], key="tr_crit")
    X, y = make_2d(ds, n, noise, n_classes=3 if ds in ("blobs", "spiral") else 2)
    Xtr, Xte, ytr, yte = split(X, y)
    with c1:
        q = probe_sliders(X)

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    clf = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=0).fit(Xtr, ytr)

    with c2:
        a, b, c3 = st.columns(3)
        a.metric("Leaves", clf.get_n_leaves())
        b.metric("Train accuracy", f"{clf.score(Xtr, ytr)*100:.0f}%")
        c3.metric("Test accuracy", f"{clf.score(Xte, yte)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        scatter(ax, Xtr, ytr)
        scatter(ax, Xte, yte, hollow=True)
        mark(ax, q); style(ax)
        st.pyplot(fig, width="stretch")
        st.caption("filled = train (the model saw these) · hollow = test (it did not)")
        fig2, ax2 = plt.subplots(figsize=(9, 3.2))
        plot_tree(clf, ax=ax2, filled=True, impurity=True, fontsize=6,
                  feature_names=["x", "y"])
        st.pyplot(fig2, width="stretch")
    st.info("All boundaries are axis-aligned rectangles — the tree can only ask "
            "'is x ≤ t?'. Watch it build staircases on curvy data, and memorise noise "
            "when max depth is large.")

    st.subheader("The exact numbers — ⭐'s walk down the tree")
    t = clf.tree_
    node, steps = 0, []
    while t.children_left[node] != -1:
        fidx, thr = int(t.feature[node]), float(t.threshold[node])
        name, val = ("x", q[0]) if fidx == 0 else ("y", q[1])
        goes_left = val <= thr
        steps.append(f"{len(steps) + 1}. node {node}: is {name} = {val:.3f} ≤ {thr:.3f}? "
                     f"**{'yes → go left' if goes_left else 'no → go right'}**")
        node = int(t.children_left[node]) if goes_left else int(t.children_right[node])
    st.markdown("  \n".join(steps) if steps else "The tree is a single leaf — no questions asked.")
    st.markdown(f"Reached **leaf {node}**: {int(t.n_node_samples[node])} training points landed here, "
                f"{crit} impurity = {t.impurity[node]:.3f}, class mix = "
                f"{np.round(t.value[node][0], 3).tolist()} → "
                f"P(class) = {np.round(clf.predict_proba(q[None])[0], 3).tolist()} → "
                f"prediction **class {int(clf.predict(q[None])[0])}**")
    quiz("tree", "Why do decision-tree boundaries look like staircases on curved data?",
         ["Because trees can only ask one-feature threshold questions like 'is x ≤ t?'",
          "Because the data is noisy",
          "Because gini impurity is a step function"],
         0, "each split is an axis-aligned cut, so a curve can only be approximated by stacking "
            "rectangles — more depth just makes the stairs finer.")
    with st.expander("Show the code"):
        st.code("""from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
clf.fit(X_train, y_train)
plot_tree(clf, filled=True)""")

# ---------------------------------------------------------------- nb
else:
    st.header("Gaussian Naive Bayes — a bell curve per class, then Bayes' rule")
    use_defaults({"nb_ds": "blobs", "nb_n": 120, "nb_noise": 0.15, "nb_k": 2})
    story_row("nb", {
        "Assumptions hold": {
            "set": {"nb_ds": "blobs", "nb_n": 120, "nb_noise": 0.15, "nb_k": 2},
            "note": "Gaussian blobs are literally the model's world view — one bell curve per "
                    "class. Train and test accuracy match, with only a handful of learned numbers."},
        "Assumptions break": {
            "set": {"nb_ds": "spiral", "nb_n": 120, "nb_noise": 0.15, "nb_k": 2},
            "note": "two intertwined spiral arms share the same mean AND nearly the same "
                    "variance, so the two fitted Gaussians are almost identical — look at the "
                    "overlapping ellipses and the near-50/50 posteriors below. A model is only "
                    "as good as its assumptions. (Try *circles* too: NB survives there because "
                    "the class variances still differ.)"},
    })
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["blobs", "moons", "circles", "spiral"], key="nb_ds")
        n = st.slider("Points", 30, 300, key="nb_n")
        noise = st.slider("Noise", 0.0, 0.4, step=0.05, key="nb_noise")
        n_classes = st.slider("Classes (blobs only)", 2, 3, key="nb_k")
    X, y = make_2d(ds, n, noise, n_classes=n_classes)
    Xtr, Xte, ytr, yte = split(X, y)
    with c1:
        q = probe_sliders(X)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB().fit(Xtr, ytr)

    with c2:
        a, b = st.columns(2)
        a.metric("Train accuracy", f"{clf.score(Xtr, ytr)*100:.0f}%")
        b.metric("Test accuracy", f"{clf.score(Xte, yte)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        scatter(ax, Xtr, ytr)
        scatter(ax, Xte, yte, hollow=True)
        for c in range(len(clf.classes_)):
            mx, my = clf.theta_[c]
            sx, sy = np.sqrt(clf.var_[c])
            for mfac, ls in [(1, "-"), (2, "--")]:
                ax.add_patch(Ellipse((mx, my), 2 * mfac * sx, 2 * mfac * sy,
                                     fill=False, color=STRONG[c], lw=1.4, ls=ls, zorder=5))
            ax.plot(mx, my, "+", c=INK, ms=10, mew=2, zorder=6)
        mark(ax, q)
        style(ax)
        st.pyplot(fig, width="stretch")
        st.caption("filled = train (the model saw these) · hollow = test (it did not)")
        st.write("**The entire learned model:**")
        st.write({f"class {int(c)}": {"prior": round(float(p), 3),
                                      "mean": np.round(clf.theta_[i], 2).tolist(),
                                      "var": np.round(clf.var_[i], 3).tolist()}
                  for i, (c, p) in enumerate(zip(clf.classes_, clf.class_prior_))})
    st.info("Ellipses are 1σ (solid) and 2σ (dashed) of the fitted Gaussians — always "
            "axis-aligned, because features are assumed independent. The whole model is "
            "the handful of numbers printed above.")

    st.subheader("The exact numbers — Bayes' rule at ⭐")
    st.latex(r"P(c\mid\star)\;\propto\;P(c)\cdot"
             r"\mathcal N(x_0;\,\theta_{c,x},\sigma^2_{c,x})\cdot"
             r"\mathcal N(y_0;\,\theta_{c,y},\sigma^2_{c,y})")
    dens = np.exp(-(q - clf.theta_) ** 2 / (2 * clf.var_)) / np.sqrt(2 * np.pi * clf.var_)
    joint = clf.class_prior_ * dens.prod(axis=1)
    post = joint / joint.sum()
    st.dataframe(pd.DataFrame({"class": clf.classes_.astype(int),
                               "prior P(c)": clf.class_prior_.round(4),
                               "density at x₀": dens[:, 0].round(5),
                               "density at y₀": dens[:, 1].round(5),
                               "prior × densities": joint.round(6),
                               "posterior P(c|⭐)": post.round(4)}), hide_index=True)
    st.markdown(f"argmax of the last column → **class {int(clf.classes_[post.argmax()])}** "
                f"(sklearn predict_proba: {np.round(clf.predict_proba(q[None])[0], 4).tolist()}). "
                f"Each density is just the bell-curve formula evaluated with the θ and σ² above.")
    quiz("nb", "Gaussian Naive Bayes is nearly blind on the 2-class spiral. Why?",
         ["The spiral has too many points",
          "Both classes have almost the same mean and variance, so the fitted Gaussians are "
          "nearly identical",
          "Bayes' rule does not apply to spirals"],
         1, "the model only sees per-class means and variances — and the spiral arms share both. "
            "Nearly identical Gaussians → posteriors stuck near 50/50.")
    with st.expander("Show the code"):
        st.code("""from sklearn.naive_bayes import GaussianNB

clf = GaussianNB().fit(X_train, y_train)
clf.theta_        # per-class feature means
clf.var_          # per-class feature variances
clf.class_prior_  # P(class)""")
