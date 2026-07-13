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

INK, MOSS, RUST, GOLD = "#1E2A22", "#2F6F4E", "#A6472F", "#B8892B"
STRONG = [MOSS, RUST, GOLD]
DIM = ListedColormap(["#C7DBCB", "#EDD6CD", "#EFE0BC"])

st.title("Classical machine learning, interactively")
st.caption("Every model here is scikit-learn — the same library you'll use in the labs. "
           "Expand **Show the code** under each demo to see the 5 lines doing the work.")

model = st.sidebar.radio("Model", [
    "Polynomial regression", "k-Nearest Neighbors", "Support Vector Machine",
    "Decision tree", "Gaussian Naive Bayes"])
seed = st.sidebar.number_input("Random seed", 0, 999, 42)
rng = np.random.default_rng(int(seed))


def scatter(ax, X, y, svmask=None):
    for c in np.unique(y):
        m = y == c
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
        s.set_color("#CFCDBB")
    ax.tick_params(colors="#4A5A4E", labelsize=8)


def probe_sliders(X):
    st.markdown("**Probe point ⭐**")
    qx = st.slider("x₀", float(X[:, 0].min() - .5), float(X[:, 0].max() + .5),
                   float(X[:, 0].mean()))
    qy = st.slider("y₀", float(X[:, 1].min() - .5), float(X[:, 1].max() + .5),
                   float(X[:, 1].mean()))
    return np.array([qx, qy])


def mark(ax, q):
    ax.plot(q[0], q[1], marker="*", ms=17, c=INK, mec="white", mew=1.3, zorder=6)


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
    c1, c2 = st.columns([1, 2])
    with c1:
        pattern = st.selectbox("True pattern", ["linear", "quadratic", "sine"])
        n = st.slider("Points", 10, 120, 30)
        noise = st.slider("Noise σ", 0.0, 0.5, 0.15, 0.01)
        deg = st.slider("Polynomial degree", 1, 12, 1)
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
    with st.expander("Show the code"):
        st.code("""from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(x_train.reshape(-1, 1), y_train)
y_hat = model.predict(x_test.reshape(-1, 1))""")

# ---------------------------------------------------------------- knn
elif model == "k-Nearest Neighbors":
    st.header("k-NN — the neighbors vote")
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["moons", "circles", "blobs", "spiral"])
        n = st.slider("Points", 30, 300, 120)
        noise = st.slider("Noise", 0.0, 0.4, 0.2, 0.05)
        k = st.slider("k (neighbors)", 1, 31, 3, 2)
        weights = st.selectbox("Vote weighting", ["uniform", "distance"])
    X, y = make_2d(ds, n, noise, n_classes=3 if ds in ("blobs", "spiral") else 2)
    with c1:
        q = probe_sliders(X)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    clf = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(X, y)
    cv = cross_val_score(clf, X, y, cv=5).mean()
    dist, idx = clf.kneighbors(q[None])
    dist, idx = dist[0], idx[0]
    w = np.ones_like(dist) if weights == "uniform" else 1.0 / np.maximum(dist, 1e-9)

    with c2:
        st.metric("5-fold cross-validation accuracy", f"{cv*100:.1f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X); scatter(ax, X, y); style(ax)
        for i in idx:
            ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], c=INK, lw=.8, alpha=.55, zorder=5)
        mark(ax, q)
        ax.set_title(f"k={k}, weights={weights}", fontsize=9, color=INK)
        st.pyplot(fig, width="stretch")
    st.info("k=1 memorises every noise point (jagged islands). Large k smooths the boundary "
            "but can wash out real structure. Cross-validation is how we pick k honestly.")

    st.subheader("The exact numbers — the vote at ⭐")
    st.dataframe(pd.DataFrame({"rank": np.arange(1, len(idx) + 1),
                               "neighbor #": idx,
                               "x": X[idx, 0].round(3), "y": X[idx, 1].round(3),
                               "distance to ⭐": dist.round(4),
                               "class": y[idx].astype(int),
                               "vote weight": w.round(4)}), hide_index=True)
    tally = {int(c): float(w[y[idx] == c].sum()) for c in np.unique(y[idx])}
    st.markdown("distance = √((x−x₀)² + (y−y₀)²) ; weight = 1 if uniform, 1/distance otherwise.  \n"
                "Vote totals — " + " · ".join(f"class {c}: **{v:.4f}**" for c, v in tally.items())
                + f" → prediction **class {int(clf.predict(q[None])[0])}** "
                f"(predict_proba = {np.round(clf.predict_proba(q[None])[0], 3).tolist()})")
    with st.expander("Show the code"):
        st.code("""from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=k, weights="uniform")
clf.fit(X, y)
pred = clf.predict(X_new)""")

# ---------------------------------------------------------------- svm
elif model == "Support Vector Machine":
    st.header("SVM — the widest street")
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["moons", "circles", "blobs"])
        n = st.slider("Points", 30, 200, 90)
        noise = st.slider("Noise", 0.0, 0.4, 0.15, 0.05)
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
        C = st.select_slider("C (margin softness)", [0.01, 0.1, 1.0, 10.0, 100.0], 1.0)
        gamma = st.select_slider("γ (RBF width)", [0.1, 0.5, 1.0, 5.0, 20.0], 1.0) \
            if kernel == "rbf" else "scale"
    X, y = make_2d(ds, n, noise)
    with c1:
        q = probe_sliders(X)

    from sklearn.svm import SVC
    clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=3).fit(X, y)
    svmask = np.zeros(len(X), bool); svmask[clf.support_] = True
    g = getattr(clf, "_gamma",
                1.0 / (X.shape[1] * X.var()) if gamma == "scale" else float(gamma))
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
        a, b = st.columns(2)
        a.metric("Support vectors", f"{len(clf.support_)} / {n}")
        b.metric("Train accuracy", f"{clf.score(X, y)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X)
        # margin contours
        xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 220), np.linspace(*ax.get_ylim(), 220))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[INK], linewidths=[.8, 1.8, .8],
                   linestyles=["--", "-", "--"])
        scatter(ax, X, y, svmask); mark(ax, q); style(ax)
        st.pyplot(fig, width="stretch")
    st.info("Solid line = boundary, dashed = the margin. Only ringed points (support vectors) "
            "define it. Lower C → wider, more forgiving street; higher γ → each support "
            "vector's influence becomes more local, boundary gets wigglier.")

    st.subheader("The exact numbers — f(⭐) is a sum over support vectors")
    st.latex(r"f(x)=\sum_{i\in SV}\alpha_i y_i\,K(x_i,x)+b")
    order = np.argsort(-np.abs(contrib))[:8]
    st.dataframe(pd.DataFrame({"support vector #": clf.support_[order],
                               "x": sv[order, 0].round(3), "y": sv[order, 1].round(3),
                               "class": y[clf.support_][order].astype(int),
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
    with st.expander("Show the code"):
        st.code("""from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=1.0, gamma=1.0)
clf.fit(X, y)
print(clf.support_)   # indices of the support vectors""")

# ---------------------------------------------------------------- tree
elif model == "Decision tree":
    st.header("Decision tree — split the plane with yes/no questions")
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["blobs", "moons", "circles", "spiral"])
        n = st.slider("Points", 30, 300, 120)
        noise = st.slider("Noise", 0.0, 0.4, 0.15, 0.05)
        depth = st.slider("Max depth", 1, 12, 3)
        crit = st.selectbox("Criterion", ["gini", "entropy"])
    X, y = make_2d(ds, n, noise, n_classes=3 if ds in ("blobs", "spiral") else 2)
    with c1:
        q = probe_sliders(X)

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    clf = DecisionTreeClassifier(max_depth=depth, criterion=crit, random_state=0).fit(X, y)

    with c2:
        a, b = st.columns(2)
        a.metric("Leaves", clf.get_n_leaves())
        b.metric("Train accuracy", f"{clf.score(X, y)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X); scatter(ax, X, y); mark(ax, q); style(ax)
        st.pyplot(fig, width="stretch")
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
    with st.expander("Show the code"):
        st.code("""from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
clf.fit(X, y)
plot_tree(clf, filled=True)""")

# ---------------------------------------------------------------- nb
else:
    st.header("Gaussian Naive Bayes — a bell curve per class, then Bayes' rule")
    c1, c2 = st.columns([1, 2])
    with c1:
        ds = st.selectbox("Dataset", ["blobs", "moons", "circles"])
        n = st.slider("Points", 30, 300, 120)
        noise = st.slider("Noise", 0.0, 0.4, 0.15, 0.05)
        n_classes = st.slider("Classes (blobs only)", 2, 3, 2)
    X, y = make_2d(ds, n, noise, n_classes=n_classes)
    with c1:
        q = probe_sliders(X)

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB().fit(X, y)

    with c2:
        st.metric("Train accuracy", f"{clf.score(X, y)*100:.0f}%")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        regions(ax, clf, X); scatter(ax, X, y)
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
    with st.expander("Show the code"):
        st.code("""from sklearn.naive_bayes import GaussianNB

clf = GaussianNB().fit(X, y)
clf.theta_        # per-class feature means
clf.var_          # per-class feature variances
clf.class_prior_  # P(class)""")
