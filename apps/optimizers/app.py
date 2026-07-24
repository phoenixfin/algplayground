import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Gradient descent & optimizers", layout="wide")

INK, MOSS, RUST, GOLD, SLATE = "#2D2D2D", "#C17A3E", "#A8392A", "#2F7069", "#4A6B8A"
OPT_COLORS = {"GD": MOSS, "SGD (noisy)": SLATE, "Momentum": RUST, "RMSProp": GOLD, "Adam": INK}

st.title("Gradient descent & the optimizer zoo")
st.caption("The single idea behind training every neural network: follow the negative "
           "gradient downhill. Everything else — momentum, RMSProp, Adam — is a patch on "
           "that idea's failure modes.")


def quiz(key, question, options, correct, explain):
    with st.expander("🧠 Check yourself"):
        ans = st.radio(question, options, index=None, key=f"quiz_{key}")
        if ans is not None:
            if options.index(ans) == correct:
                st.success("Correct — " + explain)
            else:
                st.error("Not quite — " + explain)


tab1, tab2 = st.tabs(["1D intuition — the learning rate", "2D surfaces — optimizer race"])

# ---------------------------------------------------------------- 1D
with tab1:
    c1, c2 = st.columns([1, 2])
    with c1:
        lr = st.slider("Learning rate η", 0.01, 1.2, 0.1, 0.01)
        steps = st.slider("Steps", 1, 40, 15)
        x0 = st.slider("Start x₀", -2.0, 2.0, 1.8, 0.1)
    # f(x) = x^2, f'(x) = 2x ; diverges when |1-2η| > 1, i.e. η > 1
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - lr * 2 * xs[-1])
    xs = np.array(xs)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        g = np.linspace(-2.2, 2.2, 200)
        ax.plot(g, g ** 2, c="#E4DACB", lw=2)
        ax.plot(xs, xs ** 2, c=RUST, marker="o", ms=4, lw=1.2)
        for i in range(min(len(xs) - 1, 12)):
            ax.annotate("", xy=(xs[i + 1], xs[i + 1] ** 2), xytext=(xs[i], xs[i] ** 2),
                        arrowprops=dict(arrowstyle="->", color=RUST, lw=1))
        ax.set_ylim(-0.2, 5); ax.set_title(f"x ← x − η·f′(x)   on f(x)=x²,   η={lr}", fontsize=10)
        ax.set_facecolor("white")
        st.pyplot(fig, width="stretch")
        final = abs(xs[-1])
        if final > abs(x0) * 1.5:
            st.error(f"**Diverging** — |x| grew to {final:.2f}. On f(x)=x², any η > 1 overshoots "
                     "so badly each step lands farther away than it started.")
        elif lr > 0.5:
            st.warning("Overshooting the minimum each step and bouncing side to side — "
                       "converging, but wastefully.")
        else:
            st.success(f"Converging smoothly. After {steps} steps, x = {xs[-1]:.4f}.")

    st.subheader("The exact numbers — the update rule, line by line")
    rows = int(min(len(xs) - 1, 10))
    st.dataframe(pd.DataFrame({
        "t": np.arange(rows),
        "xₜ": np.round(xs[:rows], 5),
        "gradient f′(xₜ) = 2xₜ": np.round(2 * xs[:rows], 5),
        "step = −η·f′(xₜ)": np.round(-lr * 2 * xs[:rows], 5),
        "xₜ₊₁ = xₜ + step": np.round(xs[1:rows + 1], 5)}), hide_index=True)
    st.caption(f"the first {rows} of {len(xs) - 1} steps — every row is the same one-line rule. "
               "Change η above and watch the step column flip sign or explode.")
    st.info("This one slider is the most important hyperparameter in deep learning. "
            "Too small: crawling. Too large: bouncing or exploding. There is no universally "
            "correct value — it depends on the curvature of the loss, which is exactly what "
            "the adaptive optimizers in the next tab try to handle automatically.")
    quiz("lr1d", "On f(x) = x², gradient descent diverges once η > 1. Why exactly?",
         ["The update x ← x − η·2x multiplies x by (1 − 2η), and |1 − 2η| > 1 when η > 1",
          "The gradient becomes infinite",
          "Rounding error accumulates"],
         0, "each step scales x by the factor (1 − 2η) — check it in the table above. A factor "
            "bigger than 1 in magnitude means every step lands farther out than the last.")

# ---------------------------------------------------------------- 2D
SURFACES = {
    "Ravine (ill-conditioned bowl)": {
        "f": lambda x, y: 0.5 * x ** 2 + 10 * y ** 2,
        "g": lambda x, y: (x, 20 * y),
        "xlim": (-2.2, 2.2), "ylim": (-1.2, 1.2), "start": (-2.0, 1.0), "opt": (0, 0),
        "note": "Steep across, shallow along. Plain GD zig-zags across the walls; "
                "momentum averages the zig-zags out; Adam rescales each axis."},
    "Rosenbrock valley": {
        "f": lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
        "g": lambda x, y: (-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)),
        "xlim": (-1.6, 1.6), "ylim": (-0.6, 1.6), "start": (-1.3, 1.3), "opt": (1, 1),
        "note": "The classic curved valley. Getting into the valley is easy; crawling "
                "along its flat floor to (1,1) is what separates the optimizers."},
    "Saddle point": {
        "f": lambda x, y: x ** 2 - y ** 2,
        "g": lambda x, y: (2 * x, -2 * y),
        "xlim": (-1.5, 1.5), "ylim": (-1.5, 1.5), "start": (-1.2, 1e-4), "opt": None,
        "note": "A maximum in y, minimum in x. The gradient at the saddle is ~0, so plain GD "
                "stalls there. Noise (SGD) or momentum provides the nudge to escape downhill."},
    "Two minima (Himmelblau-lite)": {
        "f": lambda x, y: (x ** 2 + y - 2) ** 2 + (x + y ** 2 - 2) ** 2,
        "g": lambda x, y: (4 * x * (x ** 2 + y - 2) + 2 * (x + y ** 2 - 2),
                           2 * (x ** 2 + y - 2) + 4 * y * (x + y ** 2 - 2)),
        "xlim": (-3, 3), "ylim": (-3, 3), "start": (0.1, -0.1), "opt": None,
        "note": "Multiple minima: which one you land in depends on the start point and the "
                "optimizer's dynamics. Gradient descent is a *local* method."},
}


def run_opt(name, grad_fn, start, lr, steps, noise, rng, trace=None):
    p = np.array(start, dtype=float)
    m = np.zeros(2); v = np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = np.array(grad_fn(*p))
        if name == "SGD (noisy)":
            g = g + rng.normal(0, noise * (np.linalg.norm(g) + 1.0), 2)
        prev = p.copy()
        if name in ("GD", "SGD (noisy)"):
            p = p - lr * g
        elif name == "Momentum":
            m = beta1 * m - lr * g
            p = p + m
        elif name == "RMSProp":
            v = 0.9 * v + 0.1 * g ** 2
            p = p - lr * g / (np.sqrt(v) + eps)
        elif name == "Adam":
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            mh = m / (1 - beta1 ** t); vh = v / (1 - beta2 ** t)
            p = p - lr * mh / (np.sqrt(vh) + eps)
        p = np.clip(p, -8, 8)
        if trace is not None:
            trace.append({"t": t, "prev": prev, "g": g.copy(),
                          "m": m.copy(), "v": v.copy(), "new": p.copy()})
        path.append(p.copy())
    return np.array(path)


with tab2:
    c1, c2 = st.columns([1, 2])
    with c1:
        sname = st.selectbox("Loss surface", list(SURFACES))
        opts = st.multiselect("Optimizers to race", list(OPT_COLORS),
                              default=["GD", "Momentum", "Adam"])
        lr2 = st.select_slider("Learning rate η",
                               [0.001, 0.003, 0.01, 0.03, 0.1, 0.3], 0.01)
        steps2 = st.slider("Steps", 10, 400, 120, 10)
        noise2 = st.slider("SGD gradient noise", 0.0, 1.0, 0.3, 0.05)
        seed = st.number_input("Seed", 0, 999, 7)
        st.markdown("---")
        iopt = st.selectbox("Inspect optimizer ⭐", opts) if opts else None
        t_ins = st.slider("Inspect step t", 1, steps2, min(10, steps2)) if opts else 1
    S = SURFACES[sname]
    rng = np.random.default_rng(int(seed))

    with c2:
        fig, (ax, axl) = plt.subplots(1, 2, figsize=(10, 4.2),
                                      gridspec_kw={"width_ratios": [1.2, 1]})
        gx = np.linspace(*S["xlim"], 240); gy = np.linspace(*S["ylim"], 240)
        XX, YY = np.meshgrid(gx, gy)
        ZZ = S["f"](XX, YY)
        ax.contourf(XX, YY, np.log1p(ZZ - ZZ.min()), levels=24, cmap="Greys", alpha=.55)
        ax.contour(XX, YY, np.log1p(ZZ - ZZ.min()), levels=24, colors="#AAA", linewidths=.4)
        if S["opt"]:
            ax.plot(*S["opt"], "*", c=GOLD, ms=16, zorder=6)
        finals = {}
        paths = {}
        for name in opts:
            path = run_opt(name, S["g"], S["start"], lr2, steps2, noise2,
                           np.random.default_rng(int(seed)))
            paths[name] = path
            ax.plot(path[:, 0], path[:, 1], c=OPT_COLORS[name], lw=1.6,
                    marker=".", ms=2.5, label=name, alpha=.9)
            losses = [S["f"](x, y) for x, y in path]
            axl.semilogy(np.maximum(losses, 1e-12), c=OPT_COLORS[name], lw=1.6, label=name)
            finals[name] = losses[-1]
        ax.plot(*S["start"], "o", c=INK, ms=7, zorder=6)
        if iopt in paths:
            pt = paths[iopt][min(t_ins, len(paths[iopt]) - 1)]
            ax.plot(pt[0], pt[1], marker="*", ms=15, c=OPT_COLORS[iopt],
                    mec="white", mew=1.2, zorder=7)
        ax.set_xlim(*S["xlim"]); ax.set_ylim(*S["ylim"])
        ax.set_title("Trajectories on the loss surface", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        axl.set_title("Loss vs. step (log scale)", fontsize=10)
        axl.legend(fontsize=7)
        for a in (ax, axl):
            a.set_facecolor("white")
        st.pyplot(fig, width="stretch")
        if finals:
            st.write({k: f"{v:.2e}" for k, v in
                      sorted(finals.items(), key=lambda kv: kv[1])} | {})
    st.info(S["note"])

    if iopt:
        st.subheader(f"The exact numbers — {iopt} at step {t_ins} (the ⭐ on the surface)")
        tr = []
        run_opt(iopt, S["g"], S["start"], lr2, steps2, noise2,
                np.random.default_rng(int(seed)), trace=tr)
        r = tr[t_ins - 1]
        eps_ = 1e-8
        cols = {"θ before": r["prev"], "gradient g": r["g"]}
        if iopt == "Momentum":
            cols["velocity v ← 0.9v − ηg"] = r["m"]
        elif iopt == "RMSProp":
            cols["s ← 0.9s + 0.1g²"] = r["v"]
            cols["step = −η·g/(√s+ε)"] = -lr2 * r["g"] / (np.sqrt(r["v"]) + eps_)
        elif iopt == "Adam":
            mh = r["m"] / (1 - 0.9 ** r["t"])
            vh = r["v"] / (1 - 0.999 ** r["t"])
            cols["m (smoothed g)"] = r["m"]
            cols["v (smoothed g²)"] = r["v"]
            cols["m̂ = m/(1−β₁ᵗ)"] = mh
            cols["step = −η·m̂/(√v̂+ε)"] = -lr2 * mh / (np.sqrt(vh) + eps_)
        else:
            cols["step = −η·g"] = -lr2 * r["g"]
        cols["θ after"] = r["new"]
        st.dataframe(pd.DataFrame({k: np.round(val, 5) for k, val in cols.items()},
                                  index=["x", "y"]))
        st.caption("Drag *Inspect step t* and watch the state build up: momentum's v accumulates "
                   "a direction, RMSProp's s tracks squared gradients, Adam does both plus a "
                   "bias correction. On the ravine, compare the x and y rows — that per-axis "
                   "difference is the whole story.")
    quiz("race", "Why does momentum beat plain GD in the ravine?",
         ["It uses a bigger learning rate",
          "Zig-zag gradient components cancel in its running average, while the consistent "
          "downhill component accumulates",
          "It computes second derivatives"],
         1, "averaging successive gradients kills the alternating across-the-ravine part and "
            "reinforces the steady along-the-valley part — see the v column in the table above.")
    with st.expander("The update rules (all of them fit on a napkin)"):
        st.latex(r"\textbf{GD:}\quad \theta \leftarrow \theta - \eta\, g")
        st.latex(r"\textbf{Momentum:}\quad v \leftarrow \beta v - \eta g,\qquad \theta \leftarrow \theta + v")
        st.latex(r"\textbf{RMSProp:}\quad s \leftarrow \rho s + (1-\rho)g^2,\qquad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s}+\epsilon}\, g")
        st.latex(r"\textbf{Adam:}\quad m \leftarrow \beta_1 m + (1{-}\beta_1)g,\quad v \leftarrow \beta_2 v + (1{-}\beta_2)g^2,\quad \theta \leftarrow \theta - \eta\, \frac{\hat m}{\sqrt{\hat v}+\epsilon}")
        st.markdown("Adam = momentum's *smoothed direction* + RMSProp's *per-parameter step size*, "
                    "with bias correction ($\\hat m, \\hat v$) so early steps aren't shrunken.")
