"""Regenerate apps/*/index.html from apps/*/app.py.

Edit an app.py, run `python build.py`, commit, push. That's the whole workflow.
"""
from pathlib import Path

STLITE = "1.8.1"          # https://www.npmjs.com/package/@stlite/browser
TITLES = {
    "classical_ml": "Classical ML — scikit-learn, live",
    "optimizers":   "Gradient descent & optimizers",
    "mlp":          "MLP from scratch",
}
TPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="icon" type="image/svg+xml" href="../../logo.svg"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{v}/build/stlite.css"/>
<style>
  #boot{{
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    max-width: 560px; margin: 18vh auto 0; padding: 24px 28px; text-align: center;
    color: #1E2A22; background: #FBFAF5; border: 1px solid #CFCDBB; border-radius: 6px;
  }}
  #boot h2{{ font-family: Georgia, serif; font-weight: 400; margin: 0 0 8px; }}
  #boot p{{ color: #4A5A4E; font-size: 14px; margin: 0 0 14px; }}
  #boot .bar{{ height: 6px; background: #EAE8DA; border-radius: 3px; overflow: hidden; }}
  #boot .bar i{{ display:block; height:100%; width: 30%; background:#2F6F4E; border-radius:3px;
    animation: slide 1.2s ease-in-out infinite alternate; }}
  @keyframes slide{{ from{{ transform: translateX(-10%);}} to{{ transform: translateX(260%);}} }}
  #boot small{{ color:#4A5A4E; }}
  body{{ margin:0; background:#F2F1E9; }}
</style>
</head>
<body>
<div id="root">
  <div id="boot">
    <img src="../../logo.svg" alt="" style="width:46px;height:46px;border-radius:10px;margin:0 auto 12px;display:block"/>
    <h2>{title}</h2>
    <p>Starting the Python runtime in your browser (Pyodide + NumPy + scikit-learn + Matplotlib).</p>
    <div class="bar"><i></i></div>
    <p style="margin-top:12px"><small>First visit downloads &asymp; 40 MB and can take up to a minute &mdash; it's cached after that. Nothing runs on a server; everything happens on this machine.</small></p>
  </div>
</div>

<script type="text/plain" id="app-code">
{code}
</script>

<script type="module">
  import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{v}/build/stlite.js";
  mount(
    {{
      requirements: ["numpy", "matplotlib", "scikit-learn"],
      entrypoint: "app.py",
      files: {{ "app.py": document.getElementById("app-code").textContent }},
    }},
    document.getElementById("root"),
  );
</script>
</body>
</html>
"""

for name, title in TITLES.items():
    src = Path("apps") / name / "app.py"
    code = src.read_text()
    assert "</script" not in code.lower(), f"{name}: app.py may not contain '</script'"
    out = Path("apps") / name / "index.html"
    out.write_text(TPL.format(title=title, code=code, v=STLITE))
    print("built", out)
