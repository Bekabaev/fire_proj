"""Задание 2, шаг 17. Наглядная демонстрация физического симулятора (FBP O-1b, k_ros с площадью как в факте) для презентации:
вытянутость по ветру 0/5/10 м/с, река-барьер, полоса зелёной пашни, холм; поджог - круг 2 км, барьеры в 3 км по ветру.
Выход: figures/sim/phys_demo.png."""
import os, sys, importlib.util
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
_s = importlib.util.spec_from_file_location("s15", os.path.join(os.path.dirname(__file__), "15_sim_synth.py"))
s15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s15)
N = 96; rng = np.random.default_rng(1)
cases = [("штиль", "flat", 0), ("ветер 5 м/с", "flat", 5), ("ветер 10 м/с", "flat", 10),
         ("река, 10 м/с", "river", 10), ("зелёная пашня, 10 м/с", "cropbreak", 10), ("холм, 5 м/с", "hill", 5)]
fig, ax = plt.subplots(2, 3, figsize=(12, 8.2))
cols = ["#f4a261", "#e76f51", "#c1121f", "#6a040f"]
for a, (lab, kind, ws) in zip(ax.ravel(), cases):
    L = s15.landscape(kind, rng)
    for k in ("nonb", "crop", "grass", "ndvi"):                       # барьеры ближе к поджогу: столбцы 58–64 -> 50–56
        L[k] = np.roll(L[k], -8, axis=1)
    if kind == "hill":
        L["elev"] = np.roll(L["elev"], -10, axis=1)
    b = s15.disk(48, 40, 4)
    T = s15.physics(L, b, [(ws, 0)] * 4)
    bg = np.dstack([np.full((N, N), 0.85), 0.85 + 0.1 * L["crop"], 0.7 - 0.2 * L["crop"]]); bg[L["nonb"] > 0.5] = (0.55, 0.7, 0.9)
    a.imshow(bg)
    if np.ptp(L["elev"]) > 1:
        a.contour(L["elev"], levels=8, colors="#888", linewidths=0.5)
    for h, H in enumerate((12, 24, 48, 72)):
        a.contour((T <= H).astype(float), levels=[0.5], colors=cols[h], linewidths=1.8)
    a.imshow(np.ma.masked_where(~b, b), cmap="Greys", vmin=0, vmax=1.2)
    if ws:
        a.arrow(6, 8, 7, 0, width=0.8, color="k")
    a.set_title(f"{lab}: {((T <= 72) & ~b).sum() * 0.25:.0f} км² за 72 ч", fontsize=11); a.set_xlim(20, 80); a.set_ylim(78, 18); a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Физическая модель FBP O-1b (k = {s15.K_ROS}): контуры 12 / 24 / 48 / 72 ч, ячейка 500 м", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(s15.FIGS, "phys_demo.png"), dpi=120)
print("ok")
