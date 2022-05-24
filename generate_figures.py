import pdb, sys, os, math

import numpy as np, matplotlib.pyplot as plt
from collections import OrderedDict as odict

################################################################################

n = int(1e4)
x = np.linspace(-2, 2, n)
plt.figure()
plt.plot(x, x ** 2, label="MSE", color="C1")
plt.plot(x, np.abs(x), label="exact", color="C2")
plt.plot(x, np.sqrt(np.abs(x)), label='"super"-exact', color="C3")
plt.ylim([-0.1, 1.1 * np.max(x)])
plt.legend()
plt.tight_layout()
plt.savefig("figures/penalty_functions.png", dpi=200)

################################################################################

accuracy = odict(
    [
        ("normal", 74.11326599121094),
        ("MSE", 73.14073181152344),
        ("exact", 72.19679260253906),
        ('"super"-exact', 70.16590118408203),
    ]
)

lime_score = odict(
    [
        ("normal", 0.3325492739677429),
        ("MSE", 0.009267126210033894),
        ("exact", 0.00370749831199646),
        ('"super"-exact', 0.0019427312072366476),
    ]
)

plt.figure()
plt.bar(
    list(accuracy.keys()),
    list(accuracy.values()),
    color=[f"C{i}" for i in range(len(accuracy.keys()))],
)
xmin, xmax, _, _ = plt.gca().axis()
rng = xmax - xmin
# plt.plot(["normal", '"super"-exact'], [50, 50], linestyle="--", color="black")
plt.plot(
    [xmin - 0.05 * rng, xmax + 0.05 * rng],
    [50, 50],
    linestyle="--",
    color="black",
)
plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("figures/accuracy.png", dpi=200)
# pdb.set_trace()

plt.figure()
plt.bar(
    list(lime_score.keys()),
    list(lime_score.values()),
    color=[f"C{i}" for i in range(len(lime_score.keys()))],
)
# plt.plot(["normal", '"super"-exact'], [50, 50], linestyle="--", color="black")
plt.yscale("log")
plt.ylabel("Race LIME score")
plt.tight_layout()
plt.savefig("figures/lime_score.png", dpi=200)


################################################################################
accs = [
    73.455,
    73.713,
    73.627,
    73.827,
    73.513,
    73.169,
    72.912,
    72.311,
    55.921,
    54.634,
]
lime_score = [
    0.2527238726615906,
    0.18422257900238037,
    0.11337780207395554,
    0.06382934749126434,
    0.03317005932331085,
    0.014389969408512115,
    0.006740198004990816,
    0.002512869890779257,
    0.0006253707688301802,
    0.00040408188942819834,
]

gams = [
    1.00000000e-02,
    2.78255940e-02,
    7.74263683e-02,
    2.15443469e-01,
    5.99484250e-01,
    1.66810054e00,
    4.64158883e00,
    1.29154967e01,
    3.59381366e01,
    1.00000000e02,
]

fig, axs = plt.subplots(1, 1)
axs = [axs]
axs[0].plot(gams, lime_score, marker=".", color="C0")
axs[0].set_ylabel("LIME score", color="C0")
axs[0].set_xlabel("Penalty Strength")
axs[0].spines["left"].set_color("C0")
axs[0].spines["right"].set_color("C0")
axs[0].tick_params(axis="y", colors="C0", which="both")
axs[0].set_yscale("log")
axs[0].set_xscale("log")

ax = axs[0].twinx()
ax.plot(gams, accs, marker=".", color="C1")
#ax.set_yscale("log")

ax.set_ylabel("Test Accuracy", color="C1")
ax.spines["right"].set_color("C1")
ax.tick_params(axis="y", colors="C1")

#axs[1].loglog(lime_score, accs, marker=".")

plt.tight_layout()
plt.savefig("figures/lime_reg.png", dpi=200)

################################################################################

plt.show()
