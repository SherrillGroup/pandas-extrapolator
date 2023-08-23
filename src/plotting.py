import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_violin_ALL(
    df,
    vals: {},
    title_name: str,
    pfn: str,
    bottom: float = 0.4,
    ylim=None,
) -> None:
    print(f"Plotting {pfn}")
    kcal_per_mol = "$kcal\cdot mol^{-1}$"
    dbs = list(set(df["DB"].to_list()))
    dbs = sorted(dbs, key=lambda x: x.lower())
    vLabels, vData = [], []

    annotations = []  # [(x, y, text), ...]
    cnt = 1
    plt.rcParams["text.usetex"] = True
    maximum_error, minimum_error = 0, 0
    for k, v in vals.items():
        df[v] = pd.to_numeric(df[v])
        vData.append(df[v].to_list())
        vLabels.append(k)
        m = df[v].max()
        rmse = df[v].apply(lambda x: x**2).mean() ** 0.5
        mae = df[v].apply(lambda x: abs(x)).mean()
        max_error = df[v].apply(lambda x: abs(x)).max()
        text = r"$\mathbf{%.2f}$" % mae
        text += "\n"
        text += r"$\mathit{%.2f}$" % rmse
        text += "\n"
        text += r"$\mathrm{%.2f}$" % max_error
        annotations.append((cnt, m, text))
        cnt += 1
        max_s = df[v].apply(lambda x: x).max()
        min_s = df[v].apply(lambda x: x).min()
        if max_s > maximum_error:
            maximum_error = max_s
        if min_s < minimum_error:
            minimum_error = min_s

    if ylim is None:
        ylim = (minimum_error - 5, maximum_error + 5)

    pd.set_option("display.max_columns", None)
    # print(df[vals.values()].describe(include="all"))
    # transparent figure
    fig = plt.figure(dpi=1000)
    ax = plt.subplot(111)
    vplot = ax.violinplot(
        vData,
        showmeans=True,
        showmedians=False,
        quantiles=[[0.05, 0.95] for i in range(len(vData))],
        widths=0.75,
    )
    # for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    for n, partname in enumerate(["cbars", "cmins", "cmaxes", "cmeans"]):
        vp = vplot[partname]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)
        vp.set_alpha(1)
    quantile_color = "red"
    quantile_style = "-"
    quantile_linewidth = 0.8
    for n, partname in enumerate(["cquantiles"]):
        vp = vplot[partname]
        vp.set_edgecolor(quantile_color)
        vp.set_linewidth(quantile_linewidth)
        vp.set_linestyle(quantile_style)
        vp.set_alpha(1)

    colors = ["blue" if i % 2 == 0 else "green" for i in range(len(vLabels))]
    for n, pc in enumerate(vplot["bodies"], 1):
        pc.set_facecolor(colors[n - 1])
        pc.set_alpha(0.6)

    vLabels.insert(0, "")
    xs = [i for i in range(len(vLabels))]
    xs_error = [i for i in range(-1, len(vLabels) + 1)]
    ax.plot(
        xs_error,
        [1 for i in range(len(xs_error))],
        "k--",
        label=r"$\pm$1 $\mathrm{kcal\cdot mol^{-1}}$",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        xs_error,
        [0 for i in range(len(xs_error))],
        "k--",
        linewidth=0.5,
        alpha=0.5,
        # label=r"Reference Energy",
        zorder=0,
    )
    ax.plot(
        xs_error,
        [-1 for i in range(len(xs_error))],
        "k--",
        zorder=0,
        linewidth=0.6,
    )
    ax.plot(
        [],
        [],
        linestyle=quantile_style,
        color=quantile_color,
        linewidth=quantile_linewidth,
        label=r"5-95th Percentile",
    )
    navy_blue = (0.0, 0.32, 0.96)
    ax.set_xticks(xs)
    minor_yticks = np.arange(ylim[0], ylim[1], 2)
    ax.set_yticks(minor_yticks, minor=True)
    plt.setp(ax.set_xticklabels(vLabels), rotation=90, fontsize="6")
    ax.set_xlim((0, len(vLabels)))
    ax.set_ylim(ylim)
    lg = ax.legend(loc="upper left", edgecolor="black", fontsize="8")
    # lg.get_frame().set_alpha(None)
    lg.get_frame().set_facecolor((1, 1, 1, 0.0))

    ax.set_xlabel("Level of Theory", color="k")
    ax.set_ylabel(r"Error ($\mathrm{kcal\cdot mol^{-1}}$)", color="k")
    ax.grid(color="gray", which="major", linewidth=0.5, alpha=0.3)
    ax.grid(color="gray", which="minor", linewidth=0.5, alpha=0.3)

    # Annotations of RMSE
    for x, y, text in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x, y + 0.1),
            color="black",
            fontsize="5",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    for n, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(colors[n - 1])
        xtick.set_alpha(0.8)

    plt.title(f"{title_name}")
    fig.subplots_adjust(bottom=bottom)
    plt.savefig(f"plots/{pfn}_dbs_violin.png", transparent=False)
    plt.clf()
    return
