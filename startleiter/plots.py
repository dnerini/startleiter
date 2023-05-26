import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from metpy.plots import SkewT
from metpy.units import units

from startleiter import config as CFG

SITES = CFG["sites"]


def text_title():
    return f"""
Startleiter"""


def text_site(site):
    return f"""
{site['name']}, Switzerland ({site['elevation']} masl)"""


def text_validtime(validtime):
    return f"""
{validtime}"""


def text_plot(site, validtime, flyability, max_alt_m, max_dist_km, source):
    return f"""
{site['name']}, Switzerland ({site['elevation']} masl)
{validtime}

Flyability: {flyability * 100:.0f}%
Max flying height: {max_alt_m} m
Max flying distance: {max_dist_km} km

Profile:
{source}

QFF gradients:
DWD-ICON
"""


def explainable_plot(
    site, features, shap_values, flyability, max_alt_m, max_dist_km, min_pressure_hPa
):
    features = features.bfill(dim="level", limit=3)
    # Assign units
    p = features.level.values * units.hPa
    T = features.sel(variable="TEMP").values * units.degC
    Td = (T.magnitude - features.sel(variable="DWPD").values) * units.degC
    U = features.sel(variable="U").values * units.knots
    V = features.sel(variable="V").values * units.knots
    QFF_KL = features.sel(variable="KLO-LUG").values * units.hPa
    QFF_KG = features.sel(variable="KLO-GVE").values * units.hPa

    fig = plt.figure(layout=None, figsize=(6, 6.5))
    gs = fig.add_gridspec(nrows=5, ncols=3)
    ax1 = fig.add_subplot(gs[:3, 2])
    ax2 = fig.add_subplot(gs[3, :])
    ax3 = fig.add_subplot(gs[4, :])
    skew = SkewT(fig, aspect=100, rotation=45, subplot=gs[:3, :2])

    # SHAP values
    shval = shap_values[0, :, :]
    shmax = np.quantile(np.abs(shval), 0.98)
    shval /= shmax
    shval = np.clip(shval, -1, 1)
    for i in range(len(p) - 1):
        if p[i + 1] < min_pressure_hPa * units.hPa:
            continue
        skew.plot(
            p[i : i + 2],
            T[i : i + 2],
            lw=5,
            color="r",
            alpha=np.clip(shval[i, 0], 0, 1),
        )
        skew.plot(
            p[i : i + 2],
            T[i : i + 2],
            lw=5,
            color="b",
            alpha=np.abs(np.clip(shval[i, 0], -1, 0)),
        )
        skew.plot(
            p[i : i + 2],
            Td[i : i + 2],
            lw=5,
            color="r",
            alpha=np.clip(shval[i, 1], 0, 1),
        )
        skew.plot(
            p[i : i + 2],
            Td[i : i + 2],
            lw=5,
            color="b",
            alpha=np.abs(np.clip(shval[i, 1], -1, 0)),
        )
        skew.plot_barbs(
            p[i : i + 2],
            U[i : i + 2],
            V[i : i + 2],
            color="r",
            alpha=np.clip(shval[i, 2:].mean(), 0, 1),
        )
        skew.plot_barbs(
            p[i : i + 2],
            U[i : i + 2],
            V[i : i + 2],
            color="b",
            alpha=np.abs(np.clip(shval[i, 2:].mean(), -1, 0)),
        )

    skew.plot(p, T, "k")
    skew.plot(p, Td, "--k")
    skew.ax.set_xlabel("Temperature (\N{DEGREE CELSIUS})", fontdict=dict(size="small"))
    skew.ax.set_ylabel("Pressure (hPa)", fontdict=dict(size="small"))

    Tmax = T.magnitude.max()
    skew.ax.set_ylim(1000, min_pressure_hPa)
    skew.ax.set_xlim(Tmax - 23, Tmax + 12)

    # Colorbar
    cmap = plt.get_cmap("bwr")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.67, 0.465, 0.015, 0.1])
    cbar = plt.colorbar(sm, cax=cbaxes)
    cbar.set_ticks([])
    cbar.ax.text(
        0.5,
        1.05,
        "Favourable",
        transform=cbar.ax.transAxes,
        va="bottom",
        ha="center",
        fontsize="x-small",
    )
    cbar.ax.text(
        0.5,
        -0.05,
        "Adverse",
        transform=cbar.ax.transAxes,
        va="top",
        ha="center",
        fontsize="x-small",
    )
    validtime = f"{features.attrs['validtime']:%Y-%m-%d}"
    source = f"{features.attrs['source']}"
    ax1.text(
        0,
        1.0,
        text_title(),
        fontsize="x-large",
        stretch="condensed",
        linespacing=1.1,
        va="bottom",
        ha="left",
        transform=ax1.transAxes,
    )
    ax1.text(
        0,
        1,
        text_plot(site, validtime, flyability, max_alt_m, max_dist_km, source),
        fontsize="small",
        stretch="condensed",
        linespacing=1.1,
        ha="left",
        va="top",
        transform=ax1.transAxes,
    )

    ax1.axis("off")

    tt = pd.date_range(validtime, periods=24, freq="1H")
    ax2.plot(tt, QFF_KG[:24], "k-")
    ax3.plot(tt, QFF_KL[:24], "k-")
    for i in range(len(tt) - 1):
        ax2.plot(
            tt[i : i + 2],
            QFF_KG[i : i + 2],
            lw=5,
            color="r",
            alpha=np.clip(shval[i, 4], 0, 1),
        )
        ax2.plot(
            tt[i : i + 2],
            QFF_KG[i : i + 2],
            lw=5,
            color="b",
            alpha=np.abs(np.clip(shval[i, 4], -1, 0)),
        )
        ax3.plot(
            tt[i : i + 2],
            QFF_KL[i : i + 2],
            lw=5,
            color="r",
            alpha=np.clip(shval[i, 5], 0, 1),
        )
        ax3.plot(
            tt[i : i + 2],
            QFF_KL[i : i + 2],
            lw=5,
            color="b",
            alpha=np.abs(np.clip(shval[i, 5], -1, 0)),
        )

    ax2.hlines(0, tt.min(), tt.max(), ls=":", color="black")
    ylim_min = min(0, np.min(QFF_KG[:24].magnitude)) - 2
    ylim_max = max(0, np.max(QFF_KG[:24].magnitude)) + 2
    ax2.set_ylim([ylim_min, ylim_max])
    ax2.set_xlim([tt.min(), tt.max()])
    ax2.set_ylabel(
        "KLO-GVE [hPa]",
        fontsize="small",
    )
    ax2.set_xticklabels([])

    ax3.hlines(0, tt.min(), tt.max(), ls=":", color="black")
    ylim_min = min(0, np.min(QFF_KL[:24].magnitude)) - 2
    ylim_max = max(0, np.max(QFF_KL[:24].magnitude)) + 2
    ax3.set_ylim([ylim_min, ylim_max])
    ax3.set_xlim([tt.min(), tt.max()])
    ax3.set_ylabel(
        "KLO-LUG [hPa]",
        fontsize="small",
    )
    ax3.tick_params(axis="x", labelrotation=45, labelsize="small")
    ax3.xaxis.set_major_formatter(DateFormatter("%a %H"))

    plt.tight_layout()

    return fig


def outlook_plot(site, validtimes, fly_probs, max_alts, max_dists):
    fig, axs = plt.subplots(3, 1, figsize=(7, 4.8), dpi=300)
    # axs[0].step(
    #    validtimes + [validtime + timedelta(days=1)],
    #    np.clip(fly_probs + [fly_prob], 0.01, 1),
    #    where="post",
    #    color="tab:blue",
    # )
    axs[0].text(
        0,
        1.15,
        text_title(),
        fontsize="xx-large",
        stretch="condensed",
        linespacing=1.1,
        va="bottom",
        ha="left",
        transform=axs[0].transAxes,
    )
    axs[0].text(
        0,
        1.01,
        text_site(SITES[site]),
        fontsize="small",
        stretch="condensed",
        linespacing=1.1,
        ha="left",
        va="bottom",
        transform=axs[0].transAxes,
    )

    axs[0].bar(
        validtimes,
        np.clip(fly_probs, 0.01, 1),
        width=0.9,
        align="edge",
        color="tab:blue",
    )
    axs[0].set_ylabel("Flyability []", color="tab:blue")
    axs[0].set_ylim([0, 1])
    axs[0].get_xaxis().set_ticklabels([])

    axs[1].bar(
        validtimes,
        np.clip(max_alts, SITES[site]["elevation"] + 50, None),
        width=0.9,
        align="edge",
        color="tab:orange",
    )
    axs[1].set_ylabel("Max height [masl]", color="tab:orange")
    axs[1].set_ylim(
        [
            SITES[site]["elevation"],
            max(SITES[site]["elevation"] * 2, np.max(max_alts) * 1.01),
        ]
    )
    axs[1].get_xaxis().set_ticklabels([])

    axs[2].bar(
        validtimes,
        np.clip(max_dists, 5, None),
        width=0.9,
        align="edge",
        color="tab:green",
    )
    axs[2].set_ylim([0, 220])
    axs[2].set_ylabel("Max distance [km]", color="tab:green")

    axs[2].tick_params(axis="x", rotation=45)
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    return fig
