import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from metpy.plots import SkewT
from metpy.units import units

# https://github.com/slundberg/shap/issues/2189#issuecomment-1048384801
tf.compat.v1.disable_v2_behavior()


def compute_shap(background, model, inputs):
    # e = shap.DeepExplainer(model, background)
    e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    return e.shap_values(inputs)


def explainable_plot(
    sounding, shap_values, flyability, max_alt_m, max_dist_km, min_pressure_hPa
):
    # Assign units
    p = sounding.level.values * units.hPa
    T = sounding.sel(variable="TEMP").values * units.degC
    Td = (T.magnitude - sounding.sel(variable="DWPD").values) * units.degC
    U = sounding.sel(variable="U").values * units.knots
    V = sounding.sel(variable="V").values * units.knots

    fig = plt.figure(figsize=(4.5, 5.5), dpi=300)
    skew = SkewT(fig, rotation=45, aspect=100)

    # SHAP values
    shval = shap_values[0, :, :]
    # split DWPD between TEMP and DWPT
    shval[:, 0] += shval[:, 1] / 2
    shval[:, 1] /= 2
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

    skew.ax.set_ylim(1000, min_pressure_hPa)
    skew.ax.set_xlim(T.magnitude[0] - 25, T.magnitude[0] + 10)

    # Colorbar
    cmap = plt.get_cmap("bwr")
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.19, 0.22, 0.015, 0.1])
    cbar = plt.colorbar(sm, cax=cbaxes)
    cbar.set_ticks([])
    cbar.ax.text(
        0.5,
        1.0,
        "Favourable",
        transform=cbar.ax.transAxes,
        va="bottom",
        ha="center",
        fontsize="xx-small",
    )
    cbar.ax.text(
        0.5,
        -0.01,
        "Adverse",
        transform=cbar.ax.transAxes,
        va="top",
        ha="center",
        fontsize="xx-small",
    )

    validtime = f"{sounding.attrs['validtime']:%Y-%m-%d}"
    source = f"{sounding.attrs['source']}"

    text_title = f"""
Startleiter"""
    skew.ax.text(
        0,
        1.05,
        text_title,
        fontsize="xx-large",
        stretch="condensed",
        linespacing=1.1,
        va="bottom",
        ha="left",
        transform=skew.ax.transAxes,
    )

    text_sx_top = f"""
Cimetta, Switzerland (1600 masl)"""
    skew.ax.text(
        0,
        1.01,
        text_sx_top,
        fontsize="small",
        stretch="condensed",
        linespacing=1.1,
        ha="left",
        va="bottom",
        transform=skew.ax.transAxes,
    )

    text_dx_top = f"""
{validtime}"""
    skew.ax.text(
        1.0,
        1.01,
        text_dx_top,
        fontsize="small",
        stretch="condensed",
        linespacing=1.1,
        ha="right",
        va="bottom",
        transform=skew.ax.transAxes,
    )

    plt.subplots_adjust(bottom=0.2)

    text_sx_bottom = f"""
Flyability: {flyability * 100:.0f}%
Max flying height: {max_alt_m} m
Max flying distance: {max_dist_km} km
Input: {source}"""
    skew.ax.text(
        0.0,
        -0.1,
        text_sx_bottom,
        fontsize="small",
        stretch="condensed",
        linespacing=1.1,
        ha="left",
        va="top",
        transform=skew.ax.transAxes,
    )

    return fig
