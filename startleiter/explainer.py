import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shap
from metpy.plots import SkewT
from metpy.units import units


def compute_shap(background, model, inputs):
    e = shap.DeepExplainer(model, background)
    return e.shap_values(inputs)


def explainable_plot(sounding, shap_values, prediction):
    # Assign units
    p = sounding.level.values * units.hPa
    T = sounding.sel(variable="TEMP").values * units.degC
    Td = (T.magnitude - sounding.sel(variable="DWPD").values) * units.degC
    U = sounding.sel(variable="U").values * units.knots
    V = sounding.sel(variable="V").values * units.knots

    fig = plt.figure(figsize=(6, 6.5), dpi=300)
    skew = SkewT(fig, rotation=45)

    # Add SHAP values
    shval = shap_values[0, :, :]
    shmax = np.quantile(np.abs(shval), 0.98)
    shval /= shmax
    shval = np.clip(shval, -1, 1)
    for i in range(len(p) - 1):
        skew.plot(p[i:i + 2], T[i:i + 2], lw=5, color='r', alpha=np.clip(shval[i, 0], 0, 1))
        skew.plot(p[i:i + 2], T[i:i + 2], lw=5, color='b', alpha=np.abs(np.clip(shval[i, 0], -1, 0)))
        skew.plot(p[i:i + 2], Td[i:i + 2], lw=5, color='r', alpha=np.clip(shval[i, 1], 0, 1))
        skew.plot(p[i:i + 2], Td[i:i + 2], lw=5, color='b', alpha=np.abs(np.clip(shval[i, 1], -1, 0)))
        skew.plot_barbs(p[i:i + 2], U[i:i + 2], V[i:i + 2], color='r', alpha=np.clip(shval[i, 2:].mean(), 0, 1))
        skew.plot_barbs(p[i:i + 2], U[i:i + 2], V[i:i + 2], color='b', alpha=np.abs(np.clip(shval[i, 2:].mean(), -1, 0)))

    skew.plot(p, T, 'k')
    skew.plot(p, Td, '--k')

    skew.ax.set_ylim(1000, 200)
    skew.ax.set_xlim(max((-30, Td.magnitude.min())), min((35, T.magnitude.max() + 10)))

    cmap = plt.get_cmap('bwr')
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        label="Impact on flying conditions",
        ticks=np.linspace(-1, 1, 21),
        boundaries=np.arange(-1.05, 1.1, .1),
        orientation="horizontal",
        shrink=0.5,
        pad=0.1
    )
    cbar.set_ticks([-1, 1])
    cbar.ax.set_xticklabels(['Negative', 'Positive'], fontsize="small")

    validtime = f"{sounding.attrs['validtime']:%Y-%m-%d}"

    text = f"""
    Site: Cimetta, Switzerland (1600 masl)
    Validtime: {validtime}
    Flying probability: {prediction * 100:.0f}%
    Source: https://github.com/dnerini/startleiter 
    Sounding data: 16080 Milano-Linate (weather.uwyo.edu)"""
    plt.text(-0.03, 1.01, text,
             fontsize="small",
             stretch="condensed",
             linespacing=1.1,
             va="bottom",
             transform=plt.gca().transAxes
             )

    return fig
