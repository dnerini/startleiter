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

    fig = plt.figure(figsize=(6, 5), dpi=300)
    skew = SkewT(fig, rotation=45)

    # SHAP values
    shval = shap_values[0, :, :]
    # split DWPD between TEMP and DWPT
    shval[:, 0] += shval[:, 1] / 2
    shval[:, 1] /= 2
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

    skew.ax.set_ylim(1000, 400)
    skew.ax.set_xlim(Td.magnitude[0] - 20, Td.magnitude[0] + 20)

    cmap = plt.get_cmap('bwr')
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbaxes = fig.add_axes([0.945, 0.55, 0.02, 0.25])
    cbar = plt.colorbar(sm, cax=cbaxes)
    cbar.ax.set_title('Favourable', fontsize="x-small")
    cbar.ax.set_xlabel('Adverse', fontsize="x-small")
    cbar.set_ticks([])

    validtime = f"{sounding.attrs['validtime']:%Y-%m-%d}"

    text_sx_top = f"""
Cimetta, Switzerland (1600 masl)
{validtime}
Radiosounding 00Z 16064 Cameri"""
    skew.ax.text(0, 1.01, text_sx_top,
             fontsize="small",
             stretch="condensed",
             linespacing=1.1,
             va="bottom",
             transform=skew.ax.transAxes
             )

    text_dx_top = f"""
Flying probability: {prediction * 100:.0f}%"""
    skew.ax.text(1.0, 1.01, text_dx_top,
             fontsize="small",
             stretch="condensed",
             linespacing=1.1,
             ha="right",
             va="bottom",
             transform=skew.ax.transAxes
             )

    text_sx_bottom = f"""
Credits:
    - Flight data: xcontest.org
    - Radiosounding data: weather.uwyo.edu
    - Impact score: github.com/slundberg/shap
    - SkewT plot: unidata.github.io/MetPy"""
    skew.ax.text(0.01, 0.01, text_sx_bottom,
             fontsize="x-small",
             stretch="condensed",
             linespacing=1.1,
             ha="left",
             va="bottom",
             transform=skew.ax.transAxes
             )

    return fig
