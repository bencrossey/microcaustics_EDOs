edo_pl_lightcurve_comparison.ipynb is the most important notebook.
This produces two plots, for a macrolens-embedded extended dark object (EDO) with a given tau_m = R_90 / R_lens (currently only configured for boson stars) one plot shows a source crossing the EDO's caustic curves compared against the caustic curves of a point-like lens (PL).
The second plot shows the corresponding microlensing lightcurves for the EDO and PL.

boson_star_caustics.ipynb only produces the caustics for a boson star, as well as the critical curves, and additionally plots showing how the extent of the critical curves change for varying tau_m.

pyproject.toml and uv.lock can be used with uv to ensure correct library versions are installed.
