NOTE: Currently this only works with the exact python and dependency versions shown in the pyproject.toml, I would recommend installing these direct from is file or the uv.lock file using uv.

edo_pl_lightcurve_comparison.ipynb is the most important notebook.
This produces two plots, for a macrolens-embedded extended dark object (EDO) (any with a radial mass profile normalised to r_90 can be used if the profile is provided). With a given tau_m = R_90 / R_lens, set by the tau_m_val parameter, one plot shows a source crossing the EDO's caustic curves compared against the caustic curves of a point-like lens (PL).
The second plot shows the corresponding microlensing lightcurves for the EDO and PL.

For now, the example source trajectory is hard coded into the source_trajectory function but the plan is to make this a function with impact parameter and trajectory angle inputs. If you change the source trajectory here, you need to likewise change the scaling of the t/t_E axes in the light curve plots.



boson_star_caustics.ipynb only produces the caustics for a boson star, as well as the critical curves, and additionally plots showing how the extent of the critical curves change for varying tau_m.


