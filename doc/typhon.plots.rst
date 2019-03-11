plots
=====

.. automodule:: typhon.plots

.. currentmodule:: typhon.plots

.. autosummary::
   :toctree: generated

   center_colorbar
   channels
   cmap2act
   cmap2c3g
   cmap2cpt
   cmap2ggr
   cmap2rgba
   cmap2txt
   cmap_from_act
   cmap_from_txt
   colored_bars
   colors2cmap
   figsize
   get_subplot_arrangement
   get_material_design
   heatmap
   HectoPascalFormatter
   HectoPascalLogFormatter
   label_axes
   plot_arts_lookup
   plot_bitfield
   plot_distribution_as_percentiles
   profile_p
   profile_p_log
   profile_z
   ScalingFormatter
   scatter_density_plot_matrix
   set_xaxis_formatter
   set_yaxis_formatter
   sorted_legend_handles_labels
   styles
   supcolorbar
   plot_ppath
   plot_ppath_field
   ppath_field_minmax_posbox
   adjust_ppath_field_box_by_minmax
   plot_ppath_field_zenith_coverage_per_gp_p

Typhon named colors
^^^^^^^^^^^^^^^^^^^

Typhon provides a number of named colors that can be used after importing 
:mod:`typhon.plots`:

>>> plt.plot(x, y, color='ty:uhh-red')

.. plot:: pyplots/named_colors.py

Typhon style sheet
^^^^^^^^^^^^^^^^^^

Typhon provides a number of style sheets that can be used to alter the
default appearance of matplotlib plots.

>>> plt.style.use(typhon.plots.styles.get('typhon'))

.. plot:: pyplots/stylesheet_gallery.py
