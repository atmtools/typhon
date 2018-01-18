How to construct an atm_fields_compact?
=======================================

An :arts:`atm_fields_compact` is a compact set of atmospheric fields on a
common set of grids.

Data is supposed to contain basic atmsopheric fields for a radiative transfer
calculation (i.e., temperature, altitude, and gas VMRs) and is stored in a
:class:`~typhon.arts.griddedfield.GriddedField4`.

The following code snippet demonstrates how to create an
:arts:`atm_fields_compact` and write it to an ARTS XML file.

.. code-block:: python

    import numpy as np
    import typhon


    # Initialize an empty GriddedField4 object.
    atm_fields_compact = typhon.arts.types.GriddedField4()

    # Create required grids.
    field_names = ['T', 'z', 'abs_species-H2O']
    p_grid = typhon.math.nlogspace(1000e2, 0.01e2, 50)
    lat_grid = np.array([])
    lon_grid = np.array([])

    # Assign the `grids` attribute with a list of the grids created.
    atm_fields_compact.grids = [field_names, p_grid, lat_grid, lon_grid]
    atm_fields_compact.gridnames = [
        'Fieldnames', 'Pressure', 'Latitude', 'Longitude',
    ]

    # Create (dummy) data arrays.
    T = 300 * np.ones(p_grid.size)
    z = np.linspace(0, 80e3, p_grid.size)
    vmr = 0.04 * (p_grid / p_grid[0])**1.7

    # The data is stored as :arts:`Tensor4` which is represented as a
    # four-dimensional ndarray in Python. We stack our (one-dimensional)
    # arrays to create the required tensor. The following line adds two
    # extra dimension to account for the empty latitude/longitude dimensions.
    # The resulting data has the shape `[3, 50, 1, 1]`.
    data_tensor = np.stack(
        [x.reshape(-1, 1, 1) for x in (T, z, vmr)]
    )
    atm_fields_compact.data = data_tensor

    # Manually check if grid and data dimensions match. This is done
    # automatically when assigning data or before writing GriddedFields to an
    # XML file.
    atm_fields_compact.check_dimension()

    # Write the atm_fields_compact to an XML file that can be read by ARTS.
    typhon.arts.xml.save(atm_fields_comapct, 'atmfield.xml')
