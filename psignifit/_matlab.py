_CONFIG_KEYS_MAT2PY = {
    'sigmoidName': 'sigmoid',
    'expType': 'experiment_type',
    'expN': 'experiment_choices',
    'estimateType': 'estimate_type',
    'confP': 'confidence_percentiles',
    'instantPlot': 'instant_plot',
    'maxBorderValue': 'max_bound_value',
    'moveBorders': 'move_bounds',
    'dynamicGrid': None,
    'widthalpha': 'width_alpha',
    'threshPC': 'thresh_PC',
    'CImethod': 'CI_method',
    'gridSetType': 'grid_set_type',
    'fixedPars': 'fixed_parameters',
    'nblocks': 'pool_max_blocks',
    'useGPU': None,
    'poolMaxGap': None,
    'poolMaxLength': None,
    'poolxTol': None,
    'betaPrior': 'beta_prior',
    'verbose': 'verbose',
    'stimulusRange': 'stimulus_range',
    'fastOptim': None,
}
_CONFIG_KEYS_MAT_IGNORE = ('theta0')
_MATLAB_PARAMETERS = ['threshold', 'width', 'lambda', 'gamma', 'eta']


def param_matlist2pydict(param_list):
    """ Transform parameter list from matlab-psignifit to the dict of python-psignifit. """
    return {k: v
            for k, v in zip(_MATLAB_PARAMETERS, param_list)
            if v is not None}


def param_pydict2matlist(param_dict):
    """ Transform parameter dict from python-psignifit to the list of matlab-psignifit. """
    return [param_dict[p] for p in _MATLAB_PARAMETERS]


def _exptype_mat2py(mat_type):
    mat2py = {
        'YesNo': 'yes/no',
        'equalAsymptote': 'equal asymptote',
    }
    if mat_type in mat2py:
        return mat2py[mat_type]
    else:
        return mat_type


_CONFIG_VALUES_MAT2PY = {
    'fixedPars': param_matlist2pydict,
    'expType': _exptype_mat2py,
}


def config_from_matlab(matlab_config, raise_matlab_only=True):
    """ Transform an option dict for matlab-psignifit to the configs
        expected by python-psignifit.
    """
    py_config = {}
    for mat_key, mat_value in matlab_config.items():
        if mat_key in _CONFIG_KEYS_MAT_IGNORE:
            continue
        elif mat_key not in _CONFIG_KEYS_MAT2PY:
            raise ValueError(f"Unknown psignifit option '{mat_key}'.")

        py_key = _CONFIG_KEYS_MAT2PY[mat_key]
        if py_key is None:
            if raise_matlab_only:
                raise ValueError(f"Psignifit option '{mat_key}' is supported only in the matlab version.\n"
                                 "Remove this from the configuration to use the python psignifit.")
            else:
                continue

        if mat_key in _CONFIG_VALUES_MAT2PY:
            py_config[py_key] = _CONFIG_VALUES_MAT2PY[mat_key](mat_value)
        else:
            py_config[py_key] = mat_value

    return py_config