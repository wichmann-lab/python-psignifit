"""This module defines the basic configuration object for psignifit.


"""


class PsignifitConfException(Exception):
    pass

class Conf:
    """The basic configuration object for psignifit.

    This class contains a set of valid options and the corresponding sanity
    checks.

    It raises `PsignifitConfException` if an invalid option is specified or
    if a valid option is specified with a value outside of the allowed range.

    Note for the developer: if you want to add a new valid option `foobar`,
    expand the `Conf.valid_opts` tuple (in alphabetical order) and add any
    check in a newly defined method `def check_foobar(self, value)`, which
    raises `PsignifitConfException` if `value` is outside of the accepted range
    for `foobar`.
    """
    # set of valid options for psignifit. Add new attributes to this tuple
    valid_opts = (
             'beta_prior',
             'borders',
             'CI_method',
             'confP',
             'dynamic_grid',
             'estimate_type',
             'experiment_type',
             'grid_set_eval',
             'instant_plot',
             'max_border_value',
             'mb_stepN',
             'move_borders',
             'nblocks',
             'pool_maxgap',
             'pool_max_length',
             'pool_xtol',
             'priors',
             'set_borders_type',
             'sigmoid',
             'stepN',
             'stimulus_range'
             'threshPC',
             'uniform_weight',
             'width_alpha',
             )

    def __init__(self, **kwargs):
        # we only allow keyword arguments
        #
        # - first set defaults
        #    ... no defaults yet ...
        # - overwrite defaults with user preferences
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def __setattr__(self, name, value):
        if name in self.valid_opts:
            # first run checks for the supplied option, if any are available
            if hasattr(self, 'check_'+name):
                # run the check
                # the check method should raise if value is not valid
                getattr(self, 'check_'+name)(value)
            super().__setattr__(name, value)
        else:
            raise PsignifitConfException(f'Invalid option "{name}"!')

    # template for an option checking method
    # def check_foobar(self, value):
    #    if value > 10:
    #       raise PsignifitConfException(f'Foobar must be < 10: {value} given!')

    def __repr__(self):
        # give an nice string representation of ourselves
        _str = []
        for name in sorted(self.valid_opts):
            # if name is not defined, returns None
            value = getattr(self, name, None)
            _str.append(f'{name}: {value}')
        return '\n'.join(_str)

