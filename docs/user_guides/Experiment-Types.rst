.. _experiment-types:

Experiment Types
================

Depending on the experimental condition, the experimenter has different a priori knowledge about the asymptotes. In python-psignifit 4 this is handled by an option called `experiment_type`. There are three different types available:

### nAFC Experiments ###
The first is meant for n-alternative-forced-choice experiments, i. e. for experiments for which n alternatives are given, an answer is enforced and exactly one alternative is right. You can choose it by passing the string "nACF" to the options, where you replace n with the number of options. For example, if you use 2, 3 or 4 alternatives you may pass:

    options['experiment_type'] = '2AFC'
    options['experiment_type'] = '3AFC'
    options['experiment_type'] = '4AFC'

This mode fixes the lower asymptote gamma to 1/n and leaves the upper asymptote free to vary.

### Yes/No Experiments ###
Intended for simple detection experiments asking subjects whether they perceived a single presented stimulus or not, or any other experiment which has two possible answers of which one is reached for "high" stimulus levels and the other for "low" stimulus levels. You choose it with:

    options['experiment_type'] = 'yes/no'

This sets both asymptotes free to vary and applies a prior to them favouring small values, e.g. asymptotes near 0 and 1 respectively.

### Equal Asymptote Experiments ###
This setting is essentially a special case of Yes/No experiments. Here the asymptotes are "yoked", i. e. they are assumed to be equally far from 0 or 1. This corresponds to the assumption that stimulus independent errors are equally likely for clear "Yes" answers as for clear "No" answers. It is chosen with:

    options['experiment_type'] = 'equal asymptote'

Note that this will make fitting the psychometric function considerably faster than "YesNo" because psignifit 4 has to only fit four rather than five parameters. In this case `gamma=lambda` is enforced.

Find an interactive example in :ref:`Demo 1 <sphx_glr_generated_examples_demo_001.py>`.

