r""" Variational Auto-encoder (VAE) for 3D Point Clouds 

#########################
Variational Approximation
#########################

Let :math:`X` be the observed, :math:`Z` be the hidden (latent) variable, such that :math:`\Prob[q]{Z}` is the approximation posterior to :math:`\Prob[p]{Z}{X}`.

.. math:: 
    \underbrace{\Prob{Z}{X}}_\text{Posterior} &= \frac{\overbrace{\Prob{X}{Z}}^\text{Likelihood} \overbrace{\Prob{Z}}^\text{Prior}}{\underbrace{\Prob{X}}_\text{Evidence}} \\
    \shortintertext{Evidence:}
    \Prob{X}    &= \frac{\Prob{X}{Z} \Prob{Z}}{\Prob{Z}{X}} \\
                &\approx \frac{\Prob{X}{Z} \Prob{Z}}{\Prob[q]{Z}}

*******************************************
Kullback-Leibler Divergence (KL-divergence) 
*******************************************

.. math::
    \KL{q}{p} &= - \int \Prob[q]{x} \log\frac{\Prob[p]{x}}{\Prob[q]{x}} \dd{x}

Properties
==========

- :math:`\KL{q}{p} \geq 0`
    
    - :math:`\KL{q}{p} = 0` iff :math:`\forall x: \Prob[q]{x} \equiv \Prob[p]{x}`

- :math:`\KL{q}{p} \neq \KL{p}{q}`

.. math::
    q^\ast &= \argmin_{q \in Q} \KL{\Prob[q]{Z}}{\Prob[p]{Z}{X}}

However, the problem remains: KL-divergence explicitly depends on posterior.

Transformations
---------------

.. math::
    \KL{\Prob[q]{Z}}{\Prob[p]{Z}{X}}
        &= - \int \Prob[q]{z} \log\frac{\Prob[p]{z}{X}}{\Prob[q]{z}} \dd{z} \\
        &= \int \Prob[q]{z} \log\frac{\Prob[q]{z}}{\Prob[p]{z}{X}} \dd{z} \\
        &= \int \Prob[q]{z} \log\frac{\Prob[q]{z} \Prob[p]{X}}{\Prob[p]{z, X}} \dd{z} \\
        &= \underbrace{\E[q]{\log\Prob{X}}}_\text{does not depends on q} + \int \Prob[q]{z} \log\frac{\Prob[q]{z}}{\Prob[p]{z, X}} \dd{z} \\
        &= \underbrace{\log\Prob{X}}_\text{Evidence} - \underbrace{\int \Prob[q]{z} \log\frac{\Prob[p]{z, X}}{\Prob[q]{z}} \dd{z}}_\text{Evidence Lower Bound: ELBO} \\
    \shortintertext{Rearranging:}
    \text{Evidence} &= \text{ELBO} + \underbrace{\mathop{KL}}_\text{KL $\geq 0$} \\
                    &\geq \text{ELBO}


Variational Auto-encoder
------------------------

*Variational Auto-encoders* (VAEs) are popular *generative models* being used in many domains.

Traditional Derivation of a VAE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We assume some *process* that generate the data, such as a latent variable generative model.

We can formalize as follows:

1. Sample *latent* representation :math:`z` from some *prior* distribution :math:`z \follows \Prob[p]{z}`
2. Based on the sample, create the actual representation :math:`x`, modelled itself as a stochastic process :math:`x \follows \Prob[p]{x}{z}`




Objective Function: Evidence Lower BOund (ELBO)
-----------------------------------------------

The objective function is to maximize the ELBO of :math:`x`.

.. math::
    \max_q \ELBO{x}

.. math::
    \ELBO{x}    &= \underbrace{\int \Prob[q]{z}{x} \log\Prob[p]{x}{z} \dd{z}}_\text{the \textit{reconstruction term}} + \underbrace{\int \Prob[q]{z}{x} \log\frac{\Prob[q]{z}{x}}{\Prob[p]{z}} \dd{z}}_\text{KL Divergence} \\
                &= \int \Prob[q]{z}{x} \log\Prob{x}{z} \dd{z} + \KL{q}{p}   

where

- :math:`\Prob[p]{z}` is the *prior* on the *latent representation* :math:`z`
- :math:`\Prob[q]{z}{x}` is the *variational encoder*
- :math:`\Prob[p]{x}{z}` is the *decoder*: how likely is the input :math:`x` given the latent representation :math:`z`

"""

import os

from functools import partial
from itertools import product, tee

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import tensorflow as tf

    from keras.models import *
    from keras.layers import *
    from keras.initializers import *
    from keras.optimizers import *
    from keras.objectives import *
    from keras.callbacks import * 
    from keras import backend as K
    # load vgg model
    from keras.applications.vgg16 import VGG16

    from keras.utils import generic_utils

    from tensorflow_probability.python.layers import *

from compression.utils import *

class VAE:
    def __init__(self, *argv, **kwargs):
        """
        self.encoder = Sequential([
            InputLayer(input_shape=self.input_shape),
            Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
            Conv3D(
        """

