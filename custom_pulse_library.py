from __future__ import annotations

from typing import Any, Callable, Dict
import warnings

import numpy as np
from numpy.typing import ArrayLike
from laboneq.core.utilities.pulse_sampler import _pulse_samplers, _pulse_factories
from laboneq.dsl.experiment.pulse import (
    PulseFunctional,
    PulseSampled,
)
# deprecated alias for _pulse_samples, use pulse_library.pulse_sampler(...) instead:
pulse_function_library = _pulse_samplers

def register_pulse_functional(sampler: Callable, name: str | None = None):
    """Build & register a new pulse type from a sampler function.

    The sampler function must have the following signature:

    ``` py

        def sampler(x: ndarray, **pulse_params: Dict[str, Any]) -> ndarray:
            pass
    ```

    The vector ``x`` marks the points where the pulse function is to be evaluated. The
    values of ``x`` range from -1 to +1. The argument ``pulse_params`` contains all
    the sweep parameters, evaluated for the current iteration.
    In addition, ``pulse_params``  also contains the following keys:

    - ``length``: the true length of the pulse
    - ``amplitude``: the true amplitude of the pulse
    - ``sampling_rate``: the sampling rate

    Typically, the sampler function should discard ``length`` and ``amplitude``, and
    instead assume that the pulse extends from -1 to 1, and that it has unit
    amplitude. LabOne Q will automatically rescale the sampler's output to the correct
    amplitude and length.


    Args:
        sampler:
            the function used for sampling the pulse
        name:
            the name used internally for referring to this pulse type

    Returns:
        pulse_factory (function):
            A factory function for new ``Pulse`` objects.
            The return value has the following signature:
            ``` py

                def <name>(
                    uid: str = None,
                    length: float = 100e-9,
                    amplitude: float = 1.0,
                    **pulse_parameters: Dict[str, Any],
                ):
                    pass
            ```
    """
    if name is None:
        function_name = sampler.__name__
    else:
        function_name = name

    def factory(
        uid: str | None = None,
        length: float = 100e-9,
        amplitude: float = 1.0,
        can_compress=False,
        **pulse_parameters: Dict[str, Any],
    ):
        if pulse_parameters == {}:
            pulse_parameters = None
        if uid is None:
            return PulseFunctional(
                function=function_name,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
                can_compress=can_compress,
            )
        else:
            return PulseFunctional(
                function=function_name,
                uid=uid,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
                can_compress=can_compress,
            )

    factory.__name__ = function_name
    factory.__doc__ = sampler.__doc__
    # we do not wrap __qualname__, it throws off the documentation generator

    _pulse_samplers[function_name] = sampler
    _pulse_factories[function_name] = factory
    return factory

@register_pulse_functional
def GaussianSquareDRAG(
    x,
    sigma = None, 
    width = None,
    risefall_sigma_ratio = None,
    #risefall = None,
    beta0 = 0.0,
    beta1 = 0.0,
    zero_boundaries = False,
    *, # 뒤로는 키워드 인자만 받게 제한 
    length,
    **_):
    
    """Create a 2D DRAG pulse for Gaussian Square pules.

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
        risefall_sigma_ratio (float):
            
        beta0 (float):
            Relative amplitude of the first derivative component
        beta1 (float):
            Relative amplitude of the second derivative component    
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): DRAG pulse.
    """

    if sigma is None:
        raise ValueError("Sigma must be specified")
    if width is None and risefall_sigma_ratio is None:
        raise ValueError("Either the pulse width or the risefall_sigma_ratio parameter must be specified.")
    if width is not None and risefall_sigma_ratio is not None:
        raise ValueError(
            "Either the pulse width or the risefall_sigma_ratio parameter can be specified but not both.")
    # if risefall is not None and (width or risefall_sigma_ratio is not None):
    #     raise ValueError("if risefall is specified width and risefall_sigma_ratio cannot be specified")
        
    if width is None and risefall_sigma_ratio is not None:
        width = length * (1 - risefall_sigma_ratio * sigma)
        if width > length:
            raise ValueError("Pulse width cannot be longer then length")
    
    risefall_in_samples = round(len(x) * (1 - width/length)/2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2)/(2*sigma**2))
    
    deriv_gauss_part = gauss_x/(sigma**2)*gauss_part
    
    gauss_sq = np.concatenate(
        (
        gauss_part[:risefall_in_samples] + 1j*beta0*deriv_gauss_part[:risefall_in_samples] - beta1*deriv_gauss_part[:risefall_in_samples]**2,
        np.ones(flat_in_samples),
        gauss_part[risefall_in_samples:] + 1j*beta0*deriv_gauss_part[risefall_in_samples:] - beta1*deriv_gauss_part[risefall_in_samples:]**2,
        )
    )
    
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1]-gauss_x[0])
        delta = np.exp(-(t_left**2)/(2*sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1- delta
    return gauss_sq 

@register_pulse_functional
def GaussianSquare(
    x,
    sigma = None, 
    width = None,
    risefall_sigma_ratio = None,
    # risefall = None, 
    zero_boundaries = False,
    *,
    length, #length는 키워드 인자로만 받고 추가적인 인자는 무시
    **_):
    
    """Create a 2D DRAG pulse for Gaussian Square pules.

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
        risefall_sigma_ratio (float):
            
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): DRAG pulse.
    """

    if sigma is None:
        raise ValueError("Sigma must be specified")
    if width is None and risefall_sigma_ratio is None:
        raise ValueError("Either the pulse width or the risefall_sigma_ratio parameter must be specified.")
    if width is not None and risefall_sigma_ratio is not None:
        raise ValueError(
            "Either the pulse width or the risefall_sigma_ratio parameter can be specified but not both.")
    # if risefall is not None and (width or risefall_sigma_ratio is not None):
    #     raise ValueError("if risefall is specified width and risefall_sigma_ratio cannot be specified")
        
    if width is None and risefall_sigma_ratio is not None:
        width = length * (1 - risefall_sigma_ratio * sigma)

    
    # if risefall is not None:
    #     width = length - 2 * risefall
    
    if width > length:
            raise ValueError("Pulse width cannot be longer then length")
    
        
    
    risefall_in_samples = round(len(x) * (1 - width/length)/2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2)/(2*sigma**2))
    
    #deriv_gauss_part = gauss_x/(sigma**2)*gauss_part
    
    gauss_sq = np.concatenate(
        (
        gauss_part[:risefall_in_samples],
        np.ones(flat_in_samples),
        gauss_part[risefall_in_samples:],
        )
    )
    
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1]-gauss_x[0])
        delta = np.exp(-(t_left**2)/(2*sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1- delta
    return gauss_sq 


##################################################################################################

def sampled_pulse(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of waveform values.

    Arguments:
        samples (numpy.ndarray): waveform envelope data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampled(samples=samples, can_compress=can_compress)
    else:
        return PulseSampled(uid=uid, samples=samples, can_compress=can_compress)


def sampled_pulse_real(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of real values.

    Arguments:
        samples (numpy.ndarray): Real valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.

    !!! version-changed "Deprecated in version 2.51.0"
        Use `sampled_pulse` instead.

    """
    warnings.warn(
        "The `sampled_pulse_real` function, along with `PulseSampledReal`, is deprecated. "
        "Please use `sampled_pulse` instead, as `sampled_pulse_real` now calls `sampled_pulse` internally.",
        FutureWarning,
        stacklevel=2,
    )

    return sampled_pulse(samples, uid=uid, can_compress=can_compress)


def sampled_pulse_complex(
    samples: ArrayLike, uid: str | None = None, can_compress: bool = False
):
    """Create a pulse based on a array of complex values.

    Args:
        samples (numpy.ndarray): Complex valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.

    !!! version-changed "Deprecated in version 2.51.0"
        Use `sampled_pulse` instead.
    """
    warnings.warn(
        "The `sampled_pulse_complex` function, along with `PulseSampledComplex`, is deprecated. "
        "Please use `sampled_pulse` instead, as `sampled_pulse_complex` now calls `sampled_pulse` internally.",
        FutureWarning,
        stacklevel=2,
    )

    return sampled_pulse(samples, uid=uid, can_compress=can_compress)


def pulse_sampler(name: str) -> Callable:
    """Return the named pulse sampler.

    The sampler is the original function used to define the pulse.

    For example in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the sampler is the *undecorated* function `const`. Calling
    `pulse_sampler("const")` will return this undecorated function.

    This undecorate function is called a "sampler" because it is used by
    the LabOne Q compiler to generate the samples played by a pulse.

    Arguments:
        name: The name of the sampler to return.

    Return:
        The sampler function.
    """
    return _pulse_samplers[name]


def pulse_factory(name: str) -> Callable:
    """Return the named pules factory.

    The pulse factory returns the description of the pulse used to specify
    a pulse when calling LabOne Q DSl commands such as `.play(...)` and
    `.measure(...)`.

    For example, in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the factory is the *decorated* function `const`. Calling
    `pulse_factory("const")` will return this decorated function. This is
    the same function one calls when calling `pulse_library.const(...)`.

    Arguments:
        name: The name of the factory to return.

    Return:
        The factory function.
    """
    return _pulse_factories[name]
