import numpy as np


class Remedian:
    """
    Remedian object for a robust averaging method for large data sets.
    Implementation of the Remedian algorithm, see [1]_ [2]_ [3]_ for
    references. This algorithm is used to approximate the median of several
    data chunks if these data chunks cannot (or should not) be loaded into
    memory at once. See "Notes" section for further information.
    Parameters
    ----------
    obs_size : ndarray
        The shape of each data chunk (=observation) to be fed into the Remedian
        object.
    n_obs : int
        The number of observations to be stored within each array.
        If `n_obs` >= `t`, Remedian will equal the median. The smaller this
        parameter, the fewer data have to be loaded into memory at once, but
        the less accurate the approximation of the median will be.
    t : int
        The total number of observations from which a median should be
        approximated.
    Attributes
    ----------
    obs_count : int
        Counter of number of observations that have already been given
        to the Remedian object.
    remedian : None | ndarray, shape(obs_size)
        The calculated remedian of the same shape as the input data.
        Will be None until all observations `n_obs` have been fed into
        the object using the add_obs method.
    Notes
    -----
    Given a data chunk of size `obs_size`, and `t` data chunks overall, the
    Remedian class sets up a number `k_arrs` of arrays of length `n_obs`.
    The median of the `t` data chunks of size `obs_size` is then approximated
    as follows: One data chunk after another is fed into the `n_obs` positions
    of the first array. When the first array is full, its median is calculated
    and stored in the first position of the second array. After this, the first
    array is re-used to fill the second position of the second array, etc.
    When the second array is full, the median of its values is stored in the
    first position of the third array, and so on.
    The final "Remedian" is the median of the last array, after all `t` data
    chunks have been fed into the object.
    In other words, given an n-dimensional array, the Remedian class
    approximates the median of this array across the ith dimension and you have
    to break up your n-dimensional array into `t` n-1-dimensional arrays that
    are given to Remedian one after another.
    References
    ----------
    .. [1] P.J. Rousseeuw, G.W. Bassett Jr., "The remedian: A robust averaging
       method for large data sets", Journal of the American Statistical
       Association, vol. 85 (1990), pp. 97-104
    .. [2] M. Chao, G. Lin, "The asymptotic distributions of the remedians",
       Journal of Statistical Planning and Inference, vol. 37 (1993), pp. 1-11
    .. [3] Domenico Cantone, Micha Hofri, "Further analysis of the remedian
       algorithm", Theoretical Computer Science, vol. 495 (2013), pp. 1-16

       https://stackoverflow.com/questions/62352762/is-it-possible-to-compute-median-while-data-is-still-being-generated-python-onl
       https://wis.kuleuven.be/stat/robust/papers/publications-1990/rousseeuwbassett-remedian-jasa-1990.pdf
       https://github.com/sappelhoff/remedian/blob/master/remedian/remedian.py


    TODO:
        - [ ] Can we make this work so we don't need t?

    Examples:
        >>> import sys, ubelt
        >>> from geowatch.utils.remedian import Remedian
        >>> #
        >>> shape = (7, 5)
        >>> self = Remedian(shape, n_obs=11, t=101, allow_nan=True)
        >>> #
        >>> obs0 = np.arange(7 * 5).reshape(7, 5).astype(np.float32)
        >>> raw_obs = []
        >>> for _ in range(self.t):
        >>>     obs = obs0.copy()
        >>>     obs[2:, 3:] = np.random.rand(5, 2)
        >>>     obs[2:, 3:][obs[2:, 3:] > 0.99] = np.nan
        >>>     raw_obs.append(obs.copy())
        >>>     self.add_obs(obs)
        >>> approx_median = self.remedian
        >>> real_median = np.median(np.stack(raw_obs, -1), axis=-1)
        >>> delta = approx_median - real_median
        >>> print(f'approx_median={approx_median}')
        >>> print(f'real_median={real_median}')
        >>> print(f'delta={delta}')
    """

    def __init__(self, obs_size, n_obs, t, allow_nan=False):
        """Initialize the Remedian object.
        See class docstring for more thorough information.
        Parameters
        ----------
        obs_size : ndarray
            Size of the observations. Must be (1,) for scalars.
        n_obs : int
            Observations per array.
        t : int
            Number of total observations.
        """
        if n_obs <= 1:
            raise ValueError('`n_obs` of <= 1 does not make sense.')

        self.obs_size = list(obs_size)
        self.n_obs = n_obs
        self.t = t

        # Calculate the number of arrays needed and their sizes
        self.k_arrs = self._calc_k_arrs()
        self.k_arr_sizes = self._calc_k_arr_sizes()

        # Initialize the arrays
        self.arrs = [np.zeros(self.obs_size + [s]) for s in self.k_arr_sizes]

        # counter for observations within each array
        self.obs_idx_counter = [0 for arr in range(self.k_arrs)]

        # Modulos of observations to assign to correct array later
        self.modulos = [self.n_obs**i for i in range(1, 1 + self.k_arrs)]

        # Counter of received observations
        self.obs_count = 0

        # Set the median value to None until we have it
        self.remedian = None
        if allow_nan:
            self.median_func = np.nanmedian
        else:
            self.median_func = np.median

    def _calc_k_arrs(self):
        """Calculate number of arrays to accommodate the observations."""
        tmp = self.n_obs
        k_arrs = 1
        while tmp <= self.t:
            tmp *= self.n_obs
            k_arrs += 1
        return k_arrs

    def _calc_k_arr_sizes(self):
        """Calculate the size of each array to accomodate the observations."""
        k_arr_sizes = [self.n_obs for i in range(self.k_arrs)]
        k_arr_sizes[-1] = int(np.ceil(self.t / (self.n_obs**(self.k_arrs - 1))))
        return k_arr_sizes

    def add_obs(self, obs):
        """Add an observation to the Remedian.
        Parameters
        ----------
        obs : ndarray, shape(obs_size)
            A single data observation.
        """
        # We only work if:
        # ... we get an observation of correct size
        # ... we have not yet received all observations already
        if list(obs.shape) != self.obs_size:
            raise ValueError('Expected observation of size {} but received: '
                             '{}'.format(self.obs_size, list(obs.shape)))
        if self.obs_count > (self.t - 1):
            raise RuntimeError('Already collected {} observations out of t={} '
                               'The remedian is {}'.format(self.obs_count,
                                                           self.t,
                                                           self.remedian))

        # We accept a new observation
        self.obs_count += 1

        # Add the data to the first array
        # and increment the counter for the next data
        obs_idx = self.obs_idx_counter[0]
        self.arrs[0][..., obs_idx] = obs
        self.obs_idx_counter[0] += 1

        # We can notice whenever an array is full using modulo operations
        # on the observation counter self.obs_count.
        # When an array is full, calculate the median and put the result
        # into the next array. Then reset the counters and start filling
        # previous arrays again
        for arr_i, mod in enumerate(self.modulos):
            if self.obs_count % mod == 0:
                data = self.arrs[arr_i]
                m_tmp = self.median_func(data, axis=-1, overwrite_input=True)
                self.arrs[arr_i + 1][..., self.obs_idx_counter[arr_i + 1]] = m_tmp
                self.obs_idx_counter[arr_i + 1] += 1
                self.obs_idx_counter[arr_i] = 0

        # If all observations have been received,
        # calculate the median of the last array.
        # This is the robust approximation of the median
        if self.obs_count == self.t:
            self.remedian = self.median_func(self.arrs[-1], axis=-1,
                                             overwrite_input=True)
