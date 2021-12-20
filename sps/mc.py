import numpy as np 
from functools import cached_property 

class MarkovChain:
    """ Discrete-time Markov chain. Provides methods to draw samples.

    The magnitudes of the products of transition rates with frame intervals
    are assumed to be <<1. (That is, we assume that the probability of two
    state transitions in one frame interval is negligible.)
    
    init
    ----
        TM      :   2D ndarray, transition rates. The element TM[i,j] is assumed
                    to give the transition rate from the i^th to the j^th state
                    in Hz.

        dt      :   str, frame interval (seconds)

    """
    def __init__(self, TM: np.ndarray, dt: float=1.0):
        self.dt = dt
        self.TM = np.asarray(TM)
        assert len(self.TM.shape) == 2
        assert TM.shape[0] == TM.shape[1]

        self.n_states = self.TM.shape[0]
        self.states = np.arange(self.n_states)

        # Normalize transition matrix
        self.P = self.TM.copy() * dt
        self.P[self.states, self.states] = 1.0 - \
            (self.P.sum(axis=1) - np.diagonal(self.P))

    def is_diag(self, M: np.ndarray) -> bool:
        """
        Return True if a 2D ndarray is diagonal, 0 otherwise.

        """
        i, j = np.nonzero(M)
        return np.all(i==j)

    @cached_property 
    def stat_dist(self) -> np.ndarray:
        """
        Stationary distribution; 1D ndarray of length (n_states,).

        Returns
        -------
            1D ndarray of shape (n_states,), the stationary
                distribution for the Markov chain

        """
        if self.is_diag(self.P):
            return np.diag(self.TM) / self.TM.sum()
        else:
            L, V = np.linalg.eig(self.P.T)
            v = V[:,np.argmin(np.abs(L-1.0))]
            return v / v.sum()
 
    def __call__(self, n: int, initial: int=None) -> np.ndarray:
        """
        Simulate a single state history of length *n*.

        If no initial state is provided, the initial state is drawn
        from the stationary distribution.

        Parameters
        ----------
            n   :   int, number of timepoints to simulate

        Returns
        -------
            1D ndarray of shape (n,), the state indices at each
                timepoint

        """
        if self.is_diag(self.TM):
            return np.ones(n, dtype=np.int64) * (np.random.choice(self.states, p=self.stat_dist) if initial is None else initial)

        else:
            s = np.empty(n, dtype=np.int64)

            if initial is None:
                s[0] = np.random.choice(self.states, p=self.stat_dist)
            else:
                s[0] = initial 

            for j in range(1, n):
                s[j] = np.random.choice(self.states, p=self.P[s[j-1],:])
            return s

    def __enter__(self):
        return self 

    def __exit__(self, exc_type, exc_val, tb):
        if not exc_type is None:
            return False
        return True 
