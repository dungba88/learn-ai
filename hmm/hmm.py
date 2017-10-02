"""
Hidden Markov Model implementation
"""

class HMM(object):
    """
    Represent a HMM

    N is the number of states
    M is the number of emission values

    initial_prob is an array of length N, whose element a[i] is the probability that
    the initial state will be state i-th

    the state_prob is a NxN whose each cell a[i, j] is the probability to transition
    from state i-th to state j-th.

    the emissions_prob is a NxM matrix which each cell a[i, j] is the probability
    to emit value j-th from state i-th
    """

    def __init__(self, init_prob=None, state_prob=None, emission_prob=None):
        self.init_prob = init_prob
        self.state_prob = state_prob
        self.emission_prob = emission_prob

    def forward(self, sequence):
        """
        calculate the probability that the sequence is generated by this HMM
        it will only works if initial_state_prob, state_transitions and
        observation_transitions are not None.
        """
        if self.state_prob is None \
                or self.emission_prob is None \
                or self.init_prob is None:
            raise RuntimeError('one or more of the matrix probabilities are None')

        if not sequence:
            raise RuntimeError('sequence is None or empty')

        N = len(self.state_prob)
        T = len(sequence)

        caches = [[None] * N] * T

        # caches[0][i] will be the probability of state [i] emitting sequence[0:0]
        # caches[0][i] = initial prob (i) * emission prob (i -> sequence[0])
        caches[0] = []
        for i in range(0, N):
            caches[0].append(self.init_prob[i] * self.emission_prob[i][sequence[0]])

        # cache[t][j] will be the probability of sequence[0:t] AND ends up in state j
        # cache[t][j] = sum_i(caches[t-1][i] * transition prob (i -> j) * emission prob (j -> sequence[t]))
        #             = sum_i(caches[t-1][i] * transition prob (i -> j)) * emission prob (j -> sequence[t])
        for k in range(1, T):
            caches[k] = []
            for j in range(0, N):
                partial_sum = 0 # probability of sequence[0:t-1] AND transition to state j
                for i in range(0, N):
                    partial_sum += caches[k-1][i] * self.state_prob[i][j]
                caches[k].append(partial_sum * self.emission_prob[j][sequence[k]])

        return sum(caches[T-1])

def vectorize(sequence, emissions):
    """vectorize the sequence using emissions"""
    vector = []
    for i in range(0, len(sequence)):
        vector.append(emissions.index(sequence[i]))
    return vector

# this program calculates the probability of a sequence of mood of a girl
# in singapore based on the weather
if __name__ == '__main__':
    STATES = 'CRS'          # cloudy / rainy / sunny
    EMISSIONS = 'AHS'       # angry / happy / sad
    SEQUENCES = [
        'AA',
        'HH',
        'SS',
        'AH',
        'HA',
        'AS',
        'SA',
        'HS',
        'SH'
    ]

    INIT_PROB = [0.4, 0.5, 0.1]  # singapore rainy seasons
    STATE_PROB = [
        [0.4, 0.4, 0.2],    # cloudy -> 40% cloudy, 40% rainy, 20% sunny
        [0.6, 0.3, 0.1],    # rainy -> 60% cloudy, 30% rainy, 10% sunny
        [0.45, 0.45, 0.1]   # sunny -> 45% cloudy, 45% rainy, 10% sunny
    ]
    EMISSION_PROB = [
        [0.2, 0.2, 0.6],    # cloudy -> 20% angry, 20% happy, 60% sad
        [0.7, 0.05, 0.25],  # rainy -> 70% angry, 5% happy, 25% sad
        [0.05, 0.9, 0.05]   # sunny -> 5% angry, 90% happy, 5% sad
    ]

    model = HMM(INIT_PROB, STATE_PROB, EMISSION_PROB)
    for i in range(len(SEQUENCES)):
        prob = model.forward(vectorize(SEQUENCES[i], EMISSIONS))
        print('probability of sequence %s is %.2f%%' % (SEQUENCES[i], prob * 100))