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

    the state_prob is a NxN whose each cell a[i, j] is the probability of transition
    from state i-th to state j-th.

    the emissions_prob is a NxM matrix whose each cell a[i, j] is the probability
    of emitting value j-th from state i-th
    """

    def __init__(self, init_prob=None, state_prob=None, emission_prob=None):
        self.init_prob = init_prob
        self.state_prob = state_prob
        self.emission_prob = emission_prob

    def forward(self, seq):
        """
        calculate the probability that the sequence is generated by this HMM
        it will only works if initial_state_prob, state_transitions and
        observation_transitions are not None.
        """
        N = len(self.state_prob)
        T = len(seq)

        caches = [None for _ in range(T)]

        # caches[0][i] will be the probability of state [i] emitting sequence[0:0]
        # caches[0][i] = initial prob (i) * emission prob (i -> sequence[0])
        caches[0] = [self.init_prob[i] * self.emission_prob[i][seq[0]] for i in range(N)]

        # cache[t][j] will be the probability of sequence[0:t] AND ends up in state j
        # cache[t][j] = sum_i(caches[t-1][i] * transition prob (i -> j) * emission prob (j -> sequence[t]))
        #             = sum_i(caches[t-1][i] * transition prob (i -> j)) * emission prob (j -> sequence[t])
        for t in range(1, T):
            caches[t] = [0 for _ in range(N)]
            for j in range(N):
                partial_sum = sum(caches[t-1][i] * self.state_prob[i][j] for i in range(N))
                caches[t][j] = partial_sum * self.emission_prob[j][seq[t]]

        return sum(caches[T-1])

    def backward(self, seq):
        """
        same as forward but calculating the probability in backward
        """
        N = len(self.state_prob)
        T = len(seq)

        caches = [None for _ in range(T)]

        caches[T-1] = [1 for _ in range(N)]

        for t in reversed(range(T-1)):
            caches[t] = [0 for _ in range(N)]
            for i in range(N):
                caches[t][i] = sum(caches[t+1][j] * self.state_prob[i][j] * self.emission_prob[j][seq[t+1]] for j in range(N))

        return sum(self.init_prob[i] * caches[0][i] * self.emission_prob[i][seq[0]] for i in range(N))

    def viterbi(self, seq):
        """
        find the sequence of states that maximum the likelihood of observed sequence
        """
        N = len(self.state_prob)
        T = len(seq)

        caches = [None for _ in range(T)]
        tracks = [None for _ in range(T)]

        caches[0] = [self.init_prob[i] * self.emission_prob[i][seq[0]] for i in range(N)]
        tracks[0] = [0 for _ in range(N)]

        import operator

        for t in range(1, T):
            caches[t] = [0 for _ in range(N)]
            tracks[t] = [0 for _ in range(N)]
            for j in range(N):
                # the probability of seq[0:t-1] and ends up in state j
                partial_sum = [caches[t-1][i] * self.state_prob[i][j] for i in range(N)]

                max_idx, max_prob = max(enumerate(partial_sum), key=operator.itemgetter(1))
                caches[t][j] = max_prob * self.emission_prob[j][seq[t]]

                # the state that maximize the probability above
                tracks[t][j] = max_idx

        max_track = [0 for _ in range(T)]
        max_track[T-1], _ = max(enumerate(caches[T-1]), key=operator.itemgetter(1))
        for t in range(T-1):
            max_track[t] = tracks[t+1][max_track[t+1]]
        return max_track

def vectorize(sequence, emissions):
    """vectorize the sequence using emissions"""
    vector = []
    for seq in sequence:
        vector.append(emissions.index(seq))
    return vector

def track_to_state(track, states):
    """convert track to human readable states"""
    return ' -> '.join([states[i] for i in track])

def main():
    """
    this program calculates the probability of a sequence of mood of an amateur astronomer
    in singapore based on the weather
    """
    states = ['cloudy', 'rainy', 'sunny']
    emissions = 'AHS'       # angry / happy / sad
    seq_len = 3             # define the length of sequence

    init_prob = [0.35, 0.55, 0.1]  # singapore rainy seasons, 35% cloudy, 55% sunny, 10% sunny
    state_prob = [
        [0.4, 0.4, 0.2],    # cloudy -> 40% cloudy, 40% rainy, 20% sunny
        [0.6, 0.3, 0.1],    # rainy -> 60% cloudy, 30% rainy, 10% sunny
        [0.55, 0.35, 0.15]   # sunny -> 45% cloudy, 45% rainy, 10% sunny
    ]
    emission_prob = [
        [0.2, 0.2, 0.6],    # cloudy -> 20% angry, 20% happy, 60% sad
        [0.7, 0.05, 0.25],  # rainy -> 70% angry, 5% happy, 25% sad
        [0.05, 0.9, 0.05]   # sunny -> 5% angry, 90% happy, 5% sad
    ]

    import itertools
    sequences = [''.join(seq) for seq in itertools.product(emissions, repeat=seq_len)]

    model = HMM(init_prob, state_prob, emission_prob)
    for seq in sequences:
        vectorized_seq = vectorize(seq, emissions)
        prob = model.forward(vectorized_seq)
        print('probability of sequence %s is %.2f%%' % (seq, prob * 100))
        prob = model.backward(vectorized_seq)
        print('probability of sequence %s calculate in backward is %.2f%%' % (seq, prob * 100))
        max_track = model.viterbi(vectorized_seq)
        print('the most probable sequence of state is %s' % track_to_state(max_track, states))
        print()

if __name__ == '__main__':
    main()
