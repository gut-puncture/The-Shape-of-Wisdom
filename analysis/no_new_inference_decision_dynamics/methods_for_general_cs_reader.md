# Methods For A General CS Reader

The data are cached layer-by-layer scores for the four answer options A, B, C, and D on 3000 MMLU prompts for each of three instruction-tuned transformer models. At each logged layer we only observe the current four-option readout, not the hidden state itself.

`delta_soft` is the correct-option score minus the log-sum-exp of the three incorrect-option scores. It removes a discrete artifact of the hard margin, which can jump when the identity of the strongest incorrect option changes. In this four-option readout, `delta_soft` and `p_correct` are exact monotone transforms of one another: `p_correct = sigmoid(delta_soft)`.

`p_correct` is the current probability of the correct option inside the four-choice readout. It is not the probability of the full natural-language completion over the model vocabulary.

The commitment coordinate `a` is a linear contrast between the correct score and the average incorrect score after centering the four-score vector. The competitor-dispersion coordinate `q` measures how much the three incorrect options still differ from one another. The angle `theta` says which incorrect-answer direction is currently strongest inside that incorrect-option plane.

`future_flip` asks a purely empirical question: given the current readout state, does the sign of the soft margin ever change again later in depth? The empirical commitment depth is the first layer after which the trajectory stays in a region of low future-flip probability.

Everything in this package stays in answer-readout space. It does not identify hidden-state computation, internal pathways, or causality.
