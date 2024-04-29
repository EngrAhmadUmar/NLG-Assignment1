transition_probs = {
    't1': {'t1': 0.1, 't2': 0.1, 't3': 0.1, 't4': 0.7},
    't2': {'t1': 0.1, 't2': 0.1, 't3': 0.1, 't4': 0.7},
    't3': {'t1': 0.1, 't2': 0.1, 't3': 0.1, 't4': 0.7},
    't4': {'t1': 0.1, 't2': 0.1, 't3': 0.1, 't4': 0.7}
}

emission_probs = {
    't1': {'Phrase1': 0.05, 'Phrase2': 0.05, 'Phrase3': 0.05, 'Phrase4': 0.05, 'Phrase5': 0.05, 'Phrase6': 0.05,
           'Phrase7': 0.05, 'Phrase8': 0.05, 'Phrase9': 0.05, 'Phrase10': 0.05, 'Phrase11': 0.05, 'Phrase12': 0.45},
    't2': {'Phrase1': 0.05, 'Phrase2': 0.45, 'Phrase3': 0.05, 'Phrase4': 0.05, 'Phrase5': 0.05, 'Phrase6': 0.05,
           'Phrase7': 0.05, 'Phrase8': 0.05, 'Phrase9': 0.05, 'Phrase10': 0.05, 'Phrase11': 0.05, 'Phrase12': 0.05},
    't3': {'Phrase1': 0.05, 'Phrase2': 0.05, 'Phrase3': 0.45, 'Phrase4': 0.05, 'Phrase5': 0.05, 'Phrase6': 0.05,
           'Phrase7': 0.05, 'Phrase8': 0.05, 'Phrase9': 0.05, 'Phrase10': 0.05, 'Phrase11': 0.05, 'Phrase12': 0.05},
    't4': {'Phrase1': 0.05, 'Phrase2': 0.05, 'Phrase3': 0.05, 'Phrase4': 0.05, 'Phrase5': 0.05, 'Phrase6': 0.45,
           'Phrase7': 0.05, 'Phrase8': 0.05, 'Phrase9': 0.05, 'Phrase10': 0.05, 'Phrase11': 0.05, 'Phrase12': 0.05}
}

def viterbi_algorithm(observations, states, start_prob, transition_prob, emission_prob):
    # Initialize the Viterbi trellis
    trellis = [{}]
    path = {}

    # Initialize first step
    for state in states:
        trellis[0][state] = start_prob[state] * emission_prob[state][observations[0]]
        path[state] = [state]

    # Iterate over observations
    for t in range(1, len(observations)):
        trellis.append({})
        new_path = {}

        # Calculate the maximum probability and corresponding state for each observation and update path
        for state in states:
            (prob, prev_state) = max((trellis[t-1][prev_state] * transition_prob[prev_state][state] * emission_prob[state][observations[t]], prev_state) for prev_state in states)
            trellis[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    # Get the maximum probability and corresponding path
    (prob, state) = max((trellis[len(observations) - 1][final_state], final_state) for final_state in states)

    return path[state], prob

observations = ['Phrase7', 'Phrase2', 'Phrase12']

states = ['t1', 't2', 't3', 't4']

start_prob = {'t1': 0.6, 't2': 0, 't3': 0.1, 't4': 0.3}

best_path, probability = viterbi_algorithm(observations, states, start_prob, transition_probs, emission_probs)

# Format the output
output = ' → '.join(best_path) + ', ' + str(probability)

print([output])
