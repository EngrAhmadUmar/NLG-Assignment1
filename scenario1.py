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

observations = ['Phrase9', 'Phrase1', 'Phrase5']

def forward_algorithm(observations, transition_probs, emission_probs):
    # Initialize forward probabilities
    forward_probs = [{}]

    #initializing the initial probabilities for each state based on the number of states in the transition probabilities.
    #Initialize initial probabilities
    for state in transition_probs.keys():
        forward_probs[0][state] = 1.0 / len(transition_probs)

    
    # Iterate over each observation
    for t, observation in enumerate(observations):
        # Initialize the next step in forward probabilities
        forward_probs.append({})
        
        # Calculate forward probabilities for each state at each time step t
        for next_state in transition_probs.keys():
            #get the total probability of reaching next_state at time t+1 by summing over all possible previous states state.
            total_prob = sum(forward_probs[t][state] * transition_probs[state][next_state] * emission_probs[next_state].get(observation, 0) for state in transition_probs.keys())
            forward_probs[t+1][next_state] = total_prob

    #After iterating over all observations, i compute the total probability of observing the entire sequence of observations by summing the forward probabilities of all states at the last time step.
    # Calculate the total probability of observations
    total_prob = sum(forward_probs[-1][state] for state in transition_probs.keys())
    
    return total_prob

probability = forward_algorithm(observations, transition_probs, emission_probs)

print("Probability of observations {}: {}".format(observations, probability))
