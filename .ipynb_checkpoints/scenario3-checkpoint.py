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

import matplotlib.pyplot as plt

# Define function to visualize transition probabilities
def visualize_transition_probs(transition_probs, title):
    plt.figure(figsize=(8, 6))
    states = list(transition_probs.keys())
    num_states = len(states)
    plt.imshow([[transition_probs[from_state][to_state] for to_state in states] for from_state in states], cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Probability')
    plt.xticks(range(num_states), states, rotation=45)
    plt.yticks(range(num_states), states)
    plt.title(title)
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.show()

# Define function to visualize emission probabilities
def visualize_emission_probs(emission_probs, title):
    plt.figure(figsize=(10, 6))
    states = list(emission_probs.keys())
    num_states = len(states)
    num_observations = len(list(emission_probs.values())[0])
    for i, (state, probs) in enumerate(emission_probs.items()):
        plt.subplot(num_states, 1, i+1)
        plt.bar(range(num_observations), list(probs.values()), align='center')
        plt.title('State: ' + state)
        plt.xlabel('Observations')
        plt.ylabel('Probability')
        plt.xticks(range(num_observations), list(probs.keys()), rotation=45)
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

# Implement the forward-backward algorithm for parameter learning
def forward_backward_algorithm(observations, transition_probs, emission_probs, max_iter=100, tolerance=1e-6):
    num_states = len(transition_probs)
    num_observations = len(observations)
    old_transition_probs = transition_probs
    old_emission_probs = emission_probs

    for _ in range(max_iter):
        # Forward pass
        forward_probs = [{} for _ in range(num_observations)]
        scaling_factors = [0] * num_observations
        for t in range(num_observations):
            if t == 0:
                for state in transition_probs.keys():
                    forward_probs[t][state] = emission_probs[state].get(observations[t][0], 0) * 1/num_states
                    scaling_factors[t] += forward_probs[t][state]
            else:
                for state in transition_probs.keys():
                    forward_probs[t][state] = sum(forward_probs[t-1][prev_state] * transition_probs[prev_state].get(state, 0) * emission_probs[state].get(observations[t][0], 0) for prev_state in transition_probs.keys())
                    scaling_factors[t] += forward_probs[t][state]
            scaling_factors[t] = 1 / scaling_factors[t]
            for state in transition_probs.keys():
                forward_probs[t][state] *= scaling_factors[t]

        # Backward pass
        backward_probs = [{} for _ in range(num_observations)]
        for t in reversed(range(num_observations)):
            if t == num_observations - 1:
                for state in transition_probs.keys():
                    backward_probs[t][state] = 1
            else:
                for state in transition_probs.keys():
                    backward_probs[t][state] = sum(transition_probs[state].get(next_state, 0) * emission_probs[next_state].get(observations[t+1][0], 0) * backward_probs[t+1][next_state] for next_state in transition_probs.keys())
                    backward_probs[t][state] *= scaling_factors[t+1]

        # Update transition probabilities
        new_transition_probs = {}
        for state in transition_probs.keys():
            new_transition_probs[state] = {}
            for next_state in transition_probs[state].keys():
                numerator = sum(forward_probs[t][state] * transition_probs[state][next_state] * emission_probs[next_state].get(observations[t+1][0], 0) * backward_probs[t+1][next_state] for t in range(num_observations-1))
                denominator = sum(forward_probs[t][state] * backward_probs[t][state] for t in range(num_observations))
                new_transition_probs[state][next_state] = numerator / denominator

        # Update emission probabilities
        new_emission_probs = {}
        for state in emission_probs.keys():
            new_emission_probs[state] = {}
            for observation in emission_probs[state].keys():
                numerator = sum(forward_probs[t][state] * backward_probs[t][state] if observations[t][0] == observation else 0 for t in range(num_observations))
                denominator = sum(forward_probs[t][state] * backward_probs[t][state] for t in range(num_observations))
                new_emission_probs[state][observation] = numerator / denominator

        # Check for convergence
        transition_diff = sum(abs(new_transition_probs[state][next_state] - old_transition_probs[state][next_state]) for state in transition_probs.keys() for next_state in transition_probs[state].keys())
        emission_diff = sum(abs(new_emission_probs[state][observation] - old_emission_probs[state][observation]) for state in emission_probs.keys() for observation in emission_probs[state].keys())
        if transition_diff < tolerance and emission_diff < tolerance:
            break

        # Update old probabilities
        old_transition_probs = new_transition_probs
        old_emission_probs = new_emission_probs

    return new_transition_probs, new_emission_probs

observations = [['Phrase1', 'Phrase2', 'Phrase3'], ['Phrase4', 'Phrase5', 'Phrase6'], ['Phrase7', 'Phrase8', 'Phrase9'], ['Phrase10', 'Phrase11', 'Phrase12']]



visualize_transition_probs(transition_probs, 'Transition Probabilities Before Learning')

visualize_emission_probs(emission_probs, 'Emission Probabilities Before Learning')

learned_transition_probs, learned_emission_probs = forward_backward_algorithm(observations, transition_probs, emission_probs)

visualize_transition_probs(learned_transition_probs, 'Transition Probabilities After Learning')

visualize_emission_probs(learned_emission_probs, 'Emission Probabilities After Learning')


print("Learned Transition Probabilities:")
for state, probs in learned_transition_probs.items():
    print(state + ":", probs)

print("\nLearned Emission Probabilities:")
for state, probs in learned_emission_probs.items():
    print(state + ":", probs)