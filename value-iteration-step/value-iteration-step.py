import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values = np.array(values)
    transitions = np.array(transitions)
    rewards = np.array(rewards)

    expected = transitions @  values

    q_values = rewards + gamma * expected

    new_values = np.max(q_values,axis=1)

    return new_values.tolist()