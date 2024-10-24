import numpy as np
def softmax(action_values, tau=1.0):
    preferences = action_values/tau
    max_preference = np.max(preferences)
    exp_preferences = np.exp(preferences - max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences)
    action_probs = exp_preferences / sum_of_exp_preferences
    return action_probs