import numpy as np
class HiddenMarkovModel:
    """This is the class definition for the HMM object with no methods, only an __init__ for all object attributes
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """Initialization and defintiion of HMM attributes and dictionaries to store numerical values for each observation and hidden state

        Args:
            observation_states (np.ndarray): set of possible observation states for the HMM
            hidden_states (np.ndarray): set of possible hidden states for the HMM
            prior_probabilities (np.ndarray): initial prior probabilities to get initial hidden state
            transition_probabilities (np.ndarray): square matrix of transition probabilities between hidden states
            emission_probabilities (np.ndarray): square matrix of emission probabilities for each observed state
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities