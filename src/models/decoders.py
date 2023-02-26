import copy
import numpy as np
class ViterbiAlgorithm:
    """This file contains the class definition for the ViterbiAlgorithm with just an __init__ function and its only class method,
    best_hidden_state_sequence.
    """    

    def __init__(self, hmm_object):
        """ This is the initialization for a ViterbiAlgorithm object that takes in an HMM object and can compute the best 
        sequence of hidden states (method) from the attributes of the HMM (class). 

        Args:
            hmm_object (class): The HMM object that contains observation states, hidden states, and all probabilities for
            computing the most likely sequence of hidden states with the Viterbi Algorithm.
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """ This is the only method for the ViterbiAlgorithm class that takes the attributes of the HMM object and a user-defined
        sequence of observed states to return the most likely sequence of hidden states. It uses the path variable to keep track
        of possible hidden state sequences at each trellis node and the best_path variable to keep track of the most likely hidden
        state that would lead to the decode_observation_states.

        Args:
            decode_observation_states (np.ndarray): Sequence of states observed by the user from a list of HMM observation states

        Returns:
            np.ndarray: A vector of the most likely sequence of hidden states as computed by the Viterbi Algorithm
        """        
        
        # Redefine our HMM class attributes for easy access
        obs_st = self.hmm_object.observation_states
        obs_st_dict = self.hmm_object.observation_states_dict
        hid_st = self.hmm_object.hidden_states
        hid_st_dict = self.hmm_object.hidden_states_dict
        prior_probs = self.hmm_object.prior_probabilities
        emit_probs = self.hmm_object.emission_probabilities
        trans_probs = self.hmm_object.transition_probabilities

        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        # Initialize the first node of the path as the initial hidden states, initialize a best path to fill out list of hidden states
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]
        best_path = np.zeros(len(decode_observation_states)) 
        
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale
        init_ind = obs_st_dict[decode_observation_states[0]]
        delta = np.multiply(prior_probs, emit_probs.T[init_ind,:])
        #delta = delta / np.sum(delta)
        
        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):

            #
            current_obs = obs_st_dict[decode_observation_states[trellis_node]]
            product_of_delta_and_transition_emission = np.multiply(delta, trans_probs.T)
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission.T, emit_probs[:,current_obs])

            # Select the hidden state sequence with the maximum probability
            # Select probabilities and the indices for the hidden states
            max_hidden_ind = np.argmax(product_of_delta_and_transition_emission, axis=0) 
            max_hidden = np.max(product_of_delta_and_transition_emission, axis=0)
            
            # Keep track of the possible sequence of hidden states for each trellis node in path
            # Update the last index of best_path with the max hidden state from all four options
            path[trellis_node,:] = max_hidden_ind
            best_path[trellis_node - 1] = np.argmax(max_hidden)
            
            # Update delta and scale
            delta = max_hidden
            #delta = delta / np.sum(delta)
            
        # Select the last hidden state, given the best path (i.e., maximum probability)
        # Backtrace through path to choose the most probable hidden states in best_path 
        for state in range(len(decode_observation_states) - 1, 0, -1):
            best_path[state - 1] = path[state, int(best_path[state])]

        # Convert binary to hidden state words using dictionary
        best_hidden_state_path = [hid_st_dict[i] for i in best_path]
        
        return best_hidden_state_path