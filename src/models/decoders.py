import copy
import numpy as np
class ViterbiAlgorithm:
    """_summary_
    """    

    def __init__(self, hmm_object):
        """_summary_

        Args:
            hmm_object (_type_): _description_
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            decode_observation_states (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """        
        
        obs_st = self.hmm_object.observation_states
        obs_st_dict = self.hmm_object.observation_states_dict
        hid_st = self.hmm_object.hidden_states
        hid_st_dict = self.hmm_object.hidden_states_dict
        prior_probs = self.hmm_object.prior_probabilities
        emit_probs = self.hmm_object.emission_probabilities
        trans_probs = self.hmm_object.transition_probabilities

        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
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
        #best_path[0] = path[0, np.argmax(delta)]
        
        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):

            # TODO: comment the initialization, recursion, and termination steps
    
            current_obs = obs_st_dict[decode_observation_states[trellis_node]]
            #print("delt: ", delta)
            #print("trans: ", trans_probs.T)
            #print("emit: ", emit_probs[:,current_obs])

            product_of_delta_and_transition_emission = np.multiply(delta, trans_probs.T)
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission.T, emit_probs[:,current_obs])
            #print("prod: ", product_of_delta_and_transition_emission)

            # Select the hidden state sequence with the maximum probability
            max_hidden_ind = np.argmax(product_of_delta_and_transition_emission, axis=0) 
            max_hidden = np.max(product_of_delta_and_transition_emission, axis=0)
            #print("prod: ", product_of_delta_and_transition_emission)
            
            path[trellis_node,:] = max_hidden_ind
            #print("new path: ", path[trellis_node,:])
            
            #print("max hidden: ", max_hidden)
            best_path[trellis_node - 1] = np.argmax(max_hidden)
            #print("best path: ", best_path)

            # Update best path
            #for hidden_state in range(len(self.hmm_object.hidden_states)):
            #    hid_st_dict[hidden_state]
            #    np.append(best_hidden_state_path, hidden_s)
                #path.append(max_hidden_ind)
            #    best_path[trellis_node] = max_hidden_ind
            #    print("trellis: ", path[trellis_node - 1, max_hidden_ind])
            #    best_path[trellis_node - 1] = path[trellis_node - 1, max_hidden_ind]
                
            
            # Update delta and scale
            delta = max_hidden
            
            #delta = np.multiply(max_hidden, emit_probs[current_obs,:])
            #delta = delta / np.sum(delta)
            
            # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path

        # Select the last hidden state, given the best path (i.e., maximum probability)
        #best_hidden_state_path = np.array([])
        #print("final path: ", path)
        for state in range(len(decode_observation_states) - 1, 0, -1):
            best_path[state - 1] = path[state, int(best_path[state])]
        best_hidden_state_path = [hid_st_dict[i] for i in best_path]
        
        return best_hidden_state_path