"""
UCSF BMI203: Biocomputing Algorithms
Author: Bryson Choy
Date: 02/24/23
Program: BMI
Description: This test file contains 3 use cases for the ViterbiAlgorithm with tests for 3 different hypotheses: whether rotation lab funding dictates 
the dedication of a graduate student, whether practicing dictates the likelihood of the Golden State Warriors winning, and whether the location of a
grad student's lab affects the happiness of the student.
"""
import pytest
import numpy as np
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm

def _test_dims(obs: np.ndarray, hid: np.ndarray, priors: np.ndarray, trans: np.ndarray, emits: np.ndarray, obs_seq: np.ndarray, hid_seq: np.ndarray):
    assert len(obs) == len(hid)
    assert priors.shape[0] == len(obs)
    assert trans.shape == emits.shape
    assert obs_seq.shape == hid_seq.shape

    assert np.sum(priors) == 1
    assert np.allclose(np.sum(trans, axis=1), np.ones((trans.shape[0])))
    assert np.allclose(np.sum(emits, axis=1), np.ones((emits.shape[0])))
    assert len(np.unique(obs_seq)) == len(obs)
    assert len(np.unique(hid_seq)) == len(hid)


def test_use_case_lecture():
    """We test the hypothesis whether a grad student's dedication to their rotation lab (observation state) is dependent on the NIH funding source
    of the student's rotation project (hidden state). We test whether the attributes from the Viterbi HMM inheritance match the HMM class attributes,
    whether the dimensions of the HMM attributes match those of the Viterbi HMM inheritance, and whether the ViterbiAlgorithm best_hidden_state_sequence
    method finds the correct sequence of hidden states.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])

    # Check HMM dimensions and ViterbiAlgorithm
    _test_dims(observation_states, hidden_states, use_case_one_hmm.prior_probabilities, use_case_one_hmm.transition_probabilities, 
        use_case_one_hmm.emission_probabilities, use_case_one_data['observation_states'], use_case_one_data['hidden_states'])
    _test_dims(observation_states, hidden_states, use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_viterbi.hmm_object.transition_probabilities, 
        use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_data['observation_states'], use_case_one_data['hidden_states'])


def test_user_case_one():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    #assert 
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])

    # Check HMM dimensions and ViterbiAlgorithm
    _test_dims(observation_states, hidden_states, use_case_one_hmm.prior_probabilities, use_case_one_hmm.transition_probabilities, 
        use_case_one_hmm.emission_probabilities, use_case_one_data['observation_states'], use_case_one_data['hidden_states'])
    _test_dims(observation_states, hidden_states, use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_viterbi.hmm_object.transition_probabilities, 
        use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_data['observation_states'], use_case_one_data['hidden_states'])


def test_user_case_two():
    """We test the hypothesis whether the Golden State Warriors winning their games (observation state) is dependent on the team practicing (hidden state). 
    We test whether the attributes from the Viterbi HMM inheritance match the HMM class attributes, whether the dimensions of the HMM attributes match those 
    of the Viterbi HMM inheritance, and whether the ViterbiAlgorithm best_hidden_state_sequence method finds the correct sequence of hidden states.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['Warriors-win','Warriors-lose'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['practice','no-practice']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_two_priors = np.array([0.3, 0.7])
    use_case_two_transitions = np.array([[0.9, 0.1],
                                        [0.2, 0.8]])
    use_case_two_emissions = np.array([[0.95, 0.05],
                                        [0.15, 0.85]])

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_two_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_two_priors, # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_two_transitions, # transition_probabilities[:,hidden_states[i]]
                      use_case_two_emissions) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_two_viterbi = ViterbiAlgorithm(use_case_two_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_two_viterbi.hmm_object.observation_states == use_case_two_hmm.observation_states
    assert use_case_two_viterbi.hmm_object.hidden_states == use_case_two_hmm.hidden_states

    assert np.allclose(use_case_two_viterbi.hmm_object.prior_probabilities, use_case_two_hmm.prior_probabilities)
    assert np.allclose(use_case_two_viterbi.hmm_object.transition_probabilities, use_case_two_hmm.transition_probabilities)
    assert np.allclose(use_case_two_viterbi.hmm_object.emission_probabilities, use_case_two_hmm.emission_probabilities)

    # Find the best hidden state path for our observation states
    seq_observed_states = np.array(['Warriors-lose', 'Warriors-lose', 'Warriors-lose', 'Warriors-win', 'Warriors-lose', 'Warriors-win'])
    seq_hidden_states = np.array(['no-practice', 'no-practice', 'no-practice', 'no-practice', 'no-practice', 'practice'])
    use_case_decoded_hidden_states = use_case_two_viterbi.best_hidden_state_sequence(seq_observed_states)
    assert np.alltrue(use_case_decoded_hidden_states == seq_hidden_states)

    # Check HMM dimensions and ViterbiAlgorithm
    _test_dims(observation_states, hidden_states, use_case_two_hmm.prior_probabilities, use_case_two_hmm.transition_probabilities, 
        use_case_two_hmm.emission_probabilities, seq_observed_states, seq_hidden_states)
    _test_dims(observation_states, hidden_states, use_case_two_viterbi.hmm_object.prior_probabilities, use_case_two_viterbi.hmm_object.transition_probabilities, 
        use_case_two_viterbi.hmm_object.emission_probabilities, seq_observed_states, seq_hidden_states)


def test_user_case_three():
    """We test the hypothesis whether a grad student's happiness/mood (observation state) is dependent on whether the grad student's thesis lab is located
    at UCSF's Mission Bay or Parnassus campus (hidden state). We test whether the attributes from the Viterbi HMM inheritance match the HMM class attributes,
    whether the dimensions of the HMM attributes match those of the Viterbi HMM inheritance, and whether the ViterbiAlgorithm best_hidden_state_sequence
    method finds the correct sequence of hidden states.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['happy','sad'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['Mission-Bay','Parnassus']

    use_case_three_priors = np.array([0.4, 0.6])
    use_case_three_transitions = np.array([[0.8, 0.2],
                                        [0.2, 0.8]])
    use_case_three_emissions = np.array([[0.9, 0.1],
                                        [0.15, 0.85]])

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_three_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_three_priors, # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_three_transitions, # transition_probabilities[:,hidden_states[i]]
                      use_case_three_emissions) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_three_viterbi = ViterbiAlgorithm(use_case_three_hmm)

     # Check if use case three HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_three_viterbi.hmm_object.observation_states == use_case_three_hmm.observation_states
    assert use_case_three_viterbi.hmm_object.hidden_states == use_case_three_hmm.hidden_states

    assert np.allclose(use_case_three_viterbi.hmm_object.prior_probabilities, use_case_three_hmm.prior_probabilities)
    assert np.allclose(use_case_three_viterbi.hmm_object.transition_probabilities, use_case_three_hmm.transition_probabilities)
    assert np.allclose(use_case_three_viterbi.hmm_object.emission_probabilities, use_case_three_hmm.emission_probabilities)

    # Find the best hidden state path for our observation states
    seq_observed_states = np.array(['happy', 'sad', 'sad', 'happy', 'happy', 'sad'])
    seq_hidden_states = np.array(['Mission-Bay', 'Parnassus', 'Parnassus', 'Mission-Bay', 'Mission-Bay', 'Mission-Bay'])
    use_case_decoded_hidden_states = use_case_three_viterbi.best_hidden_state_sequence(seq_observed_states)
    assert np.alltrue(use_case_decoded_hidden_states == seq_hidden_states)

    # Check HMM dimensions and ViterbiAlgorithm
    _test_dims(observation_states, hidden_states, use_case_three_hmm.prior_probabilities, use_case_three_hmm.transition_probabilities, 
        use_case_three_hmm.emission_probabilities, seq_observed_states, seq_hidden_states)
    _test_dims(observation_states, hidden_states, use_case_three_viterbi.hmm_object.prior_probabilities, use_case_three_viterbi.hmm_object.transition_probabilities, 
        use_case_three_viterbi.hmm_object.emission_probabilities, seq_observed_states, seq_hidden_states)
