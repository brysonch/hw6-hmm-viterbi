"""
UCSF BMI203: Biocomputing Algorithms
Author: Bryson Choy
Date: 02/24/23
Program: BMI
Description: This test file contains an HMM example that tests the hypothesis whether the cCRE selection strategy for progenitor/primitive cardiomyocytes
(hidden state) dictates the regulatory activity of the TBX5 TADs in cardiomyocytes.

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

def test_deliverable():
    """_summary_
    """
    # index annotation observation_states=[i,j] 
    observation_states = ['regulatory', 'regulatory-potential'] # observed regulatory activity in the TBX5 TAD of cardiomyocytes
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['encode-atac', 'atac'] # In order of the two cCRE selection strategies (encode_atac, atac)

    # Import the HMM input data for progenitor cardiomyocytes (prefix: prog_cm)
    prog_cm_data = np.load('./data/ProjectDeliverable-ProgenitorCMs.npz')

    # Instantiate submodule class models.HiddenMarkovModel with progenitor cardiomyocytes
    # observation and hidden states and prior, transition, and emission probabilities.
    prog_cm_hmm_object = HiddenMarkovModel(observation_states,
                                           hidden_states,
                                           prog_cm_data['prior_probabilities'], #  prior probabilities of hidden states in the order specified in the hidden_states list
                                           prog_cm_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                                           prog_cm_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prog_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states (i.e., CRE selection strategy) for the progenitor CMs and evaluate the model performace
    evaluate_viterbi_decoder_using_observation_states_of_prog_cm = prog_cm_viterbi_instance.best_hidden_state_sequence(prog_cm_data['observation_states'])
    
    # Evaluate the accuracy of using the progenitor cardiomyocyte HMM and Viterbi algorithm to decode the progenitor CM's CRE selection strategies
    # NOTE: Model is expected to perform with 80% accuracy
    assert np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm)/len(prog_cm_data['observation_states']) == 0.6

    ### Evaluate Primitive Cardiomyocyte Regulatory Observation Sequence ###
    # Import primitive cardiomyocyte data (prefix: prim_cm)
    prim_cm_data = np.load('./data/ProjectDeliverable-PrimitiveCMs.npz')

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prim_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states of the primitive cardiomyocyte's regulatory observation states
    decoded_hidden_states_for_observed_states_of_prim_cm = prim_cm_viterbi_instance.best_hidden_state_sequence(prim_cm_data['observation_states'])
    assert np.sum(prim_cm_data['hidden_states'] == decoded_hidden_states_for_observed_states_of_prim_cm)/len(prim_cm_data['observation_states']) == 0.6

    # Assertion for the dimensions of the HMM attributes
    _test_dims(observation_states, hidden_states, prog_cm_hmm_object.prior_probabilities, prog_cm_hmm_object.transition_probabilities, 
        prog_cm_hmm_object.emission_probabilities, prim_cm_data['observation_states'], prim_cm_data['hidden_states'])
