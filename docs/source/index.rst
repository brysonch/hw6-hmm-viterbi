.. hw6-hmm documentation master file, created by
   sphinx-quickstart on Sat Feb 11 16:27:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm
============================================================================================================================================

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations.

**Project Deliverable**: Produce a simple report for functional characterization inferred from a binary regulatory observation state pattern across cardiac developmental timepoints.

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

Key Words
==========
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5


Functional Characterization Report
===================================

Please evaluate the project deliverable and briefly answer the following speculative question, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.

1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?

The size of the sliding window determines the resolution of the HMM's regulatory observations dependent on the selection strategy hidden states. A smaller sliding window may be computationally intractable, but may also offer more accurate and robust transition probabilities between hidden states, as chromatin accessibility may vary at a local scale. A larger sliding window may do the opposite, that is, reduce the resolution and accuracy of the HMM's parameters for determining regulatory state from hidden states. In both cases, there may be a sweet spot that balances between smaller (computationally infeasible) and larger (lower resolution) sliding windows, as the user may also need to consider the length of accessible chromatin in this particular TAD.

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?

Histone ChIP data for silencing and upregulation of transcription (e.g. H3K9me3 and H3K9ac), as well as TF ChIP or Hi-C data may provide additional information on the rules for chromatin accessibility in progenitor cardiomyocytes. For example, we can update our hypothesis to be: can we predict chromatin accessibility in progenitor cardiomyocytes from active and repressive histone marks? The observation states would be constituitively silent heterochromatin and active chromatin, and the hidden states would be active and repressive histone marks. The probability of being in either an active or repressive histone state in the TBX5 TAD would be the prior probability, and the likelihood of transition between histone marks would be the transition probabilities (very low probability of transitioning out of the current state, either active or repressed). The emission probabilities would be the probability of a particular histone state dictating the accessibility of chromatin at that mark, with accessible chromatin likely to associate with active histone marks and vice versa. This hypothesis can also be applied to TF ChIP (how is chromatin accessibility dictated by the presence of TF binding) and Hi-C (how it chromatin accessibility dictated by DNA looping or TAD formation).

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

After functional characterization assays, it would be possible to distinguish between cardiomyocyte-specific cCREs and CREs with less cell-type specificity over gene regulatory activity. A high-throughput CRISPRi/a screen can, for example, provide a small-scale view of the perturbations and noncoding DNA necessary for transcriptional regulation in cardiomyoctyes. Testing these assays in progenitor and primitive cardiomyocytes and in other primary cells should reveal the key regulatory elements for dicatating chromatin accessibility in these cardiomyocytes (cCREs) and annotations may be updated accordingly, while other regulatory elements with broader cell type activity may be labeled CREs.


Models Package 
======================
.. toctree::
   :maxdepth: 2
   
   modules
