# L-SING
Learning local neighborhoods of non-Gaussian graphical models  
**Accepted at AAAI'25**

---

## Overview
L-SING (Learning Local Neighborhoods of Non-Gaussian Graphical Models) is a novel approach for modeling non-Gaussian graphical structures. This repository includes the codebase and experiment scripts used for our AAAI'25 submission.

---

## Experiments
We conducted three primary experiments to evaluate the performance and versatility of L-SING:

1. **Butterfly Distribution**  
   A synthetic experiment showcasing L-SING's ability to learn complex distributions with localized dependencies.

2. **Gaussian Distribution**  
   - Pre-processing and generation of training, validation, and test datasets are included.
   - Demonstrates L-SING's compatibility with Gaussian distributions for benchmarking purposes.

3. **Ovarian Cancer Dataset**  
   - Derived from the `curatedOvarianPackage` in R.
   - The dataset was pre-processed following the methodology outlined by Shutta et al. (2022) for direct comparison between GLASSO and localized L-SING methods.

---

## Usage Instructions
To execute the experiments:

1. Verify that all paths in the configuration files are correctly set.
2. Navigate to the respective folders (`butterfly/`, `gaussian/`, or `ovarian/`) and execute the experiment scripts provided in each directory. Paths are relative to the folder structure.
3. Modify UMNN parameters (e.g., hidden layers, steps) as needed to explore alternative configurations.

This repository includes examples of differently parameterized UMNNs to replicate results described in the AAAI'25 submission in prev_exp.

For detailed experiment configurations (e.g., regularization values, specific UMNN architectures), refer to the **technical appendix** in arxiv.


---

## Citations
If you use this code, please cite the following references:

1. **Wehenkel, A., & Louppe, G. (2019).**  
   *Unconstrained Monotonic Neural Networks.*  
   In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d'Alché-Buc, F.; Fox, E.; and Garnett, R., eds., *Advances in Neural Information Processing Systems,* Vol. 32. Curran Associates, Inc.

2. **Shutta, K. H., Vito, R. D., Scholtens, D. M., & Balasubramanian, R. (2022).**  
   *Gaussian Graphical Models with Applications to Omics Analyses.*  
   *Statistics in Medicine,* 41(25), 5150–5187.

---

## Acknowledgments
The UMNN model implementation in this repository is adapted from the official source:  
[https://github.com/AWehenkel/UMNN](https://github.com/AWehenkel/UMNN).

For further inquiries or clarifications, please refer to our **technical appendix** or contact the authors.
