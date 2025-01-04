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

This repository includes examples of differently parameterized UMNNs to replicate results described in the AAAI'25 submission. For detailed experiment configurations (e.g., regularization values, specific UMNN architectures) in our paper, refer to the **technical appendix** in arxiv.


### Running an Experiment
To execute an experiment, follow these steps:

1. **Ensure Configuration Files Are Correct**:
   - Modify the paths in the `config.yaml` file to point to the correct dataset files (`training_file`, `validation_file`, `testing_file`) and desired output directory (`results_path`).
   - Each experiment's script (e.g., `run_butterfly.py`) reads the paths directly from `config.yaml`.

2. **Run the Experiment Script**:
   - Navigate to the main project directory.
   - Run the experiment script using:
     ```bash
     python -m experiments.butterfly.run_butterfly
     ```
   - Replace `butterfly` with the appropriate experiment name for other experiments (e.g., `gaussian`, `ovarian`).

3. **Check Results**:
   - Results, including the precision matrix, model files, and plots, will be saved in a uniquely indexed folder within the `results/` directory (e.g., `results/BF0/` for the first experiment run).
   - The experiment configuration used for the run is also logged in `log.txt` within the results folder.

---

### Modifying Configurations
The `config.yaml` file allows you to customize parameters for each experiment:

- **Dataset Paths**:
  - Update `training_file`, `validation_file`, and `testing_file` to point to your generated `.txt` or `.csv` dataset files.

- **UMNN Parameters**:
  - Adjust `hidden_layers`, `num_steps`, and other model-related settings as needed.

- **Regularization and Training**:
  - Modify `regularizations`, `learning_rate`, and `max_epochs` to explore alternative setups.

---

### Checking Experiment Logs
After running an experiment, the corresponding `log.txt` file in the results directory will include:

- The configuration used for the run.
- Paths to the saved precision matrix, plots, and models.


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
