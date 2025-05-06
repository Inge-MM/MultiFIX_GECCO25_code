# MultiFIX: Interpretable Multimodal AI via Deep Learning and Genetic Programming

This repository contains the code and experiments from our GECCO 2025 paper:

**"A Step towards Interpretable Multimodal AI Models with MultiFIX"**  
by Mafalda Malafaia, Thalea Schlender, Tanja Alderliesten, and Peter A. N. Bosman.

---

## 🧠 About MultiFIX

**MultiFIX** is a multimodal feature engineering and fusion pipeline designed for **interpretable AI**. It combines:
- **Deep Learning (DL)** for modality-specific feature extraction.
- **Genetic Programming (GP-GOMEA)** for interpretable symbolic expressions.
- **Grad-CAM** for visual explanations in image-based models.

MultiFIX supports **tabular + image** data and enforces compact, interpretable features per modality, followed by interpretable fusion models.

---

## 🧪 Key Features
- Intermediate fusion of engineered features from multiple modalities.
- Symbolic replacement of deep learning components using GP-GOMEA.
- Multiple training strategies: end-to-end, sequential, and hybrid.
- Grad-CAM-based visualizations for CNN feature analysis.

---

## 📁 Project Structure
```
├── problems/ # Synthetic problems configurations
├── models/ # To add saved models, including weights for feature engineering blocks
├── dependencies/ # python files needed in the training stage of MultiFIX
│ ├── train.py # training methods
│ ├── architectures.py # DL architectures for image/tabular/fusion
│ ├── dataset.py # Dataset class
│ └── utils.py # Utility functions and helpers
├── scripts/
│ ├── ae.py # train auto-encoder and save weights for image feature engineering block
│ ├── config_file.py # set configurations for experiment, i.e. data_dir, problem to solve, data modalities to use, etc.
│ ├── hpo.py # Train model using grid search for hyper-parameter optimization
│ ├── stats.ipynb # notebook to analyze statistical significance and normal distribution 
│ ├── xai.ipynb # notebook used for inference stage: grad-cam to explain image features and gp to get symbolic expressions 
│                      (script is a proxy that generates gp files and computes the accuracy of the symbolic expression from gp)
│ ├── xai_img.ipynb # notebook used for to better understand image engineered features for multifeature problem
│ └── improve_ae.ipynb # non-essencial: used to visualize reconstructed images using AE
├── hpo_results/ # Output text files with training information
├── env_specs.txt # Env requirements
├── README.md
└── LICENSE
```


---

## 🚀 How to Run

### Install dependencies and activate environment:
$ conda create --name <env> --file <this file>
$ conda activate <env>

### Edit config_file
Choose problem to use, along with device, data_dir, etc.

### Edit corresponding problem_file
In the chosen problem file, define modalities to use and training strategy.
    A_INPUT: 'fusion' (both modalities), 'img' (image data only), or 'tab' (tabular data only)
    B_TRAINING: 'end' (end-to-end training), 'seq' (sequential training), or 'hyb' (hybrid training)
    C_WTS: (useful for 'hyb' and 'seq' training) pre-training weights to load: 'ae' (autoencoder wts for the img block), or 'single' (single modality weights for both feature engineering blocks)
    D_TEMP_FREEZE: (useful for 'seq' training with 'ae' wts): if True, the image block is defrozen after 15 epochs (transitions for 'seq' to 'hyb')
      (should be False for 'end' and 'hyb')

### Run scripts/hpo.py to train the models using 5-CV and GridSearch to optimise LR and WD, and number of features to extract per modality
$ nohup python scripts/hpo.py > end_to_end_and_problem.txt &

### Interpretability for saved model:
Use xai notebook to generate image explanations and gp_file (i.e. files to give to gp). 
Use gp repo (GP-GOMEA for instance) to replace tabular feature engineering block. 
Replace DL tabular features with GP tabular features.
Use gp repo to replace fusion block.
Compute Balanced Accuracy for interpretable model.

## 🤝 Acknowledgments
This research was supported by the Gieskes-Strijbis Fonds and NWO Small Compute Grant on the Dutch National Supercomputer Snellius.

## 📚 Citation
Please cite our paper if you use this code:
@inproceedings{malafaia2025multifix,
  title={A Step towards Interpretable Multimodal AI Models with MultiFIX},
  author={Malafaia, Mafalda and Schlender, Thalea and Alderliesten, Tanja and Bosman, Peter A. N.},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year={2025}
}

## 🛡️ License
A Step towards Interpretable Multimodal AI Models with MultiFIX © 2025 by Mafalda Malafaia, Thalea Schlender, Tanja Alderliesten, and Peter A. N. Bosman is licensed under CC BY-NC-ND 4.0 

