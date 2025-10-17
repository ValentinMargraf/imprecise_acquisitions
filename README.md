# Repository for Experimental Results

This repository contains the code and data used for the experiments reported in paper.  
All identifying information (e.g., author names, affiliations, and institution-specific details) has been removed to preserve anonymity.

---

## 🔬 Overview

This repository includes implementations, experiment scripts, and evaluation routines for the methods described in the paper.  
The experiments can be reproduced using the provided configurations and instructions below.

---

## ⚙️ Requirements

- Python >= 3.10  
- Required dependencies are given in the ```requirements.txt``` file in the repository.  

You can install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

To reproduce the main experiments:
```bash
python experiments/bbob_experimenter.py 
```

The results will be stored in a database using the py_experimenter. To this end, a database connection needs to be specified in the config files in 
```
experiments/configs
```
For further information, please consider the py_experimenter documentation https://tornede.github.io/py_experimenter/

---

## 📈 Evaluation

To evaluate the methods and create the plots from the paper, use:
```bash
python experiments/plot_results_ucb.py
```

Evaluation metrics and visualizations will be stored in ```../figs```.

---

## 🔒 Anonymization Notes

- All references to institutions, datasets with identifiable origins, or personal identifiers have been removed or replaced with placeholders.  
- This repository is intended solely for **anonymous peer review** and will be updated with attribution upon acceptance.

---

