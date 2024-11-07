# Tox21 QSAR

# About
This tox_21_qsar repository contains cheminformatics and machine/deep learning code for descriptive analysis and QSAR modelling for tox_21 datasets.

Several predictive models were built using various classical machine learning and deep learning algorithms, such as:

1. [Random Forests](./notebooks/nr_ahr/tox_21_nr_ahr_modelling.ipynb)
2. [XGBoost](./notebooks/sr_p53/tox_21_sr_p53_modelling.ipynb)
3. [Feed Forward Neural Networks (FFNs)](./notebooks/nr_ahr/tox_21_nr_ahr_w_ffns.ipynb)
4. [Long Short-Term Memory Neural Networks (LSTMs)](./notebooks/nr_ahr/tox_21_nr_ahr_w_lstms.ipynb)
5. [Graph Convolutional Networks (GCNs)](./notebooks/nr_ahr/tox_21_nr_ahr_w_gcn.ipynb)
6. [Meta Classifiers](./notebooks/sr_p53/tox_21_sr_p53_modelling.ipynb), including hard and soft voting.


The projects explore various data splitting algorithms, such as [MaxMin (diversity) Picking](https://rdkit.blogspot.com/2017/11/revisting-maxminpicker.html) using various fingerprints, as well as various hyperparameter tuning approaches such as [Grid CV] (https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html) and [Randomized search](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
 
Several methods are also used for explainability, including [SHAP](https://shap.readthedocs.io/en/latest/) and [LIME](https://c3.ai/glossary/data-science/lime-local-interpretable-model-agnostic-explanations/).





