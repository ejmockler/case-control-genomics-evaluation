# Modeling case/control genomics

Model stacking workflow to evaluate genomic variants that distinguish cases from controls.

Development occurs with an interactive environment, so each workflow step can be easily debugged.

## Tools

- Python >=3.9
- Workflow: [Prefect](https://www.prefect.io)
- Modeling: [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/en/stable/)
- Conditional feature explanation: [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

## Entrypoint

[notebook/ALS_genomics_model_stack.ipynb](notebook/ALS_genomics_model_stack.ipynb)
