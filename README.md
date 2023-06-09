# Modeling case/control genomics

Model stacking workflow to evaluate genomic variants that distinguish cases from controls.

Development occurs with an interactive environment, so each workflow step can be easily debugged.

## Tools

- Python >=3.9
- Workflow: [Prefect](https://www.prefect.io)
- Modeling: [scikit-learn](https://scikit-learn.org/), [xgboost](https://xgboost.readthedocs.io/en/stable/)
- Conditional feature explanation: [SHAP](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html)

## Scaling

Prefect uses an external database to manage workflow metadata. To properly scale bootstrap sampling across multiple models, set up a Postgres database through Docker:

``` bash
docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourTopSecretPassword -e POSTGRES_DB=prefect postgres:latest
```

``` bash
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"
```

`prefect server start`

[Ensure Prefect composes workflows in a separate server process](https://github.com/PrefectHQ/prefect/issues/6492#issuecomment-1221111132):

`prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"`

Set concurrency limits in [src/config.py](src/config.py) to not overschedule resources.

## Interactive entrypoint

[notebook/ALS_genomics_model_stack.ipynb](notebook/ALS_genomics_model_stack.ipynb)
