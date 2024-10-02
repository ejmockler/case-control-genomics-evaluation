import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Union
import hail as hl
import ray  # New import for Ray
from pyspark.sql import functions as F, DataFrame as SparkDataFrame
from skopt import BayesSearchCV  

from data.sample_processor import SampleProcessor
from config import SamplingConfig
from data.genotype_processor import GenotypeProcessor

# Initialize Ray at the module level
ray.init(ignore_reinit_error=True)

@ray.remote
def evaluate_model(model, search_spaces, X_train, y_train, X_test, y_test, n_iter=10):
    """
    Remote function to perform Bayesian hyperparameter optimization for a model.

    Args:
        model: The pre-initialized model instance.
        search_spaces: Dictionary of hyperparameter search spaces.
        X_train, y_train, X_test, y_test: Training and testing data.
        n_iter: Number of iterations for Bayesian optimization.

    Returns:
        Tuple containing the best AUC and corresponding parameters.
    """
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        search = BayesSearchCV(
            pipeline,
            search_spaces,
            n_iter=n_iter,
            cv=3,
            n_jobs=-1,
            scoring='roc_auc'
        )
        
        search.fit(X_train, y_train)
        
        y_pred = search.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        
        return auc, search.best_params_
    except Exception as e:
        raise e

def bootstrap_models(
    sample_processor: SampleProcessor,
    genotype_processor: GenotypeProcessor,
    data: Union[hl.MatrixTable, SparkDataFrame],  # Updated parameter
    config: SamplingConfig,
    stack: Dict,
    random_state=42
) -> pd.DataFrame:
    """
    Perform bootstrapping of models based on the configuration.

    Args:
        sample_processor (SampleProcessor): Instance of SampleProcessor.
        genotype_processor (GenotypeProcessor): Instance of GenotypeProcessor.
        data (Union[hl.MatrixTable, SparkDataFrame]): The processed dataset.
        config (SamplingConfig): Configuration object containing bootstrap_iterations and other settings.
        stack (Dict): Dictionary of models and their hyperparameter distributions.
        random_state (int): Seed for random number generation.

    Returns:
        pd.DataFrame: Aggregated performance metrics across all iterations and models.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting bootstrapping of models.")

    results = []

    for iteration in range(config.bootstrap_iterations):
        logger.info(f"Bootstrapping iteration {iteration + 1}/{config.bootstrap_iterations}")

        # Adjust random_state to ensure different splits for each iteration
        current_random_state = random_state + iteration

        # Draw a new train-test split for each bootstrap iteration
        train_test_sample_ids = sample_processor.draw_train_test_split(
            test_size=config.test_size,
            random_state=current_random_state
        )

        train_samples = train_test_sample_ids['train']['samples']
        test_samples = train_test_sample_ids['test']['samples']

        # Extract sample IDs mapped to VCF samples
        train_sample_ids = list(train_samples.values())
        test_sample_ids = list(test_samples.values())

        logger.info(f"Number of training samples: {len(train_sample_ids)}")
        logger.info(f"Number of testing samples: {len(test_sample_ids)}")

        # Fetch genotypes for training and testing samples
        logger.info("Fetching genotypes for training samples.")
        train_genotypes = genotype_processor.fetch_genotypes(
            data=data, 
            sample_ids=train_sample_ids
        )

        logger.info("Fetching genotypes for testing samples.")
        test_genotypes = genotype_processor.fetch_genotypes(
            data=data, 
            sample_ids=test_sample_ids
        )

        # Determine the type of fetched data and convert if necessary
        if isinstance(train_genotypes, hl.MatrixTable):
            logger.info("Converting Hail MatrixTable to Spark DataFrame for training data.")
            train_spark_df = genotype_processor.to_spark_df(train_genotypes)
        elif isinstance(train_genotypes, SparkDataFrame):
            train_spark_df = train_genotypes
        else:
            logger.error("Unsupported data type for training genotypes. Skipping iteration.")
            continue

        if isinstance(test_genotypes, hl.MatrixTable):
            logger.info("Converting Hail MatrixTable to Spark DataFrame for testing data.")
            test_spark_df = genotype_processor.to_spark_df(test_genotypes)
        elif isinstance(test_genotypes, SparkDataFrame):
            test_spark_df = test_genotypes
        else:
            logger.error("Unsupported data type for testing genotypes. Skipping iteration.")
            continue

        # Convert Spark DataFrame to Pandas DataFrame for scikit-learn compatibility
        logger.info("Converting Spark DataFrame to Pandas DataFrame for ML processing.")
        try:
            X_train = train_spark_df.toPandas()
            X_test = test_spark_df.toPandas()

            # Extract labels
            y_train = sample_processor.get_labels(train_sample_ids)
            y_test = sample_processor.get_labels(test_sample_ids)

            # Ensure that the indices align
            X_train.set_index('sample_id', inplace=True)
            X_test.set_index('sample_id', inplace=True)

            # Align labels with features
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]
        except Exception as e:
            logger.error(f"Failed to convert Spark DataFrame to Pandas DataFrame: {e}")
            continue

        # List to hold Ray task references
        tasks = []

        # Iterate over each model in the stack and submit evaluation tasks to Ray
        for model, search_spaces in stack.items():
            logger.info(f"Submitting training tasks for model: {model.__class__.__name__}")

            tasks.append(
                evaluate_model.remote(
                    model=model,
                    search_spaces=search_spaces,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    n_iter=10  # You can adjust this based on your needs
                )
            )

        # Retrieve results from Ray
        try:
            ray_results = ray.get(tasks)
        except Exception as e:
            logger.error(f"Ray task retrieval failed: {e}")
            continue

        # Dictionary to store the best results per model
        best_results = {model.__class__.__name__: {'best_auc': 0, 'best_params': None} for model in stack.keys()}

        # Process the results
        for model, (auc, params) in zip(stack.keys(), ray_results):
            model_name = model.__class__.__name__
            if auc > best_results[model_name]['best_auc']:
                best_results[model_name]['best_auc'] = auc
                best_results[model_name]['best_params'] = params

        # Append the best results for this iteration
        for model_name, result in best_results.items():
            if result['best_auc'] > 0:
                results.append({
                    'iteration': iteration + 1,
                    'model': model_name,
                    'best_auc': result['best_auc'],
                    'best_params': result['best_params']
                })
                logger.info(f"Completed training for {model_name} in iteration {iteration + 1} with AUC: {result['best_auc']}")
            else:
                logger.warning(f"No successful training for {model_name} in iteration {iteration + 1}")

    # Shutdown Ray after all iterations are complete
    ray.shutdown()
    logger.info("Completed bootstrapping of models.")

    # Convert results to DataFrame for aggregation
    results_df = pd.DataFrame(results)

    return results_df