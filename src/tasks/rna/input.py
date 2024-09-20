import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import (
    SelectFromModel,
    VarianceThreshold,
)
from sklearn.metrics import balanced_accuracy_score
from skopt.searchcv import BayesSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from pydeseq2.dds import DeseqDataSet
from joblib import Parallel, delayed
from tqdm import tqdm
from config import config
import re
from sklearn.utils import check_random_state
from skopt.space import Categorical, Integer, Real


class RNASeqDataset:
    def __init__(self, counts_file, clinical_file, config):
        self.config = config
        self.counts = self._load_counts(counts_file)
        self.clinical = self._load_clinical(clinical_file)
        self.processed_data = None
        self.sample_info = None
        self.dds = None

    def _load_counts(self, file_path):
        counts = pd.read_csv(file_path, index_col="Geneid")
        return counts.drop(columns=["Unnamed: 0"])

    def _load_clinical(self, file_path):
        return pd.read_excel(file_path)

    def preprocess(self):
        # Deduplicate sample columns
        sample_cols = [col for col in self.counts.columns if "-" in col]
        unique_mask = pd.Series([col.split("-")[1] for col in sample_cols]).duplicated(
            keep="first"
        )
        deduped_samples = [sample_cols[i] for i in np.where(unique_mask == False)[0]]

        # Prepare sample info
        self.sample_info = pd.DataFrame(
            {
                "sample": deduped_samples,
                "condition": [
                    (
                        self.config["data"]["case_label"]
                        if self.config["data"]["case_label"] in col
                        else self.config["data"]["control_label"]
                    )
                    for col in deduped_samples
                ],
            }
        ).set_index("sample")

        # Add sex genotype and filtered columns
        columns_to_add = self._get_columns_to_add()
        for col in columns_to_add:
            self.sample_info[col] = [
                self._get_clinical_data(sample, col) for sample in deduped_samples
            ]

        # Remove samples with missing data
        valid_samples = self.sample_info.dropna().index
        self.counts = self.counts[valid_samples]
        self.sample_info = self.sample_info.loc[valid_samples]

        # Apply clinical filtering
        self._apply_clinical_filter()

        # Create DeseqDataSet object
        self.dds = DeseqDataSet(
            counts=self.counts.T,
            metadata=self.sample_info,
            design_factors=["condition"],
            ref_level=("condition", self.config["data"]["control_label"]),
        )

        # Perform DESeq2 normalization
        self.dds.fit_size_factors()
        self.processed_data = pd.DataFrame(
            self.dds.layers["normed_counts"],
            columns=self.dds.var_names,
            index=self.dds.obs_names,
        )

        # Remove invariant features if specified
        if self.config["preprocessing"]["remove_invariant_features"]:
            selector = VarianceThreshold()
            self.processed_data = pd.DataFrame(
                selector.fit_transform(self.processed_data),
                columns=self.processed_data.columns[selector.get_support()],
                index=self.processed_data.index,
            )

        return self.processed_data, self.sample_info

    def _get_clinical_data(self, sample_id, column):
        external_id = sample_id.split("-")[1]
        matching_row = self.clinical[
            self.clinical[self.config["clinicalTable"]["idColumn"]] == external_id
        ]
        return matching_row[column].values[0] if not matching_row.empty else None

    def _get_columns_to_add(self):
        columns = set([self.config["data"]["sex_genotype_column"]])
        filter_string = self.config["clinicalTable"]["filters"]
        if filter_string:
            # Extract column names from the filter string
            filter_columns = re.findall(r"\b(\w+)\s*[<>=!]+", filter_string)
            columns.update(filter_columns)
        return list(columns)

    def _apply_clinical_filter(self):
        filter_string = self.config["clinicalTable"]["filters"]
        if not filter_string:
            return

        print(f"Applying clinical filter: {filter_string}")
        try:
            self.sample_info = self.sample_info.query(filter_string, engine="python")
            self.counts = self.counts[self.sample_info.index]
        except Exception as e:
            print(f"Error applying filter: {e}")
            print("Continuing without applying the filter.")


class StratifiedGroupSampler:
    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit_resample(self, X, y, sex_genotype):
        random_state = check_random_state(self.random_state)

        # Ensure sex_genotype_grouped is a string with consistent dtype
        sex_genotype_grouped = sex_genotype.apply(
            lambda x: "Y" if "Y" in x else "X"
        ).astype(
            str
        )  # Use native string type

        # Ensure y is also a string with the same consistent dtype
        y_str = y.astype(str)  # Use native string type

        # Concatenate y_str and sex_genotype_grouped
        strata = np.array(
            [
                f"{y}_{sex_genotype}"
                for y, sex_genotype in zip(y_str, sex_genotype_grouped)
            ]
        )

        # Use numpy's unique method to get unique strata
        unique_strata = np.unique(strata)
        sampled_indices = []

        min_stratum_size = min(
            len(strata[strata == stratum]) for stratum in unique_strata
        )

        for stratum in unique_strata:
            stratum_indices = np.where(strata == stratum)[0]
            n_samples = min(min_stratum_size, int(len(X) / len(unique_strata)))
            sampled_indices.extend(
                random_state.choice(stratum_indices, size=n_samples, replace=True)
            )

        return X.iloc[sampled_indices], y.iloc[sampled_indices]


class LassoFeatureSelector:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.selector = None

    def fit(self, X, y):
        lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
        self.selector = SelectFromModel(lasso, prefit=False, threshold="median")
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()


class MLWorkflow:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.X = dataset.processed_data
        self.y = dataset.sample_info["condition"]
        self.sex_genotype = dataset.sample_info[config["data"]["sex_genotype_column"]]
        # No label encoding at this stage; y remains as strings

    def _preprocess(self, X):
        var_selector = VarianceThreshold(threshold=1e-8)
        X_var = var_selector.fit_transform(X)
        non_constant_features = X.columns[var_selector.get_support()]
        return pd.DataFrame(X_var, columns=non_constant_features, index=X.index)

    def _get_feature_selector(self, model_instance, random_state):
        if hasattr(model_instance, "coef_") or hasattr(
            model_instance, "feature_importances_"
        ):
            return SelectFromModel(estimator=model_instance, threshold="median")
        else:
            return LassoFeatureSelector(random_state=random_state)

    def _single_bootstrap(self, random_state):
        sampler = StratifiedGroupSampler(random_state=random_state)

        # y remains as strings here
        X_boot, y_boot = sampler.fit_resample(self.X, self.y, self.sex_genotype)
        sex_boot = self.sex_genotype.loc[y_boot.index]

        X_boot = self._preprocess(X_boot)

        sex_boot_grouped = sex_boot.apply(lambda x: "Y" if "Y" in x else "X")
        stratification = y_boot.astype(str) + "_" + sex_boot_grouped.astype(str)

        # Split the data, maintaining y as strings
        X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
            X_boot,
            y_boot,
            sex_boot,
            test_size=self.config["ml_workflow"]["test_size"],
            stratify=stratification,
            random_state=random_state,
        )

        # Encode labels only for model training
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        best_model = None
        best_accuracy = 0
        best_result = None

        for model_name, model_config in self.config["models"].items():
            print(f"Running {model_name}...")
            clf_params = model_config["fixed_params"].copy()

            if (
                model_name == "LogisticRegression"
                and clf_params.get("penalty") == "elasticnet"
            ):
                if "l1_ratio" not in clf_params:
                    clf_params["l1_ratio"] = 0.5

            model_instance = model_config["class"](**clf_params)
            feature_selector = self._get_feature_selector(model_instance, random_state)

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("feature_selection", feature_selector),
                    ("clf", model_instance),
                ]
            )

            search_space = {
                f"clf__{param}": space
                for param, space in model_config["search_space"].items()
            }

            if isinstance(feature_selector, SelectFromModel):
                search_space["feature_selection__threshold"] = Real(
                    1e-5, 1e-3, prior="log-uniform"
                )

            bayes_search = BayesSearchCV(
                pipeline,
                search_space,
                n_iter=self.config["hyperparameter_optimization"]["n_iter"],
                cv=self.config["hyperparameter_optimization"]["cv"],
                scoring=self.config["hyperparameter_optimization"]["scoring"],
                n_jobs=self.config["hyperparameter_optimization"]["n_jobs"],
                verbose=self.config["hyperparameter_optimization"]["verbose"],
                random_state=random_state,
            )

            bayes_search.fit(X_train, y_train_encoded)

            y_pred_encoded = bayes_search.predict(X_test)
            accuracy = balanced_accuracy_score(y_test_encoded, y_pred_encoded)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
                best_result = {
                    "accuracy": accuracy,
                    "best_params": bayes_search.best_params_,
                    "feature_importances": self._get_feature_importances(
                        bayes_search.best_estimator_
                    ),
                    "selected_features": bayes_search.best_estimator_.named_steps[
                        "feature_selection"
                    ].get_support(),
                    "y_pred": label_encoder.inverse_transform(
                        y_pred_encoded
                    ),  # Decode predictions
                    "y_true": y_test,  # Original string labels
                    "model": model_name,
                }

        print(f"Best model for this bootstrap: {best_model}")
        return best_result

    def _get_feature_importances(self, estimator):
        feature_selection_step = estimator.named_steps["feature_selection"]
        clf_step = estimator.named_steps["clf"]

        if hasattr(clf_step, "feature_importances_"):
            return clf_step.feature_importances_
        elif hasattr(clf_step, "coef_"):
            return (
                np.abs(clf_step.coef_[0])
                if clf_step.coef_.ndim > 1
                else np.abs(clf_step.coef_)
            )
        elif isinstance(feature_selection_step, LassoFeatureSelector):
            return np.abs(feature_selection_step.selector.estimator_.coef_)
        else:
            return None

    def analyze_results(self, results):
        accuracies = [result["accuracy"] for result in results]
        print(
            f"Mean Balanced Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})"
        )

        model_counts = pd.Series([result["model"] for result in results]).value_counts()
        print("\nModel Selection Frequency:")
        print(model_counts)

        # Analyze feature importance
        feature_importances = np.mean(
            [result["feature_importances"] for result in results], axis=0
        )
        selected_features = np.sum(
            [result["selected_features"] for result in results], axis=0
        )

        top_features = (
            pd.DataFrame(
                {
                    "Feature": self.X.columns,
                    "Importance": feature_importances,
                    "Selection_Frequency": selected_features / len(results),
                }
            )
            .sort_values("Importance", ascending=False)
            .head(self.config["analysis"]["n_top_features"])
        )

        print(f"\nTop {self.config['analysis']['n_top_features']} Features:")
        print(top_features)

        # Identify consistently accurate samples
        sample_accuracies = self._get_sample_accuracies(results)
        consistent_samples = sample_accuracies[
            sample_accuracies
            >= self.config["ml_workflow"]["balanced_accuracy_threshold"]
        ].index
        print(f"\nNumber of consistently accurate samples: {len(consistent_samples)}")
        return consistent_samples

    def _get_sample_accuracies(self, results):
        sample_correct_predictions = pd.DataFrame(
            index=self.X.index, columns=range(len(results))
        )

        for i, result in enumerate(results):
            y_pred = result["y_pred"]
            y_true = result["y_true"]
            correct_predictions = y_true == y_pred
            sample_correct_predictions.loc[y_true.index, i] = correct_predictions

        return sample_correct_predictions.mean(axis=1)

    def bootstrap_evaluate(self):
        results = [
            self._single_bootstrap(1)
            # for i in range(self.config["ml_workflow"]["n_bootstraps"])
        ]
        return results


# Usage
dataset = RNASeqDataset(
    config["data"]["counts_file"], config["data"]["clinical_file"], config
)
processed_data, sample_info = dataset.preprocess()

workflow = MLWorkflow(dataset, config)
results = workflow.bootstrap_evaluate()
consistent_samples = workflow.analyze_results(results)

# Further analysis on consistent samples
consistent_data = processed_data.loc[consistent_samples]
consistent_info = sample_info.loc[consistent_samples]
print("\nCharacteristics of consistently accurate samples:")
print(consistent_info.describe())
