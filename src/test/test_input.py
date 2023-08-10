import pytest
from prefect.testing.utilities import prefect_test_harness
import sys

sys.path.append("src")
from example_config import config
from tasks.input import load, processInputFiles


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield


def test_load_valid_paths(config):
    clinicalData, externalSamples, annotatedVCF = load.fn(config)

    assert not clinicalData.empty
    assert not annotatedVCF.empty
    assert all([not df.empty for df in externalSamples])


def test_binary_allele_model(config):
    config["vcfLike"]["binarize"] = True
    genotypeData, clinicalData = processInputFiles(config)
    for sampleType in ["case", "control", "holdout_case", "holdout_control"]:
        dataframe = getattr(genotypeData, sampleType).genotype
        assert all(dataframe.isin([0, 1]))


def test_summed_allele_model(config):
    config["vcfLike"]["binarize"] = False
    genotypeData, clinicalData = processInputFiles(config)
    for sampleType in ["case", "control", "holdout_case", "holdout_control"]:
        dataframe = getattr(genotypeData, sampleType).genotype
        assert dataframe.applymap(lambda x: isinstance(x, int)).all().all()
