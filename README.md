<h3 align="center">
  elastiv-surv
</h3>

<h4 align="center">
  Survival analysis on Big Data
</h4>


<div align="center">

[![elastic-surv Tests](https://github.com/bcebere/elastic-surv/actions/workflows/test.yml/badge.svg)](https://github.com/bcebere/elastic-surv/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/bcebere/elastic-surv/blob/main/LICENSE)
  
</div>

 elastic-surv is a library for training risk estimation models on ElasticSearch backends. Potential use cases include user churn prediction or survival probability.
 
- :key: Survival models include CoxPH, DeepHit or LogisticHazard([pycox](https://github.com/havakv/pycox)).
- :fire: ElasticSearch support using [eland](https://github.com/elastic/eland).
- :cyclone: Automatic model selection using HyperBand.
 
## Problem formulation
Risk estimation tasks require:
 - A set of covariates/features(X).
 - An outcome/event column(Y) - 0 means right censoring, 1 means that the event occured.
 - Time to event column(T) - the duration until the event or the censoring occured. 

The risk estimation task output is a survival function: for N time horizons, it outputs the probability of "survival"(event not occurring) at each horizon.
 
## Installation

For configuring the ELK stack, please follow the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).

The library can be installed using
```bash
$ pip install .
```

## Sample Usage

For each ElasticSearch data backend, we need to mention:
 - the es_index_pattern and the es_client for the ES connection.
 - which keys in the ES index stand for the time-to-event and outcome data.
 - optional: which features to include from the index.

```python
from elastic_surv.dataset import ESDataset
from elastic_surv.models import CoxPHModel

dataset = ESDataset(
    es_index_pattern = 'churn-prediction',
    time_column = 'months_active',
    event_column = 'churned',
    es_client = "localhost",
)

model = CoxPHModel(in_features = dataset.features())
    
model.train(dataset)
model.score(dataset)
```
For this example, we use a local ES index, `churn-prediction`. This can be generated using the following snippet

```python
from pysurvival.datasets import Dataset
import eland as ed

raw_dataset = Dataset('churn').load() 

ed.pandas_to_eland(raw_dataset,
                  es_client='localhost',
                  es_dest_index='churn-prediction',
                  es_if_exists='replace',
                  es_dropna=True,
                  es_refresh=True,
) 
```

## Tutorials
 - [Tutorial 1: Data backends](tutorials/tutorial_1_data_backends.ipynb)
 - [Tutorial 2: Training a survival model over ElasticSearch](tutorials/tutorial_2_model_training.ipynb)
 - [Tutorial 3: AutoML for survival analysis over ElasticSearch](tutorials/tutorial_3_automl.ipynb)
 
## Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```
