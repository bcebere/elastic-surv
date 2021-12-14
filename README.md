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
 
## Installation

For configuring the ELK stack, please follow the instructions [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).

The library can be installed using
```bash
$ pip install .
```

## Sample Usage

```python
from elastic_surv.dataset import ESDataset
from elastic_surv.models import CoxPHModel

dataset = ESDataset(
    es_index_pattern = 'churn-prediction',
    time_column = 'months_active',
    event_column = 'churned',
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
## Tests

Install the testing dependencies using
```bash
pip install .[testing]
```
The tests can be executed using
```bash
pytest -vsx
```

## Tutorials
TODO
