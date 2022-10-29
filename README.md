# Categorical Embedder

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/vitostamatti/sklearn-categorical-embedder.svg)](https://github.com/vitostamatti/sklearn-categorical-embedder/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/vitostamatti/sklearn-categorical-embedder.svg)](https://github.com/vitostamatti/sklearn-categorical-embedder/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)



## 📝 Table of Contents

- [About](#about)
- [Setup](#setup)
- [Usage](#usage)


## About <a name = "about"></a>

A scikit-learn transformer to encode categorical features into vectors of a given size.


On the ``fit()`` method computes the following steps:

1-It uses ``OrdinalEncoder`` to convert categories into indexes, adding one
extra category for unknown or nan values.

2-t adds an offset to each category index to identify all possible
values of all features with on specific value.

3-It uses a ``OneHotEncoder`` and adds a constant to each feature column.

4-It uses the ``MLPClassifier`` of sklearn to extract embeddings from
one hotted input.

On the ``transform()`` method computes the following steps:

1-It uses the fitted OrdinalEncoder to get the indexes of the inputs

2-Uses ``np.take`` to get specific feature values embeddings from the embedding matrix.

3-Reshapes the embeddings in a vector of ``shape=(n_observations,n_features*emb_size)``

## Setup <a name = "setup"></a>

To get started, clone this repo and check that you have all requirements installed.

```
git clone https://github.com/vitostamatti/sklearn-categorical-embedder.git
pip install ./sklearn-categorical-embedder
``` 

## Usage <a name = "usage"></a>

In the [notebooks](/notebooks/) directory you can find examples of usage.

```python
from categorical_embedder import CategoricalEmbedder

ce = CategoricalEmbedder(emb_size=32, random_state=123)

ce.fit(X,y)

ce.transform(X)
```



## TODOS

- [X] First commit.
- [X] Example Notebooks
- [ ] Complete docstrings



## License

[MIT](LICENSE.txt)
