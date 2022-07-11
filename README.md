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

...

## Setup <a name = "setup"></a>

To get started, clone this repo and check that you have all requirements installed.

```
git clone https://github.com/vitostamatti/sklearn-categorical-embedder.git
pip install ./sklearn-categorical-embedder
``` 

## Usage <a name = "usage"></a>

In the [notebooks](/notebooks/) directory you can find examples of usage.

```
from categorical_embedder import CategoricalEmbedder

ce = CategoricalEmbedder(emb_size=32, random_state=123)

ce.fit(X,y)

ce.transform(X)
```



## TODOS

- [X] First commit.
- [ ] Complete docstrings
- [ ] Example Notebooks


## License

[MIT](LICENSE.txt)