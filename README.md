## Nearest-Neighbor-Condensing Implementation ##

# Content
* [Overview](#overview)
* [Files](#components)
* [Dependencies](#dependencies)
* [Local Installation](#local-installation)
* [Usage](#usage)
* [Example](#example)
* [References](#references)
---
## Overview
---
This repository implements algorithm 1 and 2 from : [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368)

---
## Files
---

**nnc/NNC.py** - the implementation of the algorithm

**nnc/Metric.py** - a class for calculating distance

**tests/*** - unit testing 

**main.py** -  an example run of the library on the mnist dataset.

---
## Dependencies
---
### Libraries:
* numpy~=1.19.5
* scikit-learn~=1.0.2
* tqdm~=4.64.0
* scipy~=1.7.3
* setuptools~=61.2.0
* pandas~=1.1.0
### Backend:
* python ( => 3.6)

---
# Local Installation
---
clone the repository and navigate to it:
``` sh
$ git clone https://github.com/porat-ah/Nearest-Neighbor-Condensing.git
$ cd Nearest-Neighbor-Condensing
```
---
## Usage
---
```
1. from nnc import *

2. create an object: 
`NNC(algorithm, metric, p, n_jobs)`

`algorithm`- determine if its algorithm 1 (`"brute"`) or algorithm 2 (`"prune"`) from the article.

`metric`- is one of the metrics that are supported or a callable function.

`p` - if the metric that was chosen is `"minkowski"` then l_p is the metric function.

`n_jobs` - the number of cpus to use , -1 for all cpus.

`verbose` - show progress bar

3. nnc.fit(X, y)

4. X_new , y_new = nnc.transfrom(X,y)

you can run steps 3,4 together using:
X_new , y_new = nnc.fit_transform(X, y)

```

---
## example
---
```
>>> X = np.array([
...             [4, 1],
...             [3, 1],
...             [2, 1],
...             [2.5, 0.5],
...             [3.5, 0.5],
...             [4, 2],
...             [3, 2],
...             [2, 2],
...             [2.5, 2.5],
...             [3.5, 2.5]
...         ])
>>> y = np.array([0,0,0,0,0,1,1,1,1,1])
>>> nnc = NNC(algorithm= "prune", metric= "minkowski", p= 2, n_jobs= -1)
>>> nnc.fit(X,y)
NNC(algorithm='prune', metric='minkowski')
>>> nnc.transform(X,y)
(array([[4., 1.],
       [3., 1.],
       [2., 1.],
       [4., 2.],
       [3., 2.],
       [2., 2.]]), array([0, 0, 0, 1, 1, 1]))
```
---
## References
---
* Article : [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368)

