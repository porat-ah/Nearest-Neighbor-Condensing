## Nearest-Neighbor-Condensing Implementation ##
___
This repository implements algorithm 1 and 2 from : [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368))
___
Usage
___
this API follows the Scikit Learn API structure.
first you create an object: 
`NNC(algorithm, metric, p, n_jobs)`

`algorithm`- determine if its algorithm 1 (`"brute"`) or algorithm 2 (`"prune"`) from the article.

`metric`- is one of the metrics that are supported or a callable function.

`p` - if the metric that was chosen is `"minkowski"` then l_p is the metric function.

`n_jobs` - the number of cpus to use , -1 for all cpus.

example
___
```
"X = np.array([
                [4, 1],
                [3, 1],
                [2, 1],
                [2.5, 0.5],
                [3.5, 0.5],
                [4, 2],
                [3, 2],
                [2, 2],
                [2.5, 2.5],
                [3.5, 2.5]
            ])
y = np.array([0,0,0,0,0,1,1,1,1,1])

nnc = NNC(algorithm= "prune", metric= "minkowski", p= 2, n_jobs= -1)
nnc.fit(X,y)
>>> NNC(algorithm='prune', metric='minkowski', p=2)
nnc.transform(X,y)
>>> (array([[4., 1.],
       [3., 1.],
       [2., 1.],
       [4., 2.],
       [3., 2.],
       [2., 2.]]), array([0, 0, 0, 1, 1, 1]))
```
See `main.py` for an example on the mnist dataset.
