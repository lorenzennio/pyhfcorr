# pyhfcorr -- treating correlated uncertainties with pyhf

[`pyhf`](https://pyhf.readthedocs.io) is based on the [`HistFactory`](https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf) statistical model. This is a very general tool for statistical inference of binned data.

One feature that is not included in [`pyhf`](https://pyhf.readthedocs.io), is the correct treatment of arbitrarily correlated uncertainties. The current implementation features only fully (de)correlated uncertainties. 

The use of this package is to add the option for arbitrarily correlated uncertainties, by a simple pre-processing step of the [pyhf](https://pyhf.readthedocs.io) model. The mathematical background used is simple [principal component analysis (PCA)](https://www.cs.cmu.edu/~elaw/papers/pca.pdf).

To account for correlations between parameters, one simply adds a `correlation` field to the [`pyhf`](https://pyhf.readthedocs.io) model. Here we specify a `name`, which will be the new modifier name, the correlated variables `vars`, and the correlation matrix `corr`: 

```python
spec = {
    "channels" : ..., 
    "correlations": [
        {
            "name": "corr_1_2",
            "vars": ["unc1", "unc2"],
            "corr": [[1.0, 0.5], [0.5, 1.]],
        }
    ]
}
```

By pre-processing the model specification, we obtain a new specification, which is now pyhf compatible and has the correlation correctly implemented:

```python
new_spec = pyhfcorr.decorrelate.decorrelate(spec)
```
