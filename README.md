# pyhfcorr
## Treating correlated uncertainties with pyhf

[`pyhf`](https://pyhf.readthedocs.io) is based on the [`HistFactory`](https://cds.cern.ch/record/1456844/files/CERN-OPEN-2012-016.pdf) statistical model. This is a very general tool for statistical inference of binned data.

One feature that is not included in [`pyhf`](https://pyhf.readthedocs.io) is the correct treatment of arbitrarily correlated uncertainties. The current implementation features only fully (de)correlated uncertainties.

The use of this package is to add the option for arbitrarily correlated uncertainties, by a simple pre-processing step of the [pyhf](https://pyhf.readthedocs.io) model. The mathematical background used is simple [singular value decomposition (SVD)](https://www.cs.cmu.edu/~elaw/papers/pca.pdf) (also see below).

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

By pre-processing the model specification, we obtain a new specification, which is now [`pyhf`](https://pyhf.readthedocs.io) compatible and has the correlation correctly implemented:

```python
new_spec = pyhfcorr.decorrelate.decorrelate(spec)
```

## Singular value decomposition

We can decompose a correlation matrix $C$ as

$$ C = U S^2 U^H = USSU^H = (US)(US)^H,$$

where $U$ is a unitary transformation matrix, $UU^H=1$ and $S$ is the diagonal matrix of standard deviations, $S=S^H$.

The geometrical interpretation of this is, that by applying the transformation $(US)^{-1}$ to correlated data results in an uncorrelated data-set with unity standard deviation.

The rotation $U^{-1} = U^H$ rotates points to a new coordinate system, where correlations between the dimensions vanish. In this rotated coordinate system, $S^{-1}$ scales the dimensions accordingly.

An illustration for a 2-dimensional random multivariate dataset $x$ with correlation coefficient $\rho=0.8$ is shown here:

![pca illustration](./svd.svg)
