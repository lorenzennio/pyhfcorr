import pytest
import warnings
import numpy as np
import pyhf
import pyhfcorr

hist1 = np.array([1.5, 3., 6., 7.5, 6.3, 6.6, 5.5, 2.5, 3. , 1.5])

def test_singlesample():
    samples = [
        {
            "name": "sample1",
            "data": list(hist1),
            "modifiers": [
                {"name": "mu", "type": "normfactor", "data" : None},
                {
                    "name": "u1",
                    "type": "normfactor",
                    "data": None
                },
                {
                    "name": "u2",
                    "type": "normfactor",
                    "data": None
                }
        ],
            
        },
    ]

    spec = {
    "channels" : [{"name" : "singlechannel", "samples" : samples}], 
    "correlations": [
        {
            "name": "corr",
            "vars": ["u1", "u2"],
            "corr": [[1., 1.], [1., 1.]],
        }
    ]}

    new_spec = pyhfcorr.decorrelate(spec)
    
    pytest.warns(UserWarning, pyhfcorr.decorrelate, spec)
    
    new_spec == {"channels" : [{"name" : "singlechannel", "samples" : samples}]}
