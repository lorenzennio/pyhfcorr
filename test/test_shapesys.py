import pytest
import numpy as np
import pyhf
from pyhfcorr import decorrelate

hist1 = np.array([1.5, 3., 6., 7.5, 6.3, 6.6, 5.5, 2.5, 3. , 1.5])
hist2 = np.array([3. , 6., 9., 12., 15., 9. , 6., 3. , 3.3, 2.15])

def test_singlesample():
    samples = [
        {
            "name": "sample1",
            "data": list(hist2),
            "modifiers": [
                {"name": "mu", "type": "normfactor", "data" : None},
                {
                    "name": "u1",
                    "type": "shapesys",
                    "data": list(hist1)
                },
                {
                    "name": "u2",
                    "type": "shapesys",
                    "data": list(hist2)
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
    
    new_spec = decorrelate.decorrelate(spec)
    new_model = pyhf.Model(new_spec)
    
    auxdata = new_model.config.auxdata
    print(auxdata)
    
    assert pytest.approx(auxdata[:10], 1e-5) == list((hist2 / (hist1+hist2)) ** 2)