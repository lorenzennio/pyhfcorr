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
                    "type": "normsys",
                    "data": {"hi": 2., "lo": 0.}
                },
                {
                    "name": "u2",
                    "type": "normsys",
                    "data": {"hi": 3., "lo": 0.5}
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
    
    initial_parameters = new_model.config.suggested_init()
    
    initial_parameters[1] = -1.
    assert pytest.approx(new_model.expected_data(initial_parameters, include_auxdata=False), 1e-5) == 0.5*hist2
    
    initial_parameters[1] = 0.
    assert pytest.approx(new_model.expected_data(initial_parameters, include_auxdata=False), 1e-5) == hist2
    
    initial_parameters[1] = 1.
    assert pytest.approx(new_model.expected_data(initial_parameters, include_auxdata=False), 1e-5) == 5*hist2
    
def test_doublesample():
    samples = [
        {
            "name": "sample1",
            "data": list(hist1),
            "modifiers": [
                {"name": "mu", "type": "normfactor", "data" : None},
                {
                    "name": "u1",
                    "type": "normsys",
                    "data": {"hi": 2., "lo": 0.}
                },
                {
                    "name": "u12",
                    "type": "normsys",
                    "data": {"hi": 3., "lo": 0.5}
                }
            ],
            
        },    
        {
            "name": "sample2",
            "data": list(hist2),
            "modifiers": [
                {
                    "name": "u1",
                    "type": "normsys",
                    "data": {"hi": 4., "lo": 1.}
                },
            ],
            
        },
    ]
    
    spec = {
        "channels" : [{"name" : "singlechannel", "samples" : samples}], 
        "correlations": [
            {
                "name": "corr",
                "vars": ["u1", "u12"],
                "corr": [[1., 1.], [1., 1.]],
            }
        ]}

    
    new_spec = decorrelate.decorrelate(spec)
    
    pytest.warns(UserWarning, decorrelate.decorrelate, spec)
    
    new_model = pyhf.Model(new_spec)
    
    initial_parameters = new_model.config.suggested_init()
    
    initial_parameters[1] = -1.
    expected_data = new_model.expected_data(initial_parameters, include_auxdata=False)
    print(expected_data)
    assert pytest.approx(expected_data, 1e-5) == 0.5*hist1 + hist2
    
    initial_parameters[1] = 0.
    expected_data = new_model.expected_data(initial_parameters, include_auxdata=False)
    print(expected_data)
    assert pytest.approx(expected_data, 1e-5) == hist1 + hist2
    
    initial_parameters[1] = 1.
    expected_data = new_model.expected_data(initial_parameters, include_auxdata=False)
    print(expected_data)
    assert pytest.approx(expected_data, 1e-5) == 5*hist1 + 4*hist2
    