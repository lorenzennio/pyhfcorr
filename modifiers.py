import numpy as np

def shapesys(modifier_data, nominal, uv_subset):
    pass

def histosys(modifier_data, nominal, uv_subset):
    lo_diffs = np.array([np.subtract(m["lo_data"], nominal) for m in modifier_data])
    hi_diffs = np.array([np.subtract(m["hi_data"], nominal) for m in modifier_data])
    
    lo_shift = np.sum(uv_subset[:, np.newaxis] * lo_diffs, axis=0)
    hi_shift = np.sum(uv_subset[:, np.newaxis] * hi_diffs, axis=0)

    new_lo = nominal + lo_shift
    new_hi = nominal + hi_shift
    
    return {
            "type": "histosys",
            "data": {
                "lo_data": list(new_lo),
                "hi_data": list(new_hi)
            }
        }
    
def normsys(modifier_data, nominal, uv_subset):
    pass

def staterror(modifier_data, nominal, uv_subset):
    pass

def lumi(modifier_data, nominal, uv_subset):
    pass

def normfactor(modifier_data, nominal, uv_subset):
    pass

def shapefactor(modifier_data, nominal, uv_subset):
    pass