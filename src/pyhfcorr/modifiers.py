import numpy as np

def shapesys(modifier_data, nominal, uv_subset):
    diffs = np.array(modifier_data)

    shift = np.sum(uv_subset[:, np.newaxis] * diffs, axis=0)

    if not np.any(shift):
        return None

    return {
            "type": "shapesys",
            "data": shift
        }

def histosys(modifier_data, nominal, uv_subset):
    lo_diffs = np.array([np.subtract(m["lo_data"], nominal) for m in modifier_data])
    hi_diffs = np.array([np.subtract(m["hi_data"], nominal) for m in modifier_data])

    lo_shift = np.sum(uv_subset[:, np.newaxis] * lo_diffs, axis=0)
    hi_shift = np.sum(uv_subset[:, np.newaxis] * hi_diffs, axis=0)

    new_lo = (nominal + lo_shift).tolist()
    new_hi = (nominal + hi_shift).tolist()

    if new_lo == new_hi:
        return None

    return {
            "type": "histosys",
            "data": {
                "lo_data": new_lo,
                "hi_data": new_hi
            }
        }

def normsys(modifier_data, nominal, uv_subset):
    lo_diffs = np.array([m["lo"] for m in modifier_data])
    hi_diffs = np.array([m["hi"] for m in modifier_data])

    new_lo = np.sum(uv_subset * lo_diffs)
    new_hi = np.sum(uv_subset * hi_diffs)

    if new_lo == new_hi:
        return None

    return {
            "type": "normsys",
            "data": {
                "lo": new_lo,
                "hi": new_hi
            }
        }

def staterror(modifier_data, nominal, uv_subset):
    diffs = np.array(modifier_data)

    shift = np.sqrt(np.sum((uv_subset[:, np.newaxis] * diffs)**2, axis=0))

    if not np.any(shift):
        return None

    return {
            "type": "staterror",
            "data": shift
        }
