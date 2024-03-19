import numpy as np
from copy import deepcopy
import warnings
from pyhfcorr import modifiers

def _pca(corr, return_rot=False):
    """Principal Component analysis, moving to a space where the covariance matrix is diagonal
    https://www.cs.cmu.edu/~elaw/papers/pca.pdf

    Args:
        cov (array): Correlation matrix

    Returns:
        array: matrix of column wise error vectors (eigenvectors * sqrt(eigenvalues); sqrt(eigenvalues) = std)
    """
    svd = np.linalg.svd(corr, hermitian=True)
    uvec = svd[0] @ np.sqrt(np.diag(svd[1]))
    if return_rot:
        return uvec, svd[0]
    return uvec

def validate(corr):
    for c in corr:
        shape = np.shape(c["corr"])
        if len(shape) != 2:
            raise ValueError("Correlation matrix must be 2-dimensional.")
        if shape[0] != shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if len(c["vars"]) != len(c["corr"]):
            raise ValueError("Number of variables does not match dimension of correlation matrix")
        
def get_coords(var_name, spec):
    coords = []
    for channel_index, ch in enumerate(spec):
        for sample_index, sa in enumerate(ch["samples"]):
            for imo, mo in enumerate(sa["modifiers"]):
                if mo["name"] == var_name:
                    coords.append((channel_index, sample_index, imo))
    return coords
    
def group_coords(coords, coords_index):
    # group coords by channel and sample, but remember index
    group_coords = {}
    for c, i in zip(coords, coords_index):
        if (c[0], c[1]) not in group_coords.keys():
            group_coords[(c[0], c[1])] = {"uv_index": [], "modifier_index": []}
        group_coords[(c[0], c[1])]["uv_index" ].append(i)
        group_coords[(c[0], c[1])]["modifier_index"].append(c[2])
    return group_coords

def decorrelate(spec):
    if "correlations" in spec.keys():
        print("correlation found")
        
        validate(spec["correlations"])
        
        channels = deepcopy(spec["channels"])
        
        for corr in spec["correlations"]:
            
            # compute decorrelation 
            uvec = _pca(corr["corr"])
            
            # get channel, sample and modifier index for each variable
            coords = []
            coords_index = []
            for ic, var in enumerate(corr["vars"]):
                c = get_coords(var, channels)
                coords += get_coords(var, channels)
                coords_index += [ic for _ in c]
                
            grouped_coords = group_coords(coords, coords_index)
            
            modifier_type = [channels[c[0]]["samples"][c[1]]["modifiers"][c[2]]["type"] for c in coords]
            modifier_type = np.unique(modifier_type)
            
            # check if all modifiers have the same type
            if len(modifier_type) != 1:
                raise ValueError("Correlated modifiers must have the same type.")
            modifier_type = modifier_type[0]
            
            if modifier_type in ["lumi", "normfactor", "shapefactor"]:
                warnings.warn(f"Modifiers without data can only be full (de)correlated and should be treated with pyhf directly.")
                return {"channels": channels}
            
            # compute shifts for each independent eigenvector
            for uv_ind, uv in enumerate(uvec.T):
                for (channel_index, sample_index), mods in grouped_coords.items():
                    uv_index  = mods["uv_index"]
                    modifier_index = mods["modifier_index"]
                    
                    nominal = np.array(channels[channel_index]["samples"][sample_index]["data"])
                    modifier_data = [
                        channels[channel_index]["samples"][sample_index]["modifiers"][imo]["data"] 
                        for imo in modifier_index]
                    uv_subset = uv[uv_index]

                    new_mod = getattr(modifiers, modifier_type)(modifier_data, nominal, uv_subset)
                    
                    if new_mod is None:
                        name = corr["name"] + f"[{str(uv_ind)}]"
                        warnings.warn(f"Modifier {name} is redundant and is not added.")
                        continue
                    
                    new_mod["name"] = corr["name"] + f"[{str(uv_ind)}]"
                    
                    channels[channel_index]["samples"][sample_index]["modifiers"].append(new_mod)
                    
            for (channel_index, sample_index) in grouped_coords.keys():
                new_modifiers = []
                for m in channels[channel_index]["samples"][sample_index]["modifiers"]:
                    if m["name"] not in corr["vars"]:
                        new_modifiers.append(m)
                channels[channel_index]["samples"][sample_index]["modifiers"] = new_modifiers
        
    return {"channels": channels}