import numpy as np
from copy import deepcopy

def _pca(corr, return_rot=False):
    """Principal Component analysis, moving to a space where the covariance matrix is diagonal
    https://www.cs.cmu.edu/~elaw/papers/pca.pdf

    Args:
        cov (array): Correlation matrix

    Returns:
        array: matrix of column wise error vectors (eigenvectors * sqrt(eigenvalues); sqrt(eigenvalues) = std)
    """
    svd = np.linalg.svd(corr)
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
    for ich, ch in enumerate(spec):
        for isa, sa in enumerate(ch["samples"]):
            for imo, mo in enumerate(sa["modifiers"]):
                if mo["name"] == var_name:
                    coords.append((ich, isa, imo))
    return coords
    
def group_coords(coords, icoords):
    # group coords by channel and sample, but remember index
    group_coords = {}
    for c, i in zip(coords, icoords):
        if (c[0], c[1]) not in group_coords.keys():
            group_coords[(c[0], c[1])] = {"uv_ind": [], "mod_ind": []}
        group_coords[(c[0], c[1])]["uv_ind" ].append(i)
        group_coords[(c[0], c[1])]["mod_ind"].append(c[2])
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
            icoords = []
            for ic, var in enumerate(corr["vars"]):
                c = get_coords(var, channels)
                coords += get_coords(var, channels)
                icoords += [ic for _ in c]
                
            coords_grouped = group_coords(coords, icoords)
            
            mod_type = [channels[c[0]]["samples"][c[1]]["modifiers"][c[2]]["type"] for c in coords]
            mod_type = np.unique(mod_type)
            
            #check if all modifiers have the same type
            if len(mod_type) != 1:
                raise ValueError("Correlated modifiers must have the same type.")
            mod_type = mod_type[0]
            
            # compute shifts for each independent eigenvector
            for i_uv, uv in enumerate(uvec.T):
                for (ich, isa), mods in coords_grouped.items():
                    uv_ind  = mods["uv_ind"]
                    mod_ind = mods["mod_ind"]
                    
                    nom = np.array(channels[ich]["samples"][isa]["data"])
                    
                    lo_diffs = np.array([np.subtract(channels[ich]["samples"][isa]["modifiers"][imo]["data"]["lo_data"], nom) for imo in mod_ind])
                    hi_diffs = np.array([np.subtract(channels[ich]["samples"][isa]["modifiers"][imo]["data"]["hi_data"], nom) for imo in mod_ind])
                    
                    lo_shift = np.sum(uv[uv_ind, np.newaxis] * lo_diffs, axis=0)
                    hi_shift = np.sum(uv[uv_ind, np.newaxis] * hi_diffs, axis=0)

                    new_lo = nom + lo_shift
                    new_hi = nom + hi_shift
                    
                    channels[ich]["samples"][isa]["modifiers"].append(
                        {
                            "name": corr["name"] + f"[{str(i_uv)}]",
                            "type": mod_type,
                            "data": {
                                "lo_data": list(new_lo),
                                "hi_data": list(new_hi)
                            }
                        }
                    )
                    
            for (ich, isa), _ in coords_grouped.items():
                new_modifiers = []
                for m in channels[ich]["samples"][isa]["modifiers"]:
                    if m["name"] not in corr["vars"]:
                        new_modifiers.append(m)
                channels[ich]["samples"][isa]["modifiers"] = new_modifiers
        
    return {"channels": channels}