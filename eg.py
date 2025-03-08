import transforms3d
import numpy as np
import torch
meta_o = np.array([0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205])
meta_t = np.array([1761.278564453125, -5078.0068359375, 1606.2650146484375])/1000
meta_R = transforms3d.quaternions.quat2mat(meta_o)

h36m_t = np.array([251.42516271750836, 420.9422103702068, 5588.195881837821])/1000

h36m_R = np.array([[0.9281683400814921, 0.3721538354721445, 0.002248380248018696],
          [0.08166409428175585, -0.1977722953267526, -0.976840363061605],
          [-0.3630902204349604, 0.9068559102440475, -0.21395758897485287]])
print(meta_R)
            
print(h36m_t@-h36m_R)
print(np.rad2deg(transforms3d.euler.mat2euler(h36m_R.T)))


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

P = np.array([-10, 0, 1])
print(world_to_camera(P, meta_o, meta_t))
print((P - meta_t)@h36m_R.T)
