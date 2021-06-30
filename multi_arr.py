import numpy as np

import util

class MultiArr():
    """
    To keep N arrays each associated with a string key.
    """
    def __init__(self, data):
        if type(data) is not dict:
            raise Exception('Illegal argument; not a dict')
        self.data = data
    def from_DA(da):
        return MultiArr(da)
    def from_AD(ad):
        return MultiArr(util.arr_dict2dict_arr(np.array(ad)))
        
    def get_arr_dict(self,):
        return util.dict_arr2arr_dict(self.data)
    def get_dict_arr(self,):
        return self.data
    @property
    def shape(self):
        return list(self.data.values())[0].shape
    @property
    def n_metrics(self):
        return len(self.data.keys())
    @property
    def metrics(self):
        return list(self.data.keys())
    
    def __getitem__(self, i):
        if type(i) is str:
            return self.data[i]
        else:
            return MultiArr({key: val[i] for key, val in self.data.items()})
        
    def reshape(self, shape):
        return MultiArr({key: val.reshape(shape) for key, val in self.data.items()})
    def apply(self, npfunc, **kwargs):
        return MultiArr({key: npfunc(val, **kwargs) for key, val in self.data.items()})
    def mean(self, axis=None):
        return self.apply(np.mean, axis=axis)
    def std(self, axis=None):
        return self.apply(np.std, axis=axis)
    
    def stack(fds, axis=0):
        return MultiArr({key: np.stack([fd.data[key] for fd in fds if fd is not None], axis=axis)
                        for key in fds[1].data.keys()})
    def cat(fds, axis=0):
        return MultiArr({key: np.concatenate([fd.data[key] for fd in fds if fd is not None], axis=axis)
                        for key in fds[1].data.keys()})
    
        

