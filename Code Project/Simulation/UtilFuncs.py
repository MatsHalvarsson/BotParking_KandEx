from tensorflow import make_ndarray, is_tensor
import numpy as np




def as_np(arrayLike):
    if type(arrayLike) in (np.ndarray, np.number):
        return arrayLike
    if is_tensor(arrayLike):
        return make_ndarray(arrayLike)
    print("Unknown type provided. Provided object:\n{}".format(str(arrayLike)))
    return None