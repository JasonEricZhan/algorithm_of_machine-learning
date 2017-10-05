def compute(eta,ada_value,gradient):
    ada_value=ada_value+gradient**2
    ada_direction=(eta/np.sqrt(ada_value+1e-8))*gradient
    return  ada_direction,ada_value

"""
initailize w is important
"""
