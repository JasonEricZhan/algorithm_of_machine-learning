def compute(eta,ada_sum,gradient):
    ada_sum=ada_sum+gradient**2
    ada_direction=(eta/np.sqrt(ada_sum+1e-8))*gradient
    return  ada_direction,ada_sum

"""
initailize w is important
"""
