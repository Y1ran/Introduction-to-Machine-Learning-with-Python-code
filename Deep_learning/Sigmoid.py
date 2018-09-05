# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 16:13:45 2018

@author: Administrator
"""

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-(z)))
    ### END CODE HERE ###
    
    return s