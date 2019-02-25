"""
A series of item response theory utilities.
Written using equations from "The Basics of Item Response Theory Using R"
by F. Baker and S. Kim.

Tom Dixon - thomas.dixon@unsw.edu.au
Created: 8/01/2019
Last Modified: 10/01/2019

Some common variable definitions:
theta - ability of student(s)
r - response(s) of students (a mark) to a particular item
f - total attempt(s) for a particular item
a - discrimination of a particular item ("slope" of IRT curve roughly speaking)
b - difficulty of a particular item
c - chance of guessing the correct answer for a particular item
"""

import scipy as sp 
import numpy as np 

def estimate_ability_l1(b, u, n_iter=100, theta=0):
    """
    A function to estimate the ability of a student, assuming the test items
    follow a 1st degree logistic.

    b is a j-length vector of the difficulty for each item
    u is a j-length vector of the student's response for each item (between 0 and 1)
    n_iter is the number of iterations of ability to perform

    Using equations from page 70 of "The Basics of Item Response Theory Using R"
    """
    assert(b.shape == u.shape)

    for i in range(n_iter):
        Prob = logistic_1(theta, b)
        Q = 1-Prob
        numerator = np.sum(u-Prob)
        denominator = -np.sum(Prob*Q)
        theta -= numerator/denominator

    return theta

def estimate_ability_l2(b, a, u, n_iter=100, theta=0):
    """
    A function to estimate the ability of a student, assuming the test items
    follow a 1st degree logistic.

    b is a j-length vector of the difficulty for each item
    a is a j-length vector of the discrimination for each item
    u is a j-length vector of the student's response for each item (between 0 and 1)
    n_iter is the number of iterations of ability to perform

    Using equations from page 70 of "The Basics of Item Response Theory Using R"
    """
    assert(b.shape == u.shape)
    assert(a.shape == u.shape)

    for i in range(n_iter):
        Prob = logistic_2(theta, a, b)
        Q = 1-Prob
        numerator = np.sum(u-Prob)
        denominator = -np.sum(Prob*Q)
        theta -= numerator/denominator

    return theta


def l1_wrapper(x, **kwargs):
    """
    A wrapper function for logistic_1, to allow it to be used in scipy.optimize.
    'x' is our estimate for b
    'kwargs' contains:
        theta vector
        r/f vector
    """
    assert np.size(x) == 1
    b = x
    theta = kwargs.get('theta')
    r_on_f = kwargs.get('r_on_f')

    # We return the residual back to scipy for use in optimisation
    # (Need to reshape it from a (X,1) sized array into a 1D array...)
    residuals = np.reshape((logistic_1(theta, b) - r_on_f), np.size(theta))
    return residuals

def l2_wrapper(x, **kwargs):
    """
    A wrapper function for logistic_2, to allow it to be used in scipy.optimize.
    'x' is our estimate for a and b
    'kwargs' contains:
        theta vector
        r/f vector
    """
    assert np.size(x) == 2
    b = x[0]
    a = x[1]

    theta = kwargs.get('theta')
    r_on_f = kwargs.get('r_on_f')

    # We return the residual back to scipy for use in optimisation
    # (Need to reshape it from a (X,1) sized array into a 1D array...)
    residuals = np.reshape((logistic_2(theta, a, b) - r_on_f), np.size(theta))
    return residuals


def logistic_1(theta, b):
    """
    Function for an item response curve of 1 variable.
    Also known as a Rasch model. 
    This function returns the probability that a student of ability theta
    would correctly get the answer to a question of difficulty b. 
    It assumes that the discrimination (a) is 1.
    """
    logistic = np.asarray(theta)-np.asarray(b)
    prob = 1/(1+np.exp(-logistic))
    return prob

def logistic_2(theta, a, b):
    """
    Function for an item response curve of two variables.
    Returns the probability that a student of ability theta would correctly
    get the answer to a quesiton of difficulty b and discrimination a.
    """
    logistic = np.asarray(theta)-b
    prob = 1/(1+np.exp(-a*logistic))
    return prob


def logistic_3(theta, a, b, c):
    """
    Function for an item response curve of three variables.
    Returns the probability that a student of ability theta would correctly
    get the answer to a question of difficulty b, discrimination a, and a
    probability to guess the right answer of c.
    """
    logistic = np.asarray(theta)-b
    prob = c + (1-c)*1/(1+np.exp(-a*logistic))
    return prob
