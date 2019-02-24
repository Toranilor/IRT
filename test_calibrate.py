"""
An item-reponse-theory based test calibration funcito set.
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
import IRT_Utils as IRT 
import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.optimize


def apply_calibration(dataset, n_param=1):
	"""
	This utility applies calibration to the test:
		dataset is a numpy array with student data 
	"""

