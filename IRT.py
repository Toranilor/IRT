"""
A bunch of playing around testing out some IRT curve fits.
The goal is to replicate analysis from Phil, who has used 1 dimensional logistic regression to estimate the
difficulty of each quiz quesiton in our quizzes.

The main task will be a combination student ability AND item characteristic estimation.
"""

from irt_parameter_estimation import zlc_mle
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Student performance data
data_loc = "TestResponse.csv"
test_data = np.loadtxt(data_loc, delimiter=',')

# Estimate difficulty for each question
num_participants = test_data.shape[0]
num_questions = test_data.shape[1]
difficulty = np.zeros((num_questions, 0))

# Define some parameters that we'll estimate for later
theta = np.ones(num_participants)
a = 1
b_start = 0.5

for q_num in range(num_questions):
    r = test_data[:, q_num]
    f = np.ones(num_participants)
    print(zlc_mle.mle_1_parameter(theta, r, f, a, b_start))
    difficulty[q_num]

print(difficulty)