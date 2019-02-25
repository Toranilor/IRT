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


TODO - fix the necessicty of reshaping everything before passing into functions!
"""
from IRT import IRT_Utils as IRT
import numpy as np
from matplotlib import pyplot as plt
import random
import scipy.optimize


def apply_calibration(scores, n_param=1, n_iter=1000, qns_per_student=0):
    """
    This utility applies calibration to the test:
    	dataset is a numpy array with student data, scores for each item should be out of 1
    n_param is the number of parameters of logistic model we are using.
    """
    # Assign functions based on n_param
    if n_param == 1:
        # 1 dimensional model, aka rach model
        curve_func = IRT.logistic_1
        fit_func = IRT.l1_wrapper
    if n_param == 2:
        # 2 dimensional model, with both dificulty and discrimination.
        curve_func = IRT.logistic_2
        fit_func = IRT.l2_wrapper
    if n_param == 3:
        # 3 dimensional model, difficulty, discrimination and guess-correctly chance
        curve_func = IRT.logistic_3
        fit_func = IRT.l3_wrapper

    num_students = scores.shape[0]
    num_questions = scores.shape[1]

    # Generate a score vector (total scores of all people)
    user_scores = np.zeros((num_students, 1))
    for i in range(num_students):
        user_scores[i] = np.sum(np.nan_to_num(scores[i, :]), 0)

    # Generate a difficulty vector (beginning estimate!)
    q_diff = np.zeros((num_questions, 1))
    guess_init = 0
    if n_param > 1:
        # Generate a discrimination vector
        q_disc = np.zeros((num_questions, 1))
        guess_init = [0, 1]
    if n_param > 2:
        # Generate a guessing vector
        q_chance = np.zeros((num_questions, 1))
        guess_init = [0, 1, 0.25]

    sum_score = np.zeros((num_questions, 1))
    for i in range(num_questions):
        q_diff[i] = 3-np.sum(scores[:, i])/num_students*6
        if n_param > 1:
            q_disc[i] = 1
        if n_param > 2:
            q_chance[i] = 0.25
        sum_score[i] = np.sum(np.nan_to_num(scores[:, i]), 0)

    if qns_per_student == 0:
        # You haven't passed a value for questions_per_student, so I 
        # am assuming that it's the number of qeustions - every student takes every question
        qns_per_student = num_questions

    ### Begin the two-part iterative process!
    # Generate an initial estimate for student ability values (theta)

    theta_estimates = np.log(user_scores/(qns_per_student-user_scores))[:, 0]
    print('Beginning Iterations')

    # Start lists for residuals
    theta_resids = list()
    q_resids = list()

    for i in range(n_iter):
        print("iteration" + str(i))
        # Perform an estimate of question difficulty based on student responses
        q_diff_prev = np.copy(q_diff)
        q_resids.append(np.linalg.norm(q_diff-q_diff_prev))
        for j in range(num_questions):
            # Compress responses (remove students who did not attempt this q)
            inclusion_IDs = np.argwhere(~np.isnan(scores[:, j]))
            # Extract the relevant thetas
            theta_temp = theta_estimates[inclusion_IDs]
            scores_temp = scores[inclusion_IDs, j]

            # Group close-by thetas together (?)

            # Generate kwargs to pass to logistic function
            kwargs = {
                "theta": theta_temp,
                "r_on_f": scores_temp
            }

            q_guess = scipy.optimize.least_squares(fit_func, x0=guess_init, kwargs=kwargs).x

            q_diff[j] = q_guess[0]
            if n_param > 1:
                q_disc[j] = q_guess[1]
            if n_param > 2:
                q_chance = q_guess[2]
        q_resids.append(np.linalg.norm(q_diff-q_diff_prev))
        # "Anchor" our metric
        q_diff = q_diff - np.mean(q_diff)

        # Perform an estimate of student ability based on question difficulty 
        theta_prev = np.copy(theta_estimates)     
        for j in range(num_students):
            # Compress responses again (remove questions the student did not attempt)
            inclusion_IDs = np.argwhere(~np.isnan(scores[j, :]))
            if n_param == 1:
                theta_estimates[j] = IRT.estimate_ability_l1(
                    theta=theta_estimates[j], b=q_diff[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), 
                    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
            if n_param == 2:
                theta_estimates[j] = IRT.estimate_ability_l2(
                    theta=theta_estimates[j], b=q_diff[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), 
                    a=q_disc[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
            if n_param == 3:
                theta_estimates[j] = IRT.estimate_ability_l3(
                    theta=theta_estimates[j], b=q_diff[inclusion_IDs], 
                    a=q_disc[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                    c=q_chance[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
        theta_resids.append(np.linalg.norm(theta_estimates-theta_prev))


    # Another correction from the book
    theta_estimates = theta_estimates*(num_questions-2)/(num_questions-1)
    q_characteristics = list()
    q_characteristics.append(q_diff)

    if n_param > 1:
        q_characteristics.append(q_disc)
    if n_param > 2:
        q_characteristics.append(q_chance)

    plt.plot(theta_resids) 
    plt.xlabel("Iterations") 
    plt.ylabel("Residuals")
    plt.title("Theta Estimates Convergence")  
    plt.figure()
    plt.plot(q_resids) 
    plt.xlabel("Iterations") 
    plt.ylabel("Residuals")
    plt.title("difficulty Estimates Convergence")  

    return q_characteristics, theta_estimates