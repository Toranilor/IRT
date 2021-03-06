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
import logging


def apply_calibration(scores, n_param=1, n_iter=1000, qns_per_student=0, shunt_values=False):
    """
    This utility applies calibration to the test:
    	dataset is a numpy array with student data, scores for each item should be out of 1
    n_param is the number of parameters of logistic model we are using.
    """

    # Start the logger.
    logging.basicConfig(filename="Debug.log", level=logging.DEBUG, filemode='w+')
    logging.debug('Beginning of Logging')

    # Set a parameter absolute value that will prompt an error message
    error_range = 10


    # Assign functions based on n_param
    if n_param == 1:
        # 1 dimensional model, aka rach model
        curve_func = IRT.logistic_1
        fit_func = IRT.l1_wrapper
        disc = False
        chance = False
    if n_param == 2:
        # 2 dimensional model, with both dificulty and discrimination.
        curve_func = IRT.logistic_2
        fit_func = IRT.l2_wrapper
        disc = True
        chance = False
    if n_param == 3:
        # 3 dimensional model, difficulty, discrimination and guess-correctly chance
        curve_func = IRT.logistic_3
        fit_func = IRT.l3_wrapper
        disc = True
        chance = True

    num_students = scores.shape[0]
    num_questions = scores.shape[1]

    # Generate a score vector (total scores of all people)
    user_scores = np.zeros((num_students, 1))
    for i in range(num_students):
        user_scores[i] = np.sum(np.nan_to_num(scores[i, :]), 0)

    # Generate a difficulty vector (beginning estimate!)
    q_diff = np.zeros((num_questions, 1))
    guess_init = 0
    if disc:
        # Generate a discrimination vector
        q_disc = np.zeros((num_questions, 1))
        guess_init = [0, 1]
    if chance:
        # Generate a guessing vector
        q_chance = np.zeros((num_questions, 1))
        guess_init = [0, 1, 0.25]

    sum_score = np.zeros((num_questions, 1))
    for i in range(num_questions):
        q_diff[i] = 3-np.nanmean(scores[:, i])*6
        if disc:
            q_disc[i] = 1
        if chance:
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
    logging.debug("Initial Theta Estimates:")
    logging.debug(list(theta_estimates))
    logging.debug("Initial difficulty Estimates")
    logging.debug(list(q_diff))
    if disc:
    	logging.debug("Initial discrimination Estimates")
    	logging.debug(q_disc)
    # Start lists for residuals
    theta_resids = list()
    q_resids = list()
    q_guess = guess_init
    for i in range(n_iter):
        print("iteration" + str(i))
        logging.debug("iteration" + str(i))
        # Perform an estimate of question difficulty based on student responses
        q_diff_prev = np.copy(q_diff)
        q_resids.append(np.linalg.norm(q_diff-q_diff_prev))
        for j in range(num_questions):
            # Compress responses (remove students who did not attempt this q)
            inclusion_IDs = np.argwhere(~np.isnan(scores[:, j]))
            # Extract the relevant thetas
            theta_temp = theta_estimates[inclusion_IDs]
            scores_temp = scores[inclusion_IDs, j]

            # Generate kwargs to pass to logistic function
            kwargs = {
                "theta": theta_temp,
                "r_on_f": scores_temp
            }
            # Generate our starting guess (it's the previous value)
            guess = list(q_diff[j])
            if n_param == 2:
            	guess.append(q_disc[j])
            elif n_param == 3:
            	guess.append(q_chance[j])
            q_guess = scipy.optimize.least_squares(fit_func, x0=guess, kwargs=kwargs).x

            q_diff[j] = q_guess[0]
            if disc:
                q_disc[j] = q_guess[1]
            if chance:
                q_chance[j] = q_guess[2]
        # Create the residuals and append them  
        q_resids.append(np.linalg.norm(q_diff-q_diff_prev))

        # Check if we have any values that are outside our norm and exit!
        q_set = np.arange(np.size(q_diff))
        diff_errors = np.array(np.abs(q_diff)>error_range)[:,0]
        if True in diff_errors:
        	logging.debug("Difficulty out of bounds!")
        	logging.debug("Questions " + str(q_set[diff_errors]) + " have values of:")
        	logging.debug(q_diff[diff_errors])
        	if shunt_values:
        		print("Difficulty" + str(q_set[diff_errors]) + "Shunted!")
        		q_diff[diff_errors] = 0
        	else:
        		print("RUN TERMINATED DUE TO OUT OF BOUNDS ERROR - CHECK LOGS")
        		return 0, 0

        if disc:
        	disc_erors = np.array(np.abs(q_disc)>error_range)[:,0]
        	if True in disc_erors:
        		logging.debug("Discrimination out of bounds!")
       			logging.debug("Questions " + str(q_set[disc_erors]) + " have values of:")
       			logging.debug(q_disc[disc_erors])
       			if shunt_values:
       				print("Discrimination"  + str(q_set[disc_erors]) + " shunted!")
       				q_disc[disc_erors] = 1
       			else:
       				print("RUN TERMINATED DUE TO OUT OF BOUNDS ERROR - CHECK LOGS")
       				return 0, 0
        # "Anchor" our metric
        q_diff = q_diff - np.mean(q_diff)
        logging.debug('q_diff:')
        logging.debug(list(q_diff))
        if disc:
        	logging.debug('q_disc')
        	logging.debug(list(q_disc))
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
            	## The L2 estimate ability function doesn't work - at the moment I only use the l1.
            	## This means that the discrimination doesn't impact on our value for theta.
                #theta_estimates[j] = IRT.estimate_ability_l2(
                #    theta=theta_estimates[j], b=q_diff[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), 
                #    a=q_disc[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                #    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
                theta_estimates[j] = IRT.estimate_ability_l1(
                    theta=theta_estimates[j], b=q_diff[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), 
                    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
            if n_param == 3:
                theta_estimates[j] = IRT.estimate_ability_l3(
                    theta=theta_estimates[j], b=q_diff[inclusion_IDs], 
                    a=q_disc[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                    c=q_chance[inclusion_IDs].reshape((np.size(inclusion_IDs), 1)),
                    u=scores[j, inclusion_IDs].reshape((np.size(inclusion_IDs), 1)), n_iter=1)
        theta_resids.append(np.linalg.norm(theta_estimates-theta_prev))

        # Check for a value exceeding our error check threshold
        theta_set = np.arange(np.size(theta_estimates))
        theta_errors = np.abs(theta_estimates)>error_range
        if True in theta_errors:
        	logging.debug("Theta estimate out of bounds!")
        	logging.debug("Students " + str(theta_set[theta_errors]) + " have values of:")
        	logging.debug(theta_estimates[theta_errors])
        	if shunt_values:
        		print("Theta " + str(theta_set[theta_errors]) + "Shunted!")
        		theta_estimates[theta_errors] = 0
        	else:
        		print("RUN TERMINATED DUE TO OUR OF BOUNDS ERROR - CHECK LOGS")
        		return 0, 0 

        logging.debug('Theta:')
        logging.debug(list(theta_estimates))

    # Another correction from the book
    theta_estimates = theta_estimates*(num_questions-2)/(num_questions-1)
    q_characteristics = list()
    q_characteristics.append(q_diff)

    if disc:
        q_characteristics.append(q_disc)
    if chance:
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