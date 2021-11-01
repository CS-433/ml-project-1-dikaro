import numpy as np
import matplotlib.pyplot as plt
import proj1_helpers as prjhlp
import helpers_data as hlpdata


### Helper Functions for LS, Ridge and Linear Regression

def compute_MSE(y, tx, w):
    """ Computes MSE loss function and returns it """
    e = y - np.dot(tx, w)
    return np.mean(e ** 2, axis=0) / 2

def compute_gradient(y, tx, w):
    """ Computes gradient and returns it """
    e = y - np.dot(tx, w)
    return -(np.dot(tx.transpose(), e)) / len(y)


### Helper Functions for Logistic Regression

def sigmoid(x):
    """ Compute Sigmoid Function of input and returns it """
    return 1 / (1 + np.exp(-x))

def compute_loss_LogReg(y, tx, w):
    """ Compute Loss for Logistic Regression and returns it """
    sig = sigmoid(np.dot(tx, w))
    return -np.sum(y * np.log(sig) + (1 - y) * np.log(1 - sig)) / len(y)

def compute_loss_LogReg_approx(y, tx, w):
    """ Compute Approximation for Loss of Logistic Regression and returns it """
    return np.sum(np.dot(tx, w) * (1-y)) / len(y)

def compute_grad_LogReg(y, tx, w):
    """ Compute gradient of loss for logistic regression and returns it """
    return np.dot(tx.transpose(), sigmoid(np.dot(tx, w)) - y)

def compute_hessian(tx, w):
    """ Compute hessian matrix of logistic regression for Newton's method and
        returns it """
    sig = sigmoid(np.dot(tx, w))
    diag_vals = np.ravel(sig * (1 - sig))
    return np.dot(tx.transpose(), np.dot(np.diag(diag_vals), tx))


### Helper Functions for Mini-batch SGD

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generation of a minibatch iterator for a dataset """
    data_size = len(y)
    # shuffle
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    # generation of iterator
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def batch_data(y, tx, batch_size, max_iters):
    """ Generation of batches of dataset for each iteration and return complete batches as arrays """
    shufs_y = []
    shufs_tx = []
    for i in range(max_iters):
        shufs = next(batch_iter(y, tx, batch_size, 1, False))
        shufs_y.append(shufs[0])
        shufs_tx.append(shufs[1])
    return shufs_y, shufs_tx


### Helper Function for Cross Validation Method

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-folds """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

### Encoding/Decoding of Output Vector To Implement Logistic Regression
def encode_y(y):
    """ Transform -1 values of y to 1 and 1 values of y to 1 """
    return (y + 1) / 2
def decode_y(y):
    """ Transform 0 values of y to -1 and 1 values of y to 1 """
    return 2 * y - 1


### Basic 6 Functions Required to be Implemented

# Basic Least Squares GD
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Implements linear regression using Gradient Descent and returns last
        value of weight parameters and loss """
    # parameters
    w = initial_w
    # GD algorithm
    for n_iter in range(max_iters):
        print("GD linear Reg. - {it}/{maxit}".format(it=n_iter+1, maxit=max_iters))
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_MSE(y, tx, w)
    return w, loss

# Basic Least Squares SGD
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Implements linear regression using Stochastic Gradient Descent
                and returns last value of weight parameters and loss """
    # parameters
    w = initial_w
    shufs_y, shufs_tx = batch_data(y, tx, 1, 1, False)
    # SGD algorithm
    for n_iter in range(max_iters):
        if n_iter % 1000 == 0:
            print("SGD ({bi}/{ti})".format(bi=n_iter + 1,
                        ti=max_iters))
        grad = compute_gradient(shufs_y[n_iter], shufs_tx[n_iter], w)
        w = w - gamma * grad
    loss = compute_MSE(y, tx, w)
    return w, loss

# Basic Least Squares
def least_squares(y, tx):
    """ Implements Least Squares regression using normal equations
            and returns the weight parameter and loss """
    w = np.linalg.solve(np.dot(tx.transpose(), tx), np.dot(tx.transpose(), y))
    loss = compute_MSE(y, tx, w)
    return w, loss

# Basic Ridge Regression
def ridge_regression(y, tx, lambda_):
    """ Implements Ridge regression using normal equations
            and returns the weight parameter and loss """
    w = np.linalg.solve(np.dot(tx.transpose(), tx) + (lambda_ * 2 * y.shape[0]) * np.identity(tx.shape[1]),
                        np.dot(tx.transpose(), y))
    loss = compute_MSE(y, tx, w)
    return w, loss

# Basic Logistic Regression Using GD
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Implements Logistic regression using gradient descent
            and returns the weight parameter and loss """
    w = initial_w
    for n_iter in range(max_iters):
        print("Log Reg - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_GD_logReg(y, tx, w, gamma)
    loss = compute_loss_LogReg(y, tx, w)
    return w, loss

# Basic Regularized Logistic Regression Using GD
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        print("Regularized Log. Reg - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    loss = compute_loss_LogReg(y, tx, w) + lambda_ * np.sum(w ** 2) / 2
    return w, loss



### CUSTOMIZED FUNCTIONS OF LS, RIDGE AND LINEAR REGRESSION USING GD/MINI-BATCH SGD

# Customized Least Squares GD
def least_squares_GD_cust(y, tx, initial_w, max_iters, gamma):
    """ Implements linear regression using Gradient Descent and returns all
        values of weight parameters """
    # parameters
    w = initial_w
    ws = [w]
    # GD algorithm
    for n_iter in range(max_iters):
        if n_iter % 100 == 0:
            print("GD linear Reg. - {it}/{maxit}".format(it=n_iter+1, maxit=max_iters))
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        ws.append(w)
    return ws

# Customized Mini Batch SGD Linear Regression
def least_squares_minibatch_SGD_cust(shufs_y, shufs_tx, initial_w, batch_size, max_iters, gamma):
    """ Implements linear regression using mini-batch Stochastic Gradient Descent
        and returns all values of weight parameters """
    # parameters
    w = initial_w
    ws = []
    # SGD algorithm
    for n_iter in range(max_iters):
        if n_iter % 1000 == 0:
            print("Mini Batch" + str(batch_size) + " - Stochastic Gradient Descent({bi}/{ti})".format(bi=n_iter + 1,
                        ti=max_iters))
        grad = compute_gradient(shufs_y[n_iter], shufs_tx[n_iter], w)
        w = w - gamma * grad
        ws.append(w)
    return ws


### Search for Hyper-parameter Functions

def grid_search_degree_LS(y, tx, degrees, feats, rem_inds):
    """ Implements grid search algorithm to find the best degree parameter 
        which maximizes accuracy values for LS model """
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

    for degree in degrees:
        y, tX, ms, stds = hlpdata.preprocess_dataset(y, tx, feats, degree, rem_inds)
        tx_tr, tx_te, y_tr, y_te = prjhlp.split_data(tX, y, 0.9, seed=1)

        # Implementation of ML Algorithm
        w, l_tr = least_squares(y_tr, tx_tr)

        # Getting the Statistics
        acc_tr = get_accs(y_tr, tx_tr, [w])
        acc_te = get_accs(y_te, tx_te, [w])
        l_te = get_losses(y_te, tx_te, [w])

        acc_tr = acc_tr[-1]
        acc_te = acc_te[-1]
        l_te = l_te[-1]

        print(
            "LS degree={deg}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
                deg=degree,
                l_tr=l_tr, acc_tr=acc_tr, l_te=l_te, acc_te=acc_te))
        accs_tr.append(acc_tr)
        accs_te.append(acc_te)
        losses_tr.append(l_tr)
        losses_te.append(l_te)

    plt.figure()
    plt.plot(degrees, accs_tr, label="acc_tr")
    plt.plot(degrees, accs_te, label="acc_te")
    plt.title("Accuracy vs. degree for LS")
    plt.ylabel("Accuracy")
    plt.xlabel("Degree")
    plt.legend()
    plt.figure()
    plt.plot(degrees, losses_tr, label="loss_tr")
    plt.title("Loss_tr vs. degree for LS")
    plt.ylabel("Loss")
    plt.xlabel("Degree")
    plt.figure()
    plt.plot(degrees, losses_te, label="loss_te")
    plt.title("Loss_te vs. degree for LS")
    plt.ylabel("Loss")
    plt.xlabel("Degree")
    plt.show()

    return degrees[np.argmax(accs_te)]

def grid_search_degree_RG(y, tx, degree, lambdas, feats, rem_inds):
    """ Implements grid search algorithm to find the best lambda parameter 
        which maximizes accuracy values for RG model """
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

    
    for lambda_ in lambdas:
            y, tX, ms, stds = hlpdata.preprocess_dataset(y, tx, feats, degree, rem_inds)
            tx_tr, tx_te, y_tr, y_te = prjhlp.split_data(tX, y, 0.9, seed=1)

            # Implementation of ML Algorithm
            w, l_tr = ridge_regression(y_tr, tx_tr, lambda_)

        # Getting the Statistics
            acc_tr = get_accs(y_tr, tx_tr, [w])
            acc_te = get_accs(y_te, tx_te, [w])
            l_te = get_losses(y_te, tx_te, [w])

            acc_tr = acc_tr[-1]
            acc_te = acc_te[-1]
            l_te = l_te[-1]
            
            print(
            "RG lambda={deg}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
                deg=lambda_,
                l_tr=l_tr, acc_tr=acc_tr, l_te=l_te, acc_te=acc_te))
            accs_tr.append(acc_tr)
            accs_te.append(acc_te)
            losses_tr.append(l_tr)
            losses_te.append(l_te)

    plt.figure()
    plt.plot(lambdas, accs_tr, label="acc_tr")
    plt.plot(lambdas, accs_te, label="acc_te")
    plt.title("Accuracy vs. lambda for RG")
    plt.ylabel("Accuracy")
    plt.xlabel("lambda")
    plt.legend()
    plt.figure()
    plt.plot(lambdas, losses_tr, label="loss_tr")
    plt.plot(lambdas, losses_te, label="loss_te")
    plt.title("Loss vs. lambda for RG")
    plt.ylabel("Loss")
    plt.xlabel("lambda")
    plt.legend()
    plt.show()

    return lambdas[np.argmax(accs_te)]


### Cross Validation Functions

def cross_validation_LS(y, x, k_fold):
    """ Implements Least Squares regression using normal equations with
        Cross Validation, plots the accuracy and loss graphs for each
        iteration in cross validation and returns the average of
        weight parameters """

    # Parameters
    k_indices = build_k_indices(y, k_fold, seed=1)
    ws = []
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

    # Iteration For Cross Validation
    for k in range(k_fold):
        print("LS Cross Validation: {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.ravel(np.delete(k_indices, k, 0))

        te_y = y[inds_te]
        te_x = x[inds_te]
        tr_y = y[inds_tr]
        tr_x = x[inds_tr]

        # Implementation of ML Algorithm
        w, l_tr = least_squares(tr_y, tr_x)

        # Getting the Statistics
        acc_tr = get_accs(tr_y, tr_x, [w])
        acc_te = get_accs(te_y, te_x, [w])
        l_te = get_losses(te_y, te_x, [w])
        acc_te = acc_te[-1]
        acc_tr = acc_tr[-1]
        l_te = l_te[-1]
        print("LS k={k}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(k=k,
                                l_tr=l_tr, acc_tr=acc_tr, l_te=l_te, acc_te=acc_te))
        ws.append(w)
        accs_tr.append(acc_tr)
        accs_te.append(acc_te)
        losses_tr.append(l_tr)
        losses_te.append(l_te)

    # Getting the Average of Weight Parameters
    w_avg = np.mean(ws, axis=0)

    acc_tr_avg = np.mean(accs_tr)
    acc_tr_var = np.var(accs_tr)
    acc_te_avg = np.mean(accs_te)
    acc_te_var = np.var(accs_te)

    l_tr_avg = np.mean(losses_tr)
    l_te_avg = np.mean(losses_te)

    print("LS of resulted average w: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}, var_train = {v_tr}, var_test = {v_te}".format(
                                l_tr=l_tr_avg, acc_tr=acc_tr_avg, l_te=l_te_avg, acc_te=acc_te_avg, v_tr=acc_tr_var, v_te=acc_te_var))
    # Visualization of the Statistics
    ks = np.arange(k_fold) + 1
    plots_accs_losses_LSorRG(ks, accs_tr, accs_te, losses_tr, losses_te, "LS", acc_tr_avg, acc_te_avg, l_tr_avg, l_te_avg)

    return w_avg

def cross_validation_RG(y, x, k_fold, lambda_):
    """ Implements Ridge regression using normal equations with
        Cross Validation, plots the accuracy and loss graphs for each
        iteration in cross validation and returns the average of
        weight parameters """

    # Parameters
    k_indices = build_k_indices(y, k_fold, seed=1)
    ws = []
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []

    # Iteration for Cross Validation
    for k in range(k_fold):
        print("RG Cross Validation: {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.ravel(np.delete(k_indices, k, 0))

        te_y = y[inds_te]
        te_x = x[inds_te]
        tr_y = y[inds_tr]
        tr_x = x[inds_tr]

        # Implementation of ML algorithm
        w, l_tr = ridge_regression(tr_y, tr_x, lambda_)

        # Getting the Statistics
        acc_tr = get_accs(tr_y, tr_x, [w])
        acc_te = get_accs(te_y, te_x, [w])
        l_te = get_losses(te_y, te_x, [w])

        acc_tr = acc_tr[-1]
        acc_te = acc_te[-1]
        l_te = l_te[-1]

        print("RG k={k}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(k=k,
                                        l_tr=l_tr, acc_tr=acc_tr, l_te=l_te, acc_te=acc_te))
        ws.append(w)
        accs_tr.append(acc_tr)
        accs_te.append(acc_te)
        losses_tr.append(l_tr)
        losses_te.append(l_te)

    # Getting the Average of Weight Parameters
    w_avg = np.mean(ws, axis=0)
    acc_tr_avg = get_accs(tr_y, tr_x, [w_avg])
    acc_te_avg = get_accs(te_y, te_x, [w_avg])
    l_tr_avg = get_losses(tr_y, tr_x, [w_avg])
    l_te_avg = get_losses(te_y, te_x, [w_avg])
    acc_tr_avg = acc_tr_avg[-1]
    acc_te_avg = acc_te_avg[-1]
    l_tr_avg = l_tr_avg[-1]
    l_te_avg = l_te_avg[-1]

    acc_tr_var = np.var(accs_tr)
    acc_te_var = np.var(accs_te)
    print(
        "RG of resulted average w: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}, var_train = {v_tr}, var_test = {v_te}".format(
            l_tr=l_tr_avg, acc_tr=acc_tr_avg, l_te=l_te_avg, acc_te=acc_te_avg, v_tr=acc_tr_var, v_te=acc_te_var))
    ks = np.arange(k_fold) + 1

    # Visualization of the Statistics
    plots_accs_losses_LSorRG(ks, accs_tr, accs_te, losses_tr, losses_te, "RG", acc_tr_avg, acc_te_avg, l_tr_avg, l_te_avg)

    return w_avg

def cross_validation_least_squares_GD(y, tX, gamma, max_iters, k_fold, div, pol_degree):
    """ Implements linear regression using
        gradient descent with Cross Validation, plots the accuracy
        and loss graphs for each iteration in cross validation and
        returns the last weight parameters """

    # Parameters
    k_indices = build_k_indices(y, k_fold, True)
    iters_per_fold = int(max_iters / k_fold)
    w = np.zeros(tX.shape[1])
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []
    tot = 0

    # Iteration for Cross Validation
    for k in range(k_fold):
        print("Cross Validation: {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.delete(k_indices, k, 0)
        inds_tr = inds_tr.ravel()

        tX_tr = tX[inds_tr]
        y_tr = y[inds_tr]
        tX_te = tX[inds_te]
        y_te = y[inds_te]

        # Implemention of ML algorithm
        ws_LS = least_squares_GD_cust(y_tr, tX_tr, w, iters_per_fold, gamma)
        w = ws_LS[-1]

        # Getting the Statistics
        accs_tr, accs_te, losses_tr, losses_te = get_statistics_per_k_LS(k, y_tr, tX_tr, y_te, tX_te, ws_LS,
                                                                                div, accs_tr, accs_te,
                                                                                losses_tr,
                                                                                losses_te)
        tot = len(ws_LS) + tot

    # Visualization of the Statistics
    plot_accs_losses(accs_tr, accs_te, losses_tr, losses_te,
                            "Lin. Regr. Using GD \nfor pol. degree = " + str(pol_degree), div, tot)
    print("\naccs_tr var", np.var(accs_tr)," accs_te_var", np.var(accs_te))
    return w


def cross_validation_least_squares_SGD(y, tX, gamma, batch_size, max_iters, k_fold, div, pol_degree):
    """ Implements linear regression using mini-batch stochastic
        gradient descent with Cross Validation, plots the accuracy
        and loss graphs for each iteration in cross validation and
        returns the last weight parameters """

    # Parameters
    k_indices = build_k_indices(y, k_fold, True)
    iters_per_fold = int(max_iters / k_fold)
    w = np.zeros(tX.shape[1])
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []
    tot = 0

    # Iteration for Cross Validation
    for k in range(k_fold):
        print("Cross Validation: {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.delete(k_indices, k, 0)
        inds_tr = inds_tr.ravel()

        tX_tr = tX[inds_tr]
        y_tr = y[inds_tr]
        tX_te = tX[inds_te]
        y_te = y[inds_te]

        # Generation of Batches
        shufs_y, shufs_tx = batch_data(y_tr, tX_tr, batch_size, iters_per_fold)

        # Implemention of ML algorithm
        ws_LS = least_squares_minibatch_SGD_cust(shufs_y, shufs_tx, w, batch_size, iters_per_fold, gamma)
        w = ws_LS[-1]

        # Getting the Statistics
        accs_tr, accs_te, losses_tr, losses_te = get_statistics_per_k_LS(k, y_tr, tX_tr, y_te, tX_te, ws_LS,
                                                                                div, accs_tr, accs_te,
                                                                                losses_tr,
                                                                                losses_te)
        tot = len(ws_LS) + tot

    # Visualization of the Statistics
    plot_accs_losses(accs_tr, accs_te, losses_tr, losses_te,
                            "Lin. Regr. \nfor pol. degree = " + str(pol_degree) + ", batch size = " + str(
                                batch_size), div, tot)
    print("\naccs_tr var", np.var(accs_tr)," accs_te_var", np.var(accs_te))
    return w



### CUSTOMIZED FUNCTIONS OF LOGISTIC REGRESSION WITH/WITHOUT REGULARIZATION USING GD/MINI-BATCH SGD

# Learning Function of Logistic Regression
def learning_by_GD_logReg(y, tx, w, gamma):
    """ Implement learning algorithm for one step and returns the updated
        weight parameter """
    grad = compute_grad_LogReg(y, tx, w)
    w = w - gamma * grad
    return w

# Generation of Necessary Values for Newton's Method
def logistic_regression_Newton(y, tx, w):
    """ Generates Hessian matrix, loss, and gradient for Newton's method and
        returns them. """
    H = calculate_hessian(tx, w)
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    return loss, grad, H

# Customized Function for
def logistic_regression_cust(y, tx, w, max_iters, gamma):
    ws = []
    for n_iter in range(max_iters):
        if n_iter+1 % 100 == 0:
            print("Log Reg - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_GD_logReg(y, tx, w, gamma)
        ws.append(w)
    return ws
def logistic_regression_SGD(shufs_y, shufs_tx, w, max_iters, gamma):
    ws = []
    for n_iter in range(max_iters):
        if n_iter % 1000 == 1:
            print("Log Reg - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_GD_logReg(shufs_y[n_iter], shufs_tx[n_iter], w, gamma)
        ws.append(w)
    return ws

# Newton's Method
def learning_by_newton_method(y, tx, w, gamma):
    loss, grad, H = logistic_regression_Newton(y, tx, w)
    w = w - gamma * np.linalg.solve(H, grad)
    return loss, w
def newton_method(y, tx, w, max_iters, gamma):
    ws = []
    losses = []
    for n_iter in range(max_iters):
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        ws.append(w)
        losses.append(loss)
        print("Newton - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
    return ws

# Regularized Logistic Regression
def penalized_logistic_regression(y, tx, w, lambda_):
    #H = compute_hessian(tx, w) + lambda_ * np.eye(tx.shape[1])
    #loss = compute_loss_LogReg(y, tx, w) + lambda_ * np.sum(w ** 2)/2
    grad = compute_grad_LogReg(y, tx, w) + lambda_ * w
    return grad#, H
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    #grad, H = penalized_logistic_regression(y, tx, w, lambda_)
    #w = w - gamma * np.linalg.solve(H, grad)
    grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return w
def logistic_regression_regularized_GD(y, tx, w, max_iters, gamma, lambda_):
    ws = []
    for n_iter in range(max_iters):
        if (n_iter+1) % 1000 == 0:
            print("Regularized Log. Reg - GD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        ws.append(w)
    return ws
def logistic_regression_regularized_SGD(shufs_y, shufs_tx, w, max_iters, gamma, lambda_):
    ws = []
    for n_iter in range(max_iters):
        if (n_iter+1) % 1000 == 0:
            print("Regularized Log. Reg - SGD - iter: {it}/{maxit}".format(it=n_iter + 1, maxit=max_iters))
        w = learning_by_penalized_gradient(shufs_y[n_iter], shufs_tx[n_iter], w, gamma, lambda_)
        ws.append(w)
    return ws


# Cross Validation Functions

def cross_validation_log_reg_GD(y, tX, gamma, max_iters, k_fold, div, pol_degree):
    """ Implements logistic regression using
        gradient descent with Cross Validation, plots the accuracy
        and loss graphs for each iteration in cross validation and
        returns the last weight parameters """

    # Parameters
    k_indices = build_k_indices(y, k_fold, True)
    iters_per_fold = int(max_iters / k_fold)
    w = np.zeros(tX.shape[1])
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []
    tot = 0

    # Iteration for Cross Validation
    for k in range(k_fold):
        print("Cross Validation: {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.delete(k_indices, k, 0)
        inds_tr = inds_tr.ravel()

        tX_tr = tX[inds_tr]
        y_tr = y[inds_tr]
        tX_te = tX[inds_te]
        y_te = y[inds_te]

        y_tr = encode_y(y_tr)
        y_te = encode_y(y_te)

        # Implemention of ML algorithm
        ws_LR = logistic_regression_cust(y_tr, tX_tr, w, iters_per_fold, gamma)
        w = ws_LR[-1]

        # Getting the Statistics
        accs_tr, accs_te, losses_tr, losses_te = get_statistics_per_k(k, y_tr, tX_tr, y_te, tX_te, ws_LR, div, 0, accs_tr, accs_te, losses_tr, losses_te)
        tot = len(ws_LR) + tot

    # Visualization of the Statistics
    plot_accs_losses(accs_tr, accs_te, losses_tr, losses_te,
                            "Log. Regr. Using GD \nfor pol. degree = " + str(pol_degree), div, tot)
    print("\naccs_tr var", np.var(accs_tr)," accs_te_var", np.var(accs_te))
    return w


 # Regularized Logistic Regression using Cross Validation & SGD
def cross_validation_log_regr_SGD(y, tX, gamma, batch_size, max_iters, k_fold, div, pol_degree):
    return cross_validation_log_regr_reg_SGD(y, tX, gamma, 0, batch_size, max_iters, k_fold, div, pol_degree)
def cross_validation_log_regr_reg_SGD(y, tX, gamma, lambda_, batch_size, max_iters, k_fold, div, pol_degree):
    k_indices = build_k_indices(y, k_fold, True)
    iters_per_fold = int(max_iters/k_fold)
    w = np.zeros(tX.shape[1])
    accs_tr = []
    accs_te = []
    losses_tr = []
    losses_te = []
    tot = 0
    for k in range(k_fold):
        print("Cross Validation: k = {k}/{kfold}".format(k=k + 1, kfold=k_fold))
        inds_te = k_indices[k]
        inds_tr = np.delete(k_indices, k, 0)
        inds_tr = inds_tr.ravel()

        tX_tr = tX[inds_tr]
        y_tr = y[inds_tr]
        tX_te = tX[inds_te]
        y_te = y[inds_te]

        y_tr = encode_y(y_tr)
        y_te = encode_y(y_te)

        shufs_y, shufs_tx = batch_data(y_tr, tX_tr, batch_size, iters_per_fold)

        ws_LR_reg = logistic_regression_regularized_SGD(shufs_y, shufs_tx, w, iters_per_fold, gamma, lambda_)
        w = ws_LR_reg[-1]

        accs_tr, accs_te, losses_tr, losses_te = get_statistics_per_k(k, y_tr, tX_tr, y_te, tX_te, ws_LR_reg, div, lambda_, accs_tr, accs_te, losses_tr, losses_te)
        tot = len(ws_LR_reg) + tot

    plot_accs_losses(accs_tr, accs_te, losses_tr, losses_te, "Regularized Log. Regr. \nfor pol. degree = " + str(pol_degree) + ", batch size = " + str(batch_size), div, tot)
    print("\naccs_tr var", np.var(accs_tr)," accs_te_var", np.var(accs_te))
    return w



### FUNCTIONS FOR STATISTICS

def get_accs(y, tx, ws, accs=[]):
    """ Calculates accuracy metrics for linear models and returns it """
    nobs = len(y)
    for w in ws:
        y_pred = prjhlp.predict_labels(w, tx)
        accs.append(np.sum(y_pred == y)*100/nobs)
    return accs

def get_losses(y, tx, ws, losses=[]):
    """ Calculates MSE loss metrics for linear models and returns it """
    for w in ws:
        losses.append(compute_MSE(y,tx,w))
    return losses

def get_accs_logReg(y, tx, ws, accs=[]):
    """ Calculates accuracy metrics for logistic regression and returns it """
    nobs = len(y)
    for w in ws:
        y_pred = prjhlp.predict_labels_LogReg(w, tx)
        accs.append(np.sum(y_pred == y)*100/nobs)
    return accs

def get_losses_logReg(y, tx, ws, lambda_, losses=[]):
    """ Calculates loss metrics for logistic regression and returns it """
    for w in ws:
        losses.append(compute_loss_LogReg_approx(y, tx, w) + (lambda_ * np.sum(w ** 2)/2))
    return losses

def get_statistics_LSorRG(y_tr, tx_tr, y_te, tx_te, w, loss_tr, name):
    """ Obtains accuracy, loss metrics for training and testing dataset for LS or Rigde 
        Regression models and prints out the result """
    acc_tr = get_accs(y_tr, tx_tr, [w])
    acc_te = get_accs(y_te, tx_te, [w])
    loss_te = get_losses(y_te, tx_te, [w])
    print(name + ": loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
        l_tr=loss_tr,
        acc_tr=acc_tr[0], l_te=loss_te[0], acc_te=acc_te[0]))

def get_statistics_GD(y_tr, tx_tr, y_te, tx_te, ws, div):
    """ Obtains accuracy, loss metrics for training and testing dataset for Linear Regression
        with gradient descent models, prints out and plots the result """
    inds = np.linspace(0,len(ws)-1, int(len(ws)/div)).astype(int)
    print("Number of samples used for statistics = {l}".format(l=len(inds)))
    ws_red = np.array(ws)[inds]
    accs_tr = get_accs(y_tr, tx_tr, ws_red)
    accs_te = get_accs(y_te, tx_te, ws_red)
    losses_tr = get_losses(y_tr, tx_tr, ws_red)
    losses_te = get_losses(y_te, tx_te, ws_red)

    print("GD: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
        l_tr=losses_tr[-1],
        acc_tr=accs_tr[-1], l_te=losses_te[-1], acc_te=accs_te[-1]))

    plt.figure()
    plt.plot(inds, accs_tr, 'ro', label='acc_train')
    plt.ylabel("Accuracy of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, accs_te, 'bo', label='acc_test')
    plt.legend()

    plt.figure()
    plt.plot(inds, losses_tr, 'ro', label='loss_train')
    plt.ylabel("Losses of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, losses_te, 'bo', label='loss_test')
    plt.legend()
    plt.show()

def get_statistics_logReg(y_tr, tx_tr, y_te, tx_te, ws, div, lambda_=0, accs_tr=[], accs_te=[], losses_tr=[], losses_te=[]):
    """ Obtains accuracy, loss metrics for training and testing dataset for Logistic Regression
        with gradient descent models, prints out and plots the result """
    inds = np.linspace(0, len(ws) - 1, int(len(ws) / div)).astype(int)
    print("Number of samples used for statistics = {l}".format(l=len(inds)))
    ws_red = np.array(ws)[inds]
    accs_tr = get_accs_logReg(y_tr, tx_tr, ws_red, accs_tr)
    accs_te = get_accs_logReg(y_te, tx_te, ws_red, accs_te)
    losses_te = get_losses_logReg(y_te, tx_te, ws_red, lambda_, losses_te)
    losses_tr = get_losses_logReg(y_tr, tx_tr, ws_red, lambda_, losses_tr)

    print("LogReg GD: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
        l_tr=losses_tr[-1],
        acc_tr=accs_tr[-1], l_te=losses_te[-1], acc_te=accs_te[-1]))

    plt.figure()
    plt.plot(inds, accs_tr, 'ro', label='acc_train')
    plt.ylabel("Accuracy of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, accs_te, 'bo', label='acc_test')
    plt.legend()

    plt.figure()
    plt.plot(inds, losses_tr, 'ro', label='loss_train')
    plt.ylabel("Losses of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, losses_te, 'bo', label='loss_test')
    plt.legend()
    plt.show()


def get_statistics_per_k_LS(k, y_tr, tx_tr, y_te, tx_te, ws, div, accs_tr=[], accs_te=[], losses_tr=[], losses_te=[]):
    """ Obtains accuracy, loss metrics for training and testing dataset for one step of 
        cross-validation for LS models and prints out the result """
    inds = np.linspace(0, len(ws) - 1, int(len(ws) / div)).astype(int)
    ws_red = np.array(ws)[inds]

    accs_tr = get_accs(y_tr, tx_tr, ws_red, accs_tr)
    accs_te = get_accs(y_te, tx_te, ws_red, accs_te)
    losses_te = get_losses(y_te, tx_te, ws_red, losses_te)
    losses_tr = get_losses(y_tr, tx_tr, ws_red, losses_tr)
    print("LS GD k={k}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
        k=k+1, l_tr=losses_tr[-1], acc_tr=accs_tr[-1], l_te=losses_te[-1], acc_te=accs_te[-1]))
    return accs_tr, accs_te, losses_tr, losses_te

def get_statistics_per_k(k, y_tr, tx_tr, y_te, tx_te, ws, div, lambda_=0, accs_tr=[], accs_te=[], losses_tr=[], losses_te=[]):
    """ Obtains accuracy, loss metrics for training and testing dataset for one step of 
        cross-validation for logistic regression models and prints out the result """
    inds = np.linspace(0, len(ws) - 1, int(len(ws) / div)).astype(int)
    ws_red = np.array(ws)[inds]
    accs_tr = get_accs_logReg(y_tr, tx_tr, ws_red, accs_tr)
    accs_te = get_accs_logReg(y_te, tx_te, ws_red, accs_te)
    losses_te = get_losses_logReg(y_te, tx_te, ws_red, lambda_, losses_te)
    losses_tr = get_losses_logReg(y_tr, tx_tr, ws_red, lambda_, losses_tr)
    print("LogReg GD k={k}: loss_train = {l_tr}, acc_train = {acc_tr}, loss_test = {l_te}, acc_test = {acc_te}".format(
        k=k+1, l_tr=losses_tr[-1], acc_tr=accs_tr[-1], l_te=losses_te[-1], acc_te=accs_te[-1]))
    return accs_tr, accs_te, losses_tr, losses_te

def plot_accs_losses(accs_tr, accs_te, losses_tr, losses_te, name, div, tot):
    """ Plots accuracy vs iteration and losses vs. iteration """
    print(tot)
    #inds = np.linspace(0, tot - 1, int(tot / div)).astype(int)
    inds = np.arange(len(accs_tr))*div
    plt.figure()
    plt.plot(inds, accs_tr, 'ro', label='acc_train')
    plt.ylabel("Accuracy of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, accs_te, 'bx', label='acc_test')
    plt.title("Accuracy vs Iteration for " + name)
    plt.legend()

    plt.figure()
    plt.plot(inds, losses_tr, 'ro', label='loss_train')
    plt.ylabel("Losses of Training")
    plt.xlabel("Iteration")
    plt.plot(inds, losses_te, 'bx', label='loss_test')
    plt.legend()
    plt.title("Loss vs Iteration for " + name)
    plt.savefig("fig.png")
    plt.show()

def plots_accs_losses_LSorRG(ks, accs_tr, accs_te, losses_tr, losses_te, name, acc_tr_avg, acc_te_avg, loss_tr_avg, loss_te_avg):
    """ Plots accuracy vs k and losses vs. k, where k is cross-validation step """
    plt.figure()
    plt.plot(ks, accs_te, 'ro', label="acc_te")
    plt.plot(ks, accs_tr, 'bx', label="acc_tr")
    plt.axhline(y=acc_tr_avg, color='b', linestyle=":", label="acc_tr of avg w")
    plt.axhline(y=acc_te_avg, color='r', linestyle="dashed", label="acc_te of avg w")
    plt.ylabel("Accuracies")
    plt.xlabel("k values")
    plt.title(name + " Accuracy values vs Cross Validation Step numbers, k")
    plt.legend()
    plt.figure()
    plt.plot(ks, losses_te, 'ro', label="loss_te")
    plt.plot(ks, losses_tr, 'bx', label="loss_tr")
    plt.axhline(y=loss_tr_avg, color='b', linestyle=":", label="loss_tr of avg w")
    plt.axhline(y=loss_te_avg, color='r', linestyle="dashed", label="loss_te of avg w")
    plt.ylabel("Losses")
    plt.xlabel("k values")
    plt.title(name + " Loss values vs Cross Validation Step numbers, k")
    plt.legend()
    plt.show()
