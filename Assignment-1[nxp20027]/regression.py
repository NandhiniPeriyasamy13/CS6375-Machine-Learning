# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np   # For all our math needs
import matplotlib.pyplot as plt          # For all our plotting needs
# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split

# All functions defined here

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    phi = []
    for value in X :
        temp = []
        for dim in range(0,d+1):
            temp.append(np.power(value,dim))
        phi.append(temp)
    return np.asarray(phi)
    
# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(Phi),Phi)),np.transpose(Phi)),y)
  
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    ypred = Phi @ w
    err = (ypred-y) ** 2
    sum = 0
    for val in err :
        sum = sum + val
    return (sum/err.shape[0])
    

# Main code goes here
def generate_data():
    n = 750                                  # Number of data points
    X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
    e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
    y = f_true(X) + e   
    
    
    
    plt.figure()
    
    # Plot the data
    plt.scatter(X, y, 12, marker='o')           
    
    # Plot the true function, which is really "unknown"
    x_true = np.arange(-7.5, 7.5, 0.05)
    y_true = f_true(x_true)
    plt.plot(x_true, y_true, marker='None', color='r')
    
    
    tst_frac = 0.3  # Fraction of examples to sample for the test set
    val_frac = 0.1  # Fraction of examples to sample for the validation set
    
    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)
    
    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
    
    # Plot the three subsets
    plt.figure()
    plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
    plt.scatter(X_val, y_val, 12, marker='o', color='green')
    plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
    return X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true

def evaluate_and_plot_polynomial_basis_function(X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true):
    # Plotting the results
    w = {}               # Dictionary to store all the trained models
    validationErr = {}   # Validation error of the models
    testErr = {}         # Test error of all the models
    
    
    for d in range(3, 25, 3):  # Iterate over polynomial degree
        Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
        w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
        
        Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
        validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
        
        Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
        testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data
    
    # Plot all the models
    plt.figure()
    plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
    plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Polynomial degree', fontsize=16)
    plt.ylabel('Validation/Test error', fontsize=16)
    plt.xticks(list(validationErr.keys()), fontsize=12)
    plt.legend(['Validation Error', 'Test Error'], fontsize=16)
    plt.axis([2, 25, 15, 60])
    
    plt.figure()
    plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
    
    for d in range(9, 25, 3):
      X_d = polynomial_transform(x_true, d)
      y_d = X_d @ w[d]
      plt.plot(x_true, y_d, marker='None', linewidth=2)
    
    plt.legend(['true'] + list(range(9, 25, 3)))
    plt.axis([-8, 8, -15, 15])


# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    phi = []
    for row in X :
        temp = []
        for col in B :
            z = -gamma*(np.power(row-col,2))
            temp.append(np.exp(z))
        phi.append(temp)
    phi = np.asarray(phi)
    #print(phi)
    return phi

# Phi float(n, d): transformed data
# y float(n, ): labels
# lam float : regularization parameter
def train_ridge_model(Phi, y, lam):
    iMatrix = np.eye(Phi.shape[0])
    eq = np.linalg.inv((np.transpose(Phi) @ Phi) + (lam * iMatrix))
    w = eq @ np.transpose(Phi) @ y
    #print(w)
    return w

def evaluate_and_plot_radial_basis_function(X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true):
    w = {}
    validationErr = {}
    testErr = {}
    lam = 0.001
    while(lam < 1001):
        Phi_trn = radial_basis_transform(X_trn, X_trn)                
        w[lam] = train_ridge_model(Phi_trn, y_trn, lam)                       
        
        Phi_val = radial_basis_transform(X_val, X_trn)                 
        validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam])  
        
        Phi_tst = radial_basis_transform(X_tst, X_trn)           
        testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam])  
        lam = lam*10
    
    
    # Plot all the models
    plt.figure()
    plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=4, markersize=12)
    plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
    plt.xlabel('Lamda', fontsize=16)
    plt.ylabel('Validation/Test error', fontsize=16)
    plt.xticks(list([0.001,0.01,0.1,1,10,100,1000]), fontsize=12)
    plt.legend(['Validation Error', 'Test Error'], fontsize=16)
    plt.xlim(0.001,1000)
    plt.xscale('log')
   # plt.yscale('log')
    plt.figure()
    plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')
    
    lam = 0.001
    while(lam <1001):
      X_lam = radial_basis_transform(x_true, X_trn,)
      y_lam = X_lam @ w[lam]
      plt.plot(x_true, y_lam, marker='None', linewidth=2)
      lam = lam*10
    
    plt.legend(['true'] + list([0.001,0.01,0.1,1,10,100,1000]))
    plt.axis([-8, 8, -30, 30])

def main():
    X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true = generate_data()
    evaluate_and_plot_polynomial_basis_function(X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true)
    evaluate_and_plot_radial_basis_function(X_trn,y_trn,X_val,y_val,X_tst,y_tst,x_true,y_true)

main()    
    



