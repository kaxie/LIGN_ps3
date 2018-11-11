import numpy as np
import torch
from sympy import *

######################################## BEGIN STARTER CODE ########################################

def relu(x):
	if x<0:
		return 0
	else:
		return x

def loss(y_predicted, y_observed):
	return (y_predicted - y_observed)**2


def mlp(x,W0,W1,W2):


	r0_0 = x*W0[0]
	r0_1 = x*W0[1]
	r0_2 = x*W0[2]
	r0 = np.array([r0_0,r0_1,r0_2])

	h0_0 = relu(r0_0)
	h0_1 = relu(r0_1)
	h0_2 = relu(r0_2)
	h0 = np.array([h0_0,h0_1,h0_2])



	r1_0 = h0_0*W1[0,0] + h0_1*W1[0,1]+ h0_2*W1[0,2]
	r1_1 = h0_0*W1[1,0] + h0_1*W1[1,1]+ h0_2*W1[1,2]
	r1_2 = h0_0*W1[2,0] + h0_1*W1[2,1]+ h0_2*W1[2,2]
	r1 = np.array([r1_0,r1_1,r1_2])

	h1_0 = relu(r1_0)
	h1_1 = relu(r1_1)
	h1_2 = relu(r1_2)
	h1 = np.array([h1_0,h1_1,h1_2])

	y_predicted = h1_0*W2[0] + h1_1*W2[1]+ h1_2*W2[2]

	variable_dict = {}
	variable_dict['x'] = x
	variable_dict['r0'] = r0
	variable_dict['h0'] = h0
	variable_dict['r1'] = r1
	variable_dict['h1'] = h1
	variable_dict['y_predicted'] = y_predicted

	return variable_dict


x = 10
W0 = np.array([1,2,3])
W1 = np.array([[3,4,5],[-5,4,3],[3,4,1]])
W2 = np.array([1,3,-3])

#print(mlp(x,W0,W1,W2))

###########  PyTorch code   ###########
def torch_mlp(x,W0,W1,W2):
	m = torch.nn.ReLU()
	h0 = m(torch.mul(W0,x))

	h1 = m(torch.matmul(W1,h0))

	y_predicted = torch.dot(W2,h1)

	return y_predicted

def torch_loss(y_predicted,y_observed):
	return torch.pow(y_predicted-y_observed,2)


x_torch = torch.tensor(x,dtype=torch.float)
W0_torch = torch.tensor(W0,dtype=torch.float,requires_grad=True)
W1_torch = torch.tensor(W1,dtype=torch.float,requires_grad=True)
W2_torch = torch.tensor(W2,dtype=torch.float,requires_grad=True)
output = torch_mlp(x_torch,W0_torch,W1_torch,W2_torch)

########### END PyTorch code  ###########




######################################## END STARTER CODE ########################################


# NOTICE: DO NOT EDIT FUNCTION SIGNATURES
# PLEASE FILL IN FREE RESPONSE AND CODE IN THE PROVIDED SPACES



#PROBLEM 1
def d_loss_d_ypredicted(variable_dict,y_observed):
##YOUR CODE HERE##
    predicted_val = variable_dict[y_predicted]
    return 2*predicted_val -2 * y_observed


#PROBLEM 2
def d_loss_d_W2(variable_dict,y_observed):
##YOUR CODE HERE##
    partial_d_loss = d_loss_d_ypredicted(variable_dict, y_observed)
    hid_lay_1 = np.array(variable_dict[h1])
#    predicted_val = np.dot(W2_torch, hid_lay_1)

#    partial_w0 = predicted_val.diff(W2_torch[0])
#    partial_w1 = predicted_val.diff(W2_torch[1])
#    partial_w2 = predicted_val.diff(W2_torch[2])
#    np.array([hid_lay_1[0]])
    return hid_lay_1*partial_d_loss
#PROBLEM 3
def d_loss_d_h1(variable_dict,W2,y_observed):
##YOUR CODE HERE##
    partial_d_loss = d_loss_d_ypredicted(variable_dict, y_observed)
    return W2*partial_d_loss

#PROBLEM 4
def relu_derivative(x):
##YOUR CODE HERE##
    if x>0:
        return 1
    else:
        return 0

#PROBLEM 5
def d_loss_d_r1(variable_dict,W2,y_observed):
##YOUR CODE HERE##
    d_loss_d_h1_result = d_loss_d_h1(variable_dict, W2, y_observed)
    r1_val = np.array(variable_dict[r1]);
    for x in r1_val:
        x=relu_derivative(x)
    h1_val = r1_val
    return d_loss_d_h1_result * h1_val
#PROBLEM 6
def d_loss_d_W1(variable_dict,W2,y_observed):
##YOUR CODE HERE##

    d_loss_r1 = d_loss_d_r1(variable_dict,W2,y_observed)
    h0 = variable_dict['h0']
    return np.outer(d_loss_r1, h0)

#PROBLEM 7
def d_loss_d_h0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##
    loss_r1 = d_loss_r1(variable_dict, W2, y_observed)
    h0 = variable_dict(h0)
    result = np.zeros(h0.size())
    i=j=0;
    for x in result:
        sum = 0;
        for y in loss_r1:
            sum += y*W1[i][j];
            j+=1;
        x = sum;
        j=0;
        i+=1;

#PROBLEM 8
def d_loss_d_r0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##
    r0 = variable_dict[r0]
    h0_loss = d_loss_d_h0(variable_dict, W1, W2, y_observed)

    for x,i in zip(h0_loss,range(3)):
        x= x * relu_derivative(r0[i])
    return h0_loss

#PROBLEM 9
def d_loss_d_W0(variable_dict,W1,W2,y_observed):
##YOUR CODE HERE##
    d_loss_r0 = d_loss_d_r0(variable_dict, W1, W2, y_observed)
    for item in d_loss_r0:
        item = item * x
    d_loss_W0 = d_loss_r0
    return d_loss_W0
