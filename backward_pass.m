function [dL_by_dW0,dL_by_dW1,dL_by_db0,dL_by_db1] = backward_pass(features,labels,r0,r1,W0,W1,b0,b1)
% BACKWARD_PASS Run a backward pass through the network.
%
% 	[dL_by_dW0,dL_by_dW1,dL_by_db0,dL_by_db1] = BACKWARD_PASS(IMAGES,LABELS,R0,R1,W0,W1,B0,B1) calculates the gradient
%	of the loss function with respect to the network weights and biases using the activity in the hidden and
%	output layers, R0 and R1, the input IMAGES, and the target activity for the output layer, LABELS. Returns the gradients as 
%	dL_by_dW0, dL_by_dW1, dL_by_db0, and dL_by_db1.
%
%	See also FORWARD_PASS.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% check the images arguments
if ~isnumeric(features) || ~all(size(features) == [size(features,1) size(r0,2)])
	error('You must provide an IMAGES matrix with 784 rows and nimages columns. Use load_data.m');
end

% check the labels arguments
if ~isnumeric(labels) || ~all(size(labels) == size(r1))
	error('You must provide a LABELS matrix, which is a 10 x nimages matrix. Use load_data.m');
end

% check the activity arguments
if ~isnumeric(r0) || ~all(size(r0) == [size(W0,1) size(r1,2)])
	error('You must provide the r0 matrix, which is a nhid x nimages matrix.');
end
if ~isnumeric(r1) || ~all(size(r1) == [2 size(r0,2)])
	error('You must provide the r1 matrix, which is a 10 x nimages matrix.');
end

% check the weights and bias arguments
if ~isnumeric(W0) || ~all(size(W0) == [size(r0,1) size(features,1)])
	error('You must provide the W0 matrix, which is a nhid x 784 matrix.');
end
if ~isnumeric(W1) || ~all(size(W1) == [2 size(W0,1)])
	error('You must provide the W1 matrix, which is a 10 x nhid matrix.');
end
if ~isnumeric(b0) || ~all(size(b0) == [size(W0,1) 1])
	error('You must provide the b0 vector, which is a nhid element column vector.');
end
if ~isnumeric(b1) || ~all(size(b1) == [2 1])
	error('You must provide the b1 vector, which is a 10 element column vector.');
end

% calculate the gradient of the loss with respect to the output layer activity
dL_by_dr1 = r1-labels; 
% TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the output layer activity with respect to the output layer input
dr1_by_dx1 = r1.*(1-r1);
% TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the loss with respect to the output layer input
dL_by_dx1 = (dL_by_dr1.*dr1_by_dx1);    % TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the output layer input with respect to the output weights
dx1_by_dW1 = r0;    % TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the loss with respect to the output weights and biases
dL_by_dW1 =  (dL_by_dx1*dx1_by_dW1');   % TO-DO: CALCULATE THIS GRADIENT - NOTE: THIS GRADIENT REQUIRES A SUM ACROSS IMAGES 
dL_by_db1 = sum(dL_by_dx1,2);     % TO-DO: CALCULATE THIS GRADIENT - NOTE: THIS GRADIENT REQUIRES A SUM ACROSS IMAGES 

% calculate the gradient of the output layer input with respect to the hidden layer activity
dx1_by_dr0 = W1;     % TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the loss with respect to the hidden layer activity
dL_by_dr0 = (dL_by_dx1'*dx1_by_dr0);      % TO-DO: CALCULATE THIS GRADIENT - NOTE: THIS GRADIENT REQUIRES A SUM ACROSS OUTPUT UNITS

% calculate the gradient of the hidden layer activity with respect to the hidden layer input
dr0_by_dx0 = r0.*(1-r0);     % TO-DO: CALCULATE THIS GRADIENT
 
% calculate the gradient of the loss with respect to the hidden layer input
dL_by_dx0 = (dr0_by_dx0.*dL_by_dr0');       % TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the hidden layer input with respect to the hidden layer weights
dx0_by_dW0 = features; % TO-DO: CALCULATE THIS GRADIENT

% calculate the gradient of the loss with respect to the hidden layer weights and biases
dL_by_dW0 = (dL_by_dx0*dx0_by_dW0');    % TO-DO: CALCULATE THIS GRADIENT - NOTE: THIS GRADIENT REQUIRES A SUM ACROSS IMAGES
dL_by_db0 = sum(dL_by_dx0,2);    % TO-DO: CALCULATE THIS GRADIENT - NOTE: THIS GRADIENT REQUIRES A SUM ACROSS IMAGES

% function end
end
