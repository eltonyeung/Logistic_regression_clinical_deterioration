function [r0,r1] = forward_pass(features,W0,W1,b0,b1)
% FORWARD_PASS Run a forward pass through the network.
%
% 	[R0,R1] = FORWARD_PASS(IMAGES,W0,W1,B0,B1) calculates the neural responses in the hidden layer and output layer
%	and returns them as R0 and R1.
%
%	See also BACKWARD_PASS.
%

% check the images arguments
if ~isnumeric(features) || size(features,1) ~= size(features,1)
	error('You must provide a FEATURE matrix with 30 rows. Use load_data.m');
end

% check the weights and bias arguments
if ~isnumeric(W0) || size(W0,2) ~= size(features,1)
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

% calculate the hidden layer activity
r0 = sigmoid(W0*features+b0);
% TO-DO: COMPLETE THE CODE TO CALCULATE THE HIDDEN LAYER ACTIVITY 

% calculate the output layer activity
r1 = sigmoid(W1*r0+b1);
% TO-DO: COMPLETE THE CODE TO CALCULATE THE OUTPUT LAYER ACTIVITY 

% function end
end
