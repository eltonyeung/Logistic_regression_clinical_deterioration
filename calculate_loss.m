function L = calculate_loss(r1,labels)
% CALCULATE_LOSS Calculates the mean sqaured error loss on some output activity.
%
% 	L = CALCULATE_LOSS(OUTPUT,LABELS) Calculates the mean squared error loss function for a given set
%	of OUTPUT given LABELS.
%
%	See also CALCULATE_ERROR.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% check the arguments
if ~all(size(r1) == size(labels))
	error('The OUTPUT and LABELS matrices must be the same size.');
end

% calculate the loss
L = (0.5)*sum((r1-labels).^2,'all');    %L2 - dense


% L = (0.5)*sum(abs(r1-labels),'all');    %L1 - sparse

% function end
end
