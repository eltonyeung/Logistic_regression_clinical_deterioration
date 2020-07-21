function E = calculate_error(r1,labels)
% CALCULATE_ERROR Calculates the classification error on a set of images.
%
% 	E = CALCULATE_ERROR(OUTPUT,LABELS) Calculates the classsification error rate for a given set
%	of OUTPUT given LABELS. It uses the max active unit to determine classification.
%
%	See also CALCULATE_LOSS.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% check the arguments
if ~all(size(r1) == size(labels))
	error('The OUTPUT and LABELS matrices must be the same size.');
end

% determine classification by maximally active unit
classification = 1.0*(r1 == repmat(max(r1,[],1),2,1));

% determine error rate
E = 1.0 - sum(max(classification.*labels,[],1))/size(labels,2);

% function end
end
