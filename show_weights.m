function show_weights(weights)
% SHOW_WEIGHTS Displays all the receptive field structures as gray images.
%
% 	SHOW_WEIGHTS(WEIGHTS) displays all the synaptic weights (receptive fields) for the neurons.
%
%	See also SHOW_IMAGES.
%
%	Code for BIO/NROD08 Assignment 2, Winter 2019
%	Author: Blake Richards, blake.richards@utoronto.ca

% check the weights argument
if ~isnumeric(weights) || size(weights,2) ~= 784
	error('You must provide the WEIGHTS matrix, which is a nhid x 784 matrix.');
end

% create a figure
figure();

% set the colormap to black and white
colormap('gray');

% determine the best arrangement of the receptive fields
nfields = size(weights,1);
nsqrt   = sqrt(nfields);
nrows   = round(nsqrt,1);
ncols   = ceil(nfields/nrows);

% step through all weight vectors (receptive fields) and plot them
for r = 1:nrows
	for c = 1:ncols
		fieldnumber = (r-1)*ncols + c;                              
		if fieldnumber <= nfields 
			subplot(nrows,ncols,fieldnumber,'align');           
			imagesc(reshape(weights(fieldnumber,:)',28,28)');   
			axis equal; axis off;                             
		end
	end
end

% set the title for the image
set(gcf,'numbertitle','off','name','Receptive fields');

% function end
end
