function flow_img = compute_flow(im1, im2)
% COMPUTE_FLOW(im1,im2) 
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Georgia Gkioxari
% 
% This file is part of the Action Tubes code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

max_flow = 8;
scale = 128/max_flow;

im1 = double(im1);
im2 = double(im2);

flow = mex_OF(im1,im2);
mag_flow = sqrt(sum(flow.^2,3));

flow = flow*scale;  
flow = flow+128; 
flow(flow<0) = 0;
flow(flow>255) = 255;

mag_flow = mag_flow*scale;
mag_flow = mag_flow+128;
mag_flow(mag_flow<0) = 0;
mag_flow(mag_flow>255) = 255;
  
flow_img = cat(3,flow,mag_flow);
flow_img = uint8(flow_img);


