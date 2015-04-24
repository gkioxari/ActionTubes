function MotionSalBoxes = get_motion_salient_boxes(varargin)
% PARAMETERS
% annot        : JHMDB annotations (jhmdb_annot.mat)
% split        : split for JHMDB
% target       : 1 or 2 or [] for train or test or all
% ss_dir       : directory of selective search boxes
% flow_dir     : directory containing optical flow 
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

ip = inputParser;
ip.addParamValue('annot', @isstruct);
ip.addParamValue('split', @isscalar);
ip.addParamValue('target',@isscalar);
ip.addParamValue('ss_dir', @isstr);
ip.addParamValue('flow_dir', @isstr);

ip.parse(varargin{:});
opts = ip.Results;

actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
    'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
    'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};

opts.ss_dir = [opts.ss_dir '/%s/%s/%05d.mat'];
opts.flow_dir = [opts.flow_dir '/%s/%s/%05d.png'];

n_filter = 5;
motion_thresh = 0.2;
motion_inclusion = 0.3;

videos = get_data(actions,opts.split,opts.target);

MotionSalBoxes = opts.annot;

fprintf('Saliency boxes\n');
for i=1:length(videos)
    
  fprintf('[%d /%d]',i,length(videos));
  video_id = videos(i).video_id;
  action   = videos(i).action;
  
  gt_ind = find(strcmp(video_id,opts.annot.video_id));
  for j=1:length(gt_ind)
    id = gt_ind(j);
    
    f = opts.annot.frame(id);
    
    % load ss boxes
    ss_file = sprintf(opts.ss_dir,action,video_id,f);
    d = load(ss_file);
    boxes = d.boxes; clear d;
    boxes = unique(boxes,'rows');
    % note: boxes are in [y1 x1 y2 x2] format
    
    % load flow
    flow_file = sprintf(opts.flow_dir,action,video_id,f);
    if ~exist(flow_file,'file')
      error('Flow file does not exist');
    end
    flow_img = imread(flow_file);
    flow_mag = double(flow_img(:,:,3))-128;
    flow_mag = flow_mag/128;
    
    % compute saliency
    motion_map = motion_saliency(flow_mag,n_filter);
    mot_thresh = min([motion_thresh, 0.7*max(motion_map(:))]);
    motion_map = motion_map>=mot_thresh;
    
    keep_boxes = false(size(boxes,1),1);
    for b=1:size(boxes,1)
      y1 = boxes(b,1); x1 = boxes(b,2);
      y2 = boxes(b,3); x2 = boxes(b,4);
      
      mot = motion_map(y1:y2,x1:x2);
      inclusion = sum(mot(:))/sum(motion_map(:));
      if inclusion>=motion_inclusion
        keep_boxes(b)=true;
      end
      
    end

    boxes = boxes(keep_boxes,:);
    
    boxes = boxes(:,[2 1 4 3]);
    MotionSalBoxes.boxes{id} = boxes;
    
  end
end

% -------------------------------------------------------------------------
function motion_map = motion_saliency(flow_mag,n)
% -------------------------------------------------------------------------

prior = flow_mag / max(flow_mag(:));
filt = ones(n,n)/n/n;
likeli = imfilter(flow_mag,filt,'same');
motion_map = likeli.*prior;


% -------------------------------------------------------------------------
function videos = get_data(actions,split_id,target)
% -------------------------------------------------------------------------

num_actions = length(actions);
videos = [];
num = 0;

% get data
for i=1:num_actions
  split_file = sprintf('%s_test_split%d.txt',splits_dir,actions{i},split_id);
  fid = fopen(split_file,'r');
  data = textscan(fid,'%s %d');
  for j=1:length(data{1})
    video = data{1}{j};
    avi = strfind(video,'.avi');
    video = video(1:avi(1)-1);
    if ~isempty(target)
      if data{2}(j)~=target, continue; end
    end
    num = num+1;
    videos(num).video_id = video;
    videos(num).action = actions{i};
    videos(num).action_id = i;
  end
end
