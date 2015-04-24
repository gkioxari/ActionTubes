function rcnn_cache_fc7_features_jhmdb(varargin)
% PARAMETERS
% type         : 'spatial' or 'motion'
% split        : split for JHMDB
% target       : 1 or 2 for train or test 
% net_def_file : prototxt for feature extraction
% net_file     : caffemodel
% output_dir   : directory to store features
% img_dir      : directory with images (either RGB or flow)
% boxes_file   : file containing the data (SS boxes + gt) as produced by
%                get_motion_salient_boxes() (or jhmdb_motion_sal_annot.mat)
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
% ---------------------------------------------------------
% Copyright (c) 2015, Georgia Gkioxari
% 
% This file is part of the Action Tubes code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

ip = inputParser;
ip.addParamValue('type', @isstr);
ip.addParamValue('split', @isscalar);
ip.addParamValue('target',@isscalar);
ip.addParamValue('net_def_file', @isstr);
ip.addParamValue('net_file', @isstr);
ip.addParamValue('output_dir', @isstr);
ip.addParamValue('img_dir', @isstr);
ip.addParamValue('boxes_file', @isstr);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);

ip.parse(varargin{:});
opts = ip.Results;

if isequal(type,'spatial')
  opts.IMG_MEAN = [102.9801,115.9465,122.7717];
elseif isequal(type,'motion')
  opts.IMG_MEAN = [128, 128, 128];
else
  error('Type must be spatial or motion');
end
fprintf('IMG_MEAN = [%f %f %f]\n',opts.IMG_MEAN);

% of the format output_dir/type/action/video_id/ [You can change this according 
% to your preferences]
opts.output_dir = [opts.output_dir type '/%s/%s/'];

% images (frames or flow) must be stored in the format 
% img_dir/type/action/video/
opts.img_dir = [opts.img_dir '/%s/%s/%05d.png'];

% load videos for train/test of split
[videos] = get_data(opts.split,opts.target);

% -- Load annot --
q=load(opts.boxes_file);
annot = q.sal_boxes; 
clear q;

if opts.end == 0
  opts.end = length(videos);
end

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature extraction options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;
for c=1:3
  rcnn_model.cnn.image_mean(:,:,c) = opts.IMG_MEAN(c);
end

for i = opts.start:opts.end
    
  fprintf('Video [%d/%d]\n',i,opts.end);
  
  video_id = videos(i).video_id;
  action = videos(i).action;
  
  save_dir = sprintf(opts.output_dir,action,video_id);
  if ~exist(save_dir,'dir')
    mkdir(save_dir);
  end
  
  gt_ind = find(strcmp(video_id,annot.video_id));
  for j=1:length(gt_ind)
    fprintf('(%d) ',j);
    id = gt_ind(j);
    f = annot.frame(id);
    
    assert(isequal(annot.video_id{id},video_id));
    assert(isequal(annot.action{id},action));
    
    % if image does not exist continue
    if ~exist(sprintf(img_dir,action,video_id,f),'file')
        fprintf('%s - %s - %d does not exist..skipping\n',action, video_id, f);
        continue; 
    end

    save_file = sprintf('%s/%05d.mat',save_dir,f);
    if exist(save_file,'file')
      continue;
    end
     
    im = imread(sprintf(opts.img_dir,action,video_id,f));
    
    boxes = annot.boxes{id};
    if isempty(boxes)
      error('Empty boxes!');
    end
    
    gt_bound = annot.bound(id,:);
    gt_box = [gt_bound(:,1:2) gt_bound(:,1:2)+gt_bound(:,3:4)-1];
    
    d.gt = [true(size(gt_box,1),1);false(size(boxes,1),1)];
    d.boxes = [gt_box;boxes];
  
    d.feat = rcnn_features(im, d.boxes, rcnn_model);
    
    save(save_file, '-struct', 'd');
  end
  fprintf('\n');
end

function [videos] = get_data(split_id,target_id)

actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
    'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
    'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};

num_actions = length(actions);
videos = [];
num = 0;

% get data
for i=1:num_actions
  % this file must be on paths
  split_file = sprintf('%s_test_split%d.txt',actions{i},split_id);
  
  fid = fopen(split_file,'r');
  data = textscan(fid,'%s %d');
  fclose(fid); 
  
  for j=1:length(data{1})
      
    video = data{1}{j};
    avi = strfind(video,'.avi');
    video = video(1:avi(1)-1);
    
    if data{2}(j)==target_id
      num = num+1;
      videos(num).video_id = video;
      videos(num).action = actions{i};
      videos(num).action_id = i;      
    end
  end
end

