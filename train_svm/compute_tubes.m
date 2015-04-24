function tubes = compute_tubes(split, annot, rcnn_model, varargin)
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
ip.addRequired('split',         @isscalar);
ip.addRequired('annot',         @isstruct);
ip.addRequired('rcnn_model',    @isstruct);
ip.addParamValue('feat_dir',    @isstr);
ip.parse(split, annot, rcnn_model, varargin{:});
opts = ip.Results;

fprintf('Tubes\n');
feat_dir_spat = [opts.feat_dir '/spatial/%s/%s/%05d.mat'];
feat_dir_flow = [opts.feat_dir '/motion/%s/%s/%05d.mat'];

actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
    'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
    'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};
num_actions = length(actions);
nms_thresh = 0.3;

% get data for split
videos = get_data(actions,split);
tubes = videos;

for i=1:length(videos)
    
  fprintf('[%d/%d]\n',i,length(videos));
  
  video_id  = videos(i).video_id;
  action    = videos(i).action;
  
  num_frames = max(annot.frame(strcmp(video_id,annot.video_id)));
  
  frames = struct([]);
  for f=1:num_frames
      
    mat_file_spat = sprintf(feat_dir_spat,action,video_id,f);
    mat_file_flow = sprintf(feat_dir_flow,action,video_id,f);
    
    d_spat = load(mat_file_spat);
    d_flow = load(mat_file_flow);
    assert(isequal(d_spat.boxes,d_flow.boxes));
    d.feat  = [d_spat.feat d_flow.feat];
    d.boxes = d_spat.boxes;
    d.gt    = d_spat.gt;
    scores  = fc7_to_svm(d.feat,rcnn_model); 
    scores  = scores(~d.gt,:);
    boxes   = d.boxes(~d.gt,:);
    feat    = d.feat(~d.gt,:);
    frames(f).boxes = boxes;
    frames(f).scores = scores;
    frames(f).feat = feat;
  end
  clear boxes scores;  
      
  for a=1:num_actions
      
    % nms for action
    action_frames = struct([]);
    for f=1:length(frames)
      pick = nms([frames(f).boxes frames(f).scores(:,a)],nms_thresh);
      action_frames(f).boxes = frames(f).boxes(pick,:);
      action_frames(f).scores = frames(f).scores(pick,a);
      action_frames(f).feat = frames(f).feat(pick,:);
    end
   
    paths = zero_jump_link(action_frames);
    
    tubes(i).paths{a} = paths;
    
  end
  tubes(i).actions = actions;
end


% -------------------------------------------------------------------------
function feat = fc7_to_svm(feat,rcnn_model)
% -------------------------------------------------------------------------
% rescale features
target_norm = 20;
f1 = feat(:,1:4096);
f2 = feat(:,4096+(1:4096));
f1 = f1 .* (target_norm / rcnn_model.detectors.training_opts.feat_norm_mean_spat);
f2 = f2 .* (target_norm / rcnn_model.detectors.training_opts.feat_norm_mean_flow);
feat = [f1 f2];
% compute scores
feat = bsxfun(@plus, feat*rcnn_model.detectors.W, rcnn_model.detectors.B);


% -------------------------------------------------------------------------
function videos = get_data(actions,split_id)
% -------------------------------------------------------------------------

num_actions = length(actions);
videos = [];
num = 0;

% get data
for i=1:num_actions
  split_file = sprintf('%s_test_split%d.txt',actions{i},split_id);
  fid = fopen(split_file,'r');
  data = textscan(fid,'%s %d');
  for j=1:length(data{1})
    video = data{1}{j};
    avi = strfind(video,'.avi');
    video = video(1:avi(1)-1);
    if data{2}(j)==1
      continue;
    elseif data{2}(j)==2
      num = num+1;
      videos(num).video_id = video;
      videos(num).action = actions{i};
      videos(num).action_id = i;
    end
  end
end
