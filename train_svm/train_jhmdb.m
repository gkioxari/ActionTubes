function rcnn_model = train_jhmdb(split, annot, varargin)

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
ip.addRequired('split',    @isscalar);
ip.addRequired('annot',    @isstruct);
ip.addParamValue('feat_dir',                @isstr);
ip.addParamValue('save_dir',                @isstr);
ip.addParamValue('cache_name',      'none', @isstr);
ip.addParamValue('svm_C',           10^-3,  @isscalar);
ip.addParamValue('bias_mult',       10,     @isscalar);
ip.addParamValue('pos_loss_weight', 2,      @isscalar);
ip.addParamValue('layer',           7,      @isscalar);

ip.parse(split, annot, varargin{:});
opts = ip.Results;

opts.feat_dir_spat = [opts.feat_dir '/spatial/%s/%s/%05d.mat'];
opts.feat_dir_flow = [opts.feat_dir '/motion/%s/%s/%05d.mat'];

opts.actions = {'brush_hair','catch','clap','climb_stairs','golf','jump',...
    'kick_ball','pick','pour','pullup','push','run','shoot_ball','shoot_bow',...
    'shoot_gun','sit','stand','swing_baseball','throw','walk','wave'};

if ~exist(opts.save_dir,'dir')
  mkdir(opts.save_dir);
end

% get data
num_clss = length(opts.actions);
videos = get_data(split,1,opts.actions);
dataset.videos = videos;
dataset.split = opts.split;
fprintf('# of training videos: %d\n',length(videos));

% Record a log of the training and test procedure
diary_file = [opts.save_dir 'log.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');


% ------------------------------------------------------------------------
% Get or compute the average norm of the features
save_file = [opts.save_dir 'feat_stats.mat'];
try
  ld = load(save_file);
  opts.feat_norm_mean_spat = ld.feat_norm_mean_spat;
  opts.feat_norm_mean_flow = ld.feat_norm_mean_flow;
  clear ld;
catch
  [feat_norm_mean_spat, stddev_spat] = feat_stats(dataset.videos,opts.feat_dir_spat);
  [feat_norm_mean_flow, stddev_flow] = feat_stats(dataset.videos,opts.feat_dir_flow);
  save(save_file, 'feat_norm_mean_spat', 'stddev_spat', 'feat_norm_mean_flow', 'stddev_flow');
  opts.feat_norm_mean_spat = feat_norm_mean_spat;
  opts.feat_norm_mean_flow = feat_norm_mean_flow;
end
fprintf('average norm = %.3f %.3f\n', opts.feat_norm_mean_spat, opts.feat_norm_mean_flow);
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Init models
models = {};
for i = 1:num_clss
  models{i} = init_model(opts.actions{i}, dataset, opts);
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get all positive examples
save_file = [opts.save_dir 'pos_cache.mat'];
try
  load(save_file);
  fprintf('Loaded saved positives from ground truth boxes\n');
catch
  X_pos = get_positive_features(models, dataset, opts);
  save(save_file, 'X_pos', '-v7.3');
end

for i = 1:num_clss
  fprintf('%14s has %6d positive instances\n', models{i}.class, size(X_pos{i},1));
  X_pos{i} = rcnn_scale_features_combine(X_pos{i}, opts.feat_norm_mean_spat, opts.feat_norm_mean_flow);
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Init training caches
caches = {};
for i = 1:num_clss
  caches{i} = init_cache(models{i}, X_pos{i});
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Train with hard negative mining
first_time = true;
max_hard_epochs = 1;

rand_ind = randperm(length(dataset.videos));

% Train SVMs with hard negative mining
for hard_epoch = 1:max_hard_epochs
  for ii = 1:length(dataset.videos)
    i = rand_ind(ii);
    num_frames = double(max(annot.frame(strcmp(dataset.videos(i).video_id,annot.video_id))));
    for f=1:num_frames
        
      fprintf('[%d/%d] %d\n',i,length(videos),f);     
      
      % Get hard negatives for all classes at once (avoids loading feature cache
      % more than once)
      [X, keys] = get_negative_features(first_time, models, caches, dataset, i, f, opts);

      % Add sampled negatives to each classes training cache, removing
      % duplicates
      for j = 1:num_clss
        
        if ~isempty(keys{j})
          [~, ~, dups] = intersect(caches{j}.keys, keys{j}, 'rows');
          assert(isempty(dups));
          caches{j}.X_neg = cat(1, caches{j}.X_neg, X{j});
          caches{j}.keys = cat(1, caches{j}.keys, keys{j});
          caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
        end

        % Update model if
        %  - first time seeing negatives
        %  - more than retrain_limit negatives have been added
        %  - its the final image of the final epoch
        is_last_time = (hard_epoch == max_hard_epochs && ii == length(dataset.videos) && f==num_frames);
        hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
        if first_time || hit_retrain_limit || is_last_time
          fprintf('>>> Retraining %s <<<\n', models{j}.class);
          fprintf('Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
          models{j} = update_model(models{j}, caches{j}, opts);
          caches{j}.num_added = 0;

          z_pos = caches{j}.X_pos*models{j}.w + models{j}.b;
          z_neg = caches{j}.X_neg*models{j}.w + models{j}.b;

          caches{j}.pos_loss(end+1) = opts.svm_C*sum(max(0, 1 - z_pos))*opts.pos_loss_weight;
          caches{j}.neg_loss(end+1) = opts.svm_C*sum(max(0, 1 + z_neg));
          caches{j}.reg_loss(end+1) = 0.5*models{j}.w'*models{j}.w + ...
                                    0.5*(models{j}.b/opts.bias_mult)^2;
          caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                    caches{j}.neg_loss(end) + ...
                                    caches{j}.reg_loss(end);

          for t = 1:length(caches{j}.tot_loss)
            fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                  t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                  caches{j}.neg_loss(t), caches{j}.reg_loss(t));
          end

          % evict easy examples
          easy = find(z_neg < caches{j}.evict_thresh);
          caches{j}.X_neg(easy,:) = [];
          caches{j}.keys(easy,:) = [];
          fprintf('  Pruning easy negatives\n');
          fprintf('  Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
          fprintf('  %d pos support vectors\n', numel(find(z_pos <=  1)));
          fprintf('  %d neg support vectors\n', numel(find(z_neg >= -1)));

          model = models{j};
          save([opts.save_dir models{j}.class '_' num2str(length(caches{j}.tot_loss))], 'model');
          clear model;
        end
      end
      first_time = false;
    end
  end
end

for i = 1:num_clss
  model = models{i};
  % save the negative support vector keys (for viewing support vectors)
  z_neg = caches{i}.X_neg*model.w + model.b;
  model.neg_sv_keys = caches{i}.keys(find(z_neg > -1.001), :);
  save([opts.save_dir models{i}.class '_final'], 'model');
  clear model;
end
save([opts.save_dir 'models_final'], 'models');


W = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
W = cat(2, W{:});
B = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
B = cat(2, B{:});
rcnn_model.detectors.W = W;
rcnn_model.detectors.B = B;
rcnn_model.detectors.training_opts = opts;
save([opts.save_dir 'rcnn_model'], 'rcnn_model');


% ------------------------------------------------------------------------
function [X_neg, keys] = get_negative_features(first_time, models, ...
                                                  caches, dataset, ind, ...
                                                  frame, opts)
% ------------------------------------------------------------------------
%%%%
video_id = dataset.videos(ind).video_id;
action   = dataset.videos(ind).action;
mat_file_spat = sprintf(opts.feat_dir_spat,action,video_id,frame);
mat_file_flow = sprintf(opts.feat_dir_flow,action,video_id,frame);

d_spat = load(mat_file_spat);
d_flow = load(mat_file_flow);

assert(isequal(d_spat.boxes,d_flow.boxes));
assert(isequal(d_spat.gt,d_flow.gt));

feat = [d_spat.feat d_flow.feat];

bounds = [d_spat.boxes(:,1:2) d_spat.boxes(:,3:4)-d_spat.boxes(:,1:2)+1];

%%%%
feat = rcnn_scale_features_combine(feat, opts.feat_norm_mean_spat, opts.feat_norm_mean_flow);

%%%% compute overlap with gt
gt_ind = find(strcmp(video_id,opts.annot.video_id) & opts.annot.frame==frame);
gt_bounds = opts.annot.bound(gt_ind,:);
if ~isempty(gt_ind)
  overlap = inters_union(bounds,gt_bounds);
  [overlap, assignment] = max(overlap,[],2);
else
  overlap = zeros(size(bounds,1),1);
end

d.overlap = zeros(size(bounds,1),length(opts.actions));
action_id = find(strcmp(action,opts.actions));
d.overlap(:,action_id) = overlap;

%%%%

neg_ovr_thresh = 0.3;

if first_time
  
  for i = 1:length(models)
    assert(isequal(models{i}.class,opts.actions{i}));     
    I = find(d.overlap(:, i) < neg_ovr_thresh);
    X_neg{i} = feat(I,:);
    keys{i} = [ind*ones(length(I),1) frame*ones(length(I),1) I];
  end
else
  ws = cat(2, cellfun(@(x) x.w, models, 'UniformOutput', false));
  ws = cat(2, ws{:});
  bs = cat(2, cellfun(@(x) x.b, models, 'UniformOutput', false));
  bs = cat(2, bs{:});
  zs = bsxfun(@plus, feat*ws, bs);
  for i = 1:length(models)
    z = zs(:,i);
    I = find((z > caches{i}.hard_thresh) & ...
             (d.overlap(:, i) < neg_ovr_thresh));

    % Avoid adding duplicate features
    keys_ = [ind*ones(length(I),1) frame*ones(length(I),1) I];
    [~, ~, dups] = intersect(caches{i}.keys, keys_, 'rows');
    keep = setdiff(1:size(keys_,1), dups);
    I = I(keep);

    % Unique hard negatives
    X_neg{i} = feat(I,:);
    keys{i} = [ind*ones(length(I),1) frame*ones(length(I),1) I];
  end
end

% ------------------------------------------------------------------------
function X = rcnn_scale_features_combine(X, feat_norm1, feat_norm2)
% ------------------------------------------------------------------------
target_norm = 20;
X1 = X(:,1:4096);
X2 = X(:,4096+(1:4096));
assert(size(X,2)==2*4096);
X1 = X1 .* (target_norm / feat_norm1);
X2 = X2 .* (target_norm / feat_norm2);
X = [X1 X2];

% ------------------------------------------------------------------------
function model = update_model(model, cache, opts)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 3;  % l2 regularized l1 hinge loss
%liblinear_type = 5; % l1 regularized l2 hinge loss

%solver = 'liblinear-weights';
%solver = 'lbfgs';

num_pos = size(cache.X_pos, 1);
num_neg = size(cache.X_neg, 1);

switch solver
  case 'liblinear'
    ll_opts = sprintf('-w1 %.10f -c %.10f -s %d -B %.10f', ...
                      opts.pos_loss_weight, opts.svm_C, liblinear_type, opts.bias_mult);
    fprintf('liblinear opts: %s\n', ll_opts);
    X = sparse(size(cache.X_pos,2), num_pos+num_neg);
    X(:,1:num_pos) = cache.X_pos';
    X(:,num_pos+1:end) = cache.X_neg';
    y = cat(1, ones(num_pos,1), -ones(num_neg,1));
    llm = liblinear_train(y, X, ll_opts, 'col');
    model.w = single(llm.w(1:end-1)');
    model.b = single(llm.w(end)*opts.bias_mult);

  otherwise
    error('unknown solver: %s', solver);
end


% ------------------------------------------------------------------------
function X_pos = get_positive_features(models, dataset, opts)
% ------------------------------------------------------------------------
X_pos = cell(length(models), 1);

for i = 1:length(dataset.videos)
    
  fprintf('[%d/%d]\n',i,length(dataset.videos));
  
  video_id = dataset.videos(i).video_id;
  action = dataset.videos(i).action;
  num_frames = max(opts.annot.frame(strcmp(video_id,opts.annot.video_id)));
  
  for f=1:num_frames
 
    mat_file_spat = sprintf(opts.feat_dir_spat,action,video_id,f);
    mat_file_flow = sprintf(opts.feat_dir_flow,action,video_id,f);
    
    d_spat = load(mat_file_spat);
    d_flow = load(mat_file_flow);
    assert(isequal(d_spat.boxes,d_flow.boxes));
    assert(isequal(d_spat.gt,d_flow.gt));
    
    j = find(strcmp(action,opts.actions));
  
    if isempty(X_pos{j})
      X_pos{j} = single([]);
    end
    
    sel = find(d_spat.gt);
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, [d_spat.feat(sel,:) d_flow.feat(sel,:)]);
    end
    
  end
end


% ------------------------------------------------------------------------
function model = init_model(cls, dataset, opts)
% ------------------------------------------------------------------------
model.class = cls;
model.w = [];
model.b = [];
model.thresh = -1.1;
model.opts = opts;


% ------------------------------------------------------------------------
function cache = init_cache(model, X_pos)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = single([]);
cache.keys = [];
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];

% ------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% ------------------------------------------------------------------------

inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;

iou = inters./(union+0.001);

% ------------------------------------------------------------------------
function [videos] = get_data(split_id,target_id,actions)
% ------------------------------------------------------------------------

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
    if data{2}(j)==target_id
      num = num+1;
      videos(num).video_id = video;
      videos(num).action = actions{i};
      videos(num).action_id = i;
    end
  end
end

% ------------------------------------------------------------------------
function [mean_norm, stdd] = feat_stats(videos,feat_dir)
% ------------------------------------------------------------------------

num_images = min(length(videos), 200);
boxes_per_image = 200;
iind = randperm(length(videos),num_images);

ns = [];
for i=1:num_images
    
  if mod(i,50)==1
    fprintf('Mean stats: [%d/%d]\n',i,num_images);
  end
  
  id = iind(i);
  video_id = videos(id).video_id;
  action   = videos(id).action;
  
  % pick the first frame (doesn't really matter)
  f=1;
  d = load(sprintf(feat_dir,action,video_id,f));
  X = d.feat;
  ind = randperm(size(X,1),min([boxes_per_image, size(X,1)]));
  X = X(ind,:);
  
  ns = cat(1, ns, sqrt(sum(X.^2, 2)));
  
end
fprintf('# of samples: %d\n',size(ns,1));

mean_norm = mean(ns);
stdd = std(ns);