function paths = zero_jump_link(frames)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Georgia Gkioxari
% 
% This file is part of the Action Tubes code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% set of vertices
num_frames = length(frames);
V = 1:num_frames;
isempty_vertex = false(length(V),1);

% CHECK: make sure data is not empty
for i=1:num_frames
  if isempty(frames(i).boxes)
    error('Empty frame'); 
  end
end


%% dymamic programming (K-paths)
num_total = 0;

while ~any(isempty_vertex)
    
  %fprintf('# of iteration = %d\n',num_total);
  % step1: initialize data structures
  T = length(V);
  for i=1:T
    num_states = size(frames(V(i)).boxes,1);
    data(i).scores = zeros(num_states,1);
    data(i).index  = nan(num_states,1);
  end

  % step2: solve viterbi
  for i=T-1:-1:1
    edge_score = score_of_edge(frames(V(i)),frames(V(i+1)));
  
    edge_score = bsxfun(@plus,edge_score,data(i+1).scores');
    [data(i).scores, data(i).index] = max(edge_score,[],2);
  
  end

  % step3: decode
  num_total = num_total+1;
  [s, si] = sort(data(1).scores,'descend');
  id = si(1);
  score = data(1).scores(id);
  index = id;
  boxes = frames(V(1)).boxes(id,:);
  scores = frames(V(1)).scores(id,1);
  for j=1:T-1
    id = data(j).index(id);
    index = [index; id];
    boxes = [boxes; frames(V(j+1)).boxes(id,:)];
    scores = [scores; frames(V(j+1)).scores(id,:)];
  end
  paths(num_total).total_score = score/num_frames;
  paths(num_total).idx   = index;
  paths(num_total).boxes = boxes;
  paths(num_total).scores = scores;
  
  % step4: remove covered boxes
  for j=1:T
    id = paths(num_total).idx(j);
    frames(V(j)).boxes(id,:) = [];
    frames(V(j)).feat(id,:)  = [];
    frames(V(j)).scores(id)  = [];
    isempty_vertex(j) = isempty(frames(V(j)).boxes);
  end
end

% -------------------------------------------------------------------------
function score = score_of_edge(v1,v2)
% -------------------------------------------------------------------------

N1 = size(v1.boxes,1);
N2 = size(v2.boxes,1);
score = nan(N1,N2);

bounds1 = [v1.boxes(:,1:2) v1.boxes(:,3:4)-v1.boxes(:,1:2)+1];
bounds2 = [v2.boxes(:,1:2) v2.boxes(:,3:4)-v2.boxes(:,1:2)+1];

% f1 = v1.feat;
% f1 = bsxfun(@rdivide,f1,sqrt(sum(f1.^2,2)));
% f2 = v2.feat;
% f2 = bsxfun(@rdivide,f2,sqrt(sum(f2.^2,2)));

for i1=1:N1
  % feature similarity  
   feat_similarity = 0;
%   feat_similarity = f1(i1,:)*f2'; % cosine of angle
  
  % intersectin over union  
  iou = inters_union(bounds1(i1,:),bounds2);
  
  % scores
  scores2 = v2.scores;
  scores1 = v1.scores(i1);
  
  score(i1,:) = scores1+scores2'+feat_similarity+iou;
    
end



% -------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% -------------------------------------------------------------------------

inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;

iou = inters./(union+eps);
