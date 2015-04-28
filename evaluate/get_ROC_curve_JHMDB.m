function output = get_ROC_curve_JHMDB(annot,tubes,actions,iou_thresh,draw)
% ---------------------------------------------------------
% Copyright (c) 2015, Georgia Gkioxari
% 
% This file is part of the Action Tubes code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

videos = unique({tubes.video_id});
TOP = 3;

num_actions = length(actions);
cc = zeros(num_actions,1);
for a=1:num_actions
  all_scores{a} = zeros(100000,2,'single');
end
covered = cell(num_actions,1);
num_gt = zeros(num_actions,1);

for i=1:length(videos)
    
  video_id = videos(i);
  
  fprintf('(%d/%d)\n',i,length(videos));
  
  % in tubes
  tube_ind = strcmp(video_id,{tubes.video_id});
  
  % in ground truth
  gt_ind = find(strcmp(video_id, annot.video_id));
  num_frames = max(annot.frame(gt_ind));
  
  fprintf('# of frames = %d\n',num_frames);
  
  gt_bounds = nan(num_frames, 4);
  gt_action = unique(annot.action(gt_ind));
  gt_action_id = find(strcmp(gt_action,actions));
  assert(length(gt_action_id)==1); % in JHMDB there is one action per video
  for f=1:num_frames
    keep = strcmp(video_id,annot.video_id) & annot.frame==f;
    gt_bounds(f,:) = annot.bound(keep,:);
  end
  
  num_gt(gt_action_id) = num_gt(gt_action_id)+1;
  
  for a=1:num_actions
    paths = tubes(tube_ind).paths{a};
    num_paths = length(paths);
    
    iou = nan(num_paths,1);
    assignment = nan(num_paths,1);
    scrs = nan(num_paths,1);
    
    for p=1:num_paths
      overlap = nan(num_frames,1);
      for f=1:num_frames
        overlap(f) = inters_union(gt_bounds(f,:),[paths(p).boxes(f,1:2) paths(p).boxes(f,3:4)-paths(p).boxes(f,1:2)+1]);
      end
      overlap = mean(overlap,1);
      iou(p)  = overlap;
      scrs(p) = paths(p).total_score;
    end
    
    % keep top 3
    [s si] = sort(scrs,'descend');
    si = si(1:min(TOP,length(si)));
    
    iou = iou(si);
    scrs = scrs(si);
    for j=1:length(iou)
      cc(a) = cc(a)+1;
      if iou(j)>=iou_thresh && ~ismember(i,covered{a}) && a==gt_action_id
        % correct detection
        all_scores{a}(cc(a),:) = [scrs(j) 1];
        covered{a} = [covered{a}; i];
      else
        % false detection 
        all_scores{a}(cc(a),:) = [scrs(j) 0];
      end
    end
    
  end % end for a
end

%% compute auc
for a=1:num_actions
  all_scores{a} = all_scores{a}(1:cc(a),:);
  scores = all_scores{a}(:,1);  
  labels = all_scores{a}(:,2);
  assert(sum(labels)<=num_gt(a));
  [auc(a),tpr{a},fpr{a}] = get_roc_curve(scores,labels,num_gt(a));
  fprintf('%s [IOU=%.2f] AUC = %.2f [%d]\n',actions{a},iou_thresh,auc(a)*100,num_gt(a));
  if draw
    figure;
    plot(fpr{a},tpr{a}); title(actions{a}); grid on;
  end
end
fprintf('Average AUC %.2f\n',mean(auc)*100);

[all_auc,all_tpr,all_fpr] = get_average_roc_curve(tpr,fpr);
fprintf('Mean ROC: AUC = %.2f\n',all_auc*100);
output.thresh = iou_thresh;
output.auc = all_auc;
output.tpr = all_tpr;
output.fpr = all_fpr;

if draw
  figure;
  plot(all_fpr,all_tpr); axis([0 0.6 0 1]); grid on;
end


% -------------------------------------------------------------------------
function [all_auc,all_tpr,all_fpr] = get_average_roc_curve(tpr,fpr)
% -------------------------------------------------------------------------
num_curves = length(tpr);
% Average plots
all_fpr = [];
for a=1:num_curves
  all_fpr = unique(union(all_fpr,fpr{a}));
end
all_tpr = zeros(length(all_fpr),1);
for a=1:num_curves
  for i=1:length(all_fpr)
    [diff, ind1] = min(abs(all_fpr(i)-fpr{a}));
    v1 = fpr{a}(ind1);
    if v1<=all_fpr(i)
      ind2 = ind1+1;
    else
      ind2 = ind1-1;
    end
    if ind2<1 || ind2>length(tpr{a}) || diff==0, ind2 = ind1; end
    all_tpr(i) = all_tpr(i) + max(tpr{a}([ind1 ind2]));
  end
end
all_tpr = all_tpr/num_curves;


mtpr = all_tpr;
mfpr = all_fpr;

i=find(mfpr(2:end)~=mfpr(1:end-1))+1;
all_auc=sum((mfpr(i)-mfpr(i-1)).*mtpr(i));



% -------------------------------------------------------------------------
function [auc,tpr,fpr] = get_roc_curve(scores,labels,num_pos)
% -------------------------------------------------------------------------

[srt1,srtd]=sort(scores,'descend');
scores = srt1;
num_neg = sum(labels==0);

fp=cumsum(labels(srtd)==0);
tp=cumsum(labels(srtd)==1);
fn=num_pos-tp; % TP+FN=num_pos
tn=num_neg-fp; % TN+FP=num_neg

tpr = tp./(tp+fn);
fpr = fp./(fp+tn);

keep = fpr<=0.6;
tpr = tpr(keep);
fpr = fpr(keep);

mtpr = tpr;
mfpr = fpr;

i=find(mfpr(2:end)~=mfpr(1:end-1))+1;
auc=sum((mfpr(i)-mfpr(i-1)).*mtpr(i));


% -------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% -------------------------------------------------------------------------

inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;

iou = inters./(union+eps);