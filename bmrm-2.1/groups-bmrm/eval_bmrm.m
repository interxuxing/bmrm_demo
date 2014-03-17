function eval_bmrm()
%% function eval_bmrm is a script to use standard measures to evaluate
% the prediction results predicted by bmrm model

pred_dir = 'D:\workspace-limu\image-annotation\mm2014\flickr_bmrm\bmrm_demo\bmrm-2.1\groups-bmrm\preds';
prefix_predfile = 'pred_';

gt = [];
pred = [];

for file_idx = 26041:26060
    pred_filename = sprintf('%s%d.txt', prefix_predfile, file_idx);
    C = dlmread(fullfile(pred_dir, pred_filename));
    
    gt = [gt, C(:,1)];
    pred = [pred, C(:,2)];
end

% set -1 in gt to 0
gt(gt == -1) = 0;

% sort the predict value to get top 5 values
[sort_value, sort_index] = sort(pred, 2, 'descend');
sort_index_tp5 = sort_index(:, 1:5);

Pred = zeros(size(pred));

for i = 1:size(Pred, 1)
    Pred(i,sort_index_tp5(i,:)) = 1;
end

% now evaluate
results = evalRetrieval(Pred, gt);
