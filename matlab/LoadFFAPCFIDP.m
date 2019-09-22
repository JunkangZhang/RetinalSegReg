function [src_all, tgt_all] = LoadFFAPCFIDP(img_path, phase)
    if strcmp(phase(1:2), 'te'), tot = 58; else, tot = 60; end

    src_all = {}; 
    tgt_all = {};
    for i = 0:(tot-1)
        load(sprintf('%s/%s_%02d_src.mat', img_path, phase, i));
        src_all{i+1} = flip(im, 3);
        load(sprintf('%s/%s_%02d_tgt.mat', img_path, phase, i));
        tgt_all{i+1} = flip(im, 3);
    end
end