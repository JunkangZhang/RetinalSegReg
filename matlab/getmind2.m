addpath(genpath('monogenic_signal_matlab-master'))
addpath('GN-MIND2d')

%%
img_path = '../ckpt/FFAPCFIDP_random_offset';
save_path = [img_path, '_phase-mind'];
try mkdir(save_path); catch end

%%
gridsize = 40;
imshow = false;
phases = {'tr', 'te'};
tic
for p =1:numel(phases)
    phase = phases{p};
    
    [src_all, tgt_all] = LoadFFAPCFIDP(img_path, phase);
    ust = zeros(numel(src_all), size(src_all{1},1), size(src_all{1},2));
    vst = zeros(numel(src_all), size(src_all{1},1), size(src_all{1},2));
    tot = numel(src_all);

    parfor i = 0:(tot-1)
        imo_1 = tgt_all{i+1};
        if size(imo_1,3)>3
            imo_1 = imo_1(:,:,1:3);
        end

        imo_2 = src_all{i+1};
        if size(imo_2,3)>3
            imo_2 = imo_2(:,:,1:3);
        end

        %% with phase
        imlp_1 = ExtractLocalPhase(1-rgb2gray(imo_1)) / pi;
        imlp_2 = ExtractLocalPhase(rgb2gray(imo_2)) / pi;
        [u1,v1,deformed] = deformableReg2Dmind(imlp_1, imlp_2, 0.2);

        %% without phase
%         [u1,v1,deformed] = deformableReg2Dmind(1-rgb2gray(imo_1), rgb2gray(imo_2), 0.2);

        %%
        ust(i+1, :, :) = u1;
        vst(i+1, :, :) = v1;

        %%
        if imshow==true
%             figure(1); cla; imshow(imo_1)
%             figure(2); cla; imshow(imo_2)
%             imo_c = CombineView(imo_1, imo_2, gridsize, gridsize);
%             figure(3); cla; imshow(imo_c); 
%             imwrite(imo_c, sprintf('%s/%s_%02d_grid.png', save_path, phase, i));

%             figure(11); cla; imshow(imlp_1)
%             figure(12); cla; imshow(imlp_2)

%             imo_t_2 = imWarp(u1,v1,imo_2);
%             figure(21); cla; imshow(imo_t_2); 
%             imwrite(imo_t_2, sprintf('%s/%s_%02d_src_t.png', save_path, phase, i));

%             imo_t_c = CombineView(imo_1, imo_t_2, gridsize, gridsize);
%             figure(22); cla; imshow(imo_t_c); 
%             imwrite(imo_t_c, sprintf('%s/%s_%02d_grid_t.png', save_path, phase, i));
        end

        fprintf('%d\n', i);
    end
    
    save(sprintf('%s/%s_flow.mat', save_path, phase), 'ust', 'vst');
    toc
end

%%
function [imw] = CombineView(imw, imo, w, h)
    if ndims(imw) == 2 && ndims(imo)==3
        imw = cat(3, imw, imw, imw);
    end
    for r = 1:ceil(size(imo,1)/h)
        for c = 1:ceil(size(imo,2)/w)
            if mod(r+c,2)==1
                continue
            end
            rst = (r-1)*h+1; 
            ren = min(r*h, size(imo,1));
            cst = (c-1)*w+1;
            cen = min(c*w, size(imo,2));
            if ndims(imw) == 2 && ndims(imo)==2
                imw(rst:ren, cst:cen) = imo(rst:ren, cst:cen);
            elseif ndims(imw) == 3 && ndims(imo)==3
                imw(rst:ren, cst:cen, :) = imo(rst:ren, cst:cen, :);
            else%if ndims(imw) == 3 && ndims(imo)==1
                for i = 1:3
                  imw(rst:ren, cst:cen, i) = imo(rst:ren, cst:cen);
                end
            end
        end
    end
end

function [LP_mean] = ExtractLocalPhase(I)
    [Y,X] = size(I);
    cw = 5*1.5.^(0:10);
    filtStruct = createMonogenicFilters(Y,X,cw,'lg',0.55);
    [m1,m2,m3] = monogenicSignal(I,filtStruct);
    LP = localPhase(m1,m2,m3);
    LP_mean = squeeze(mean(LP, 4));
end
