% function [output, oriensmap] = ExampleBloodVesselSegmentation( )
addpath('B-COSFIRE');
addpath(path,'B-COSFIRE/Gabor/');
addpath(path,'B-COSFIRE/COSFIRE/');
addpath(path,'B-COSFIRE/Preprocessing/');
addpath(path,'B-COSFIRE/Performance/');

%%
img_path = '../ckpt/FFAPCFIDP_random_offset';
save_path = [img_path, '_bcosfire'];
try mkdir(save_path); catch end

%%
phases = {'tr', 'te'};
whichs = {'src', 'tgt'};
tic
for p = 1:numel(phases)
    phase = phases{p};
    [src_all, tgt_all] = LoadFFAPCFIDP(img_path, phase);
    for w = 1:numel(whichs)
        which = whichs{w};
        
        if strcmp(which, 'src')
            im_all = src_all; negative = false;
        else
            im_all = tgt_all; negative = true;
        end

        %% Symmetric filter params
        symmfilter = struct();
        symmfilter.sigma     = 2.4;
        symmfilter.len       = 8;
        symmfilter.sigma0    = 3;
        symmfilter.alpha     = 0.7;

        %% Asymmetric filter params
        asymmfilter = struct();
        asymmfilter.sigma     = 1.8;
        asymmfilter.len       = 22;
        asymmfilter.sigma0    = 2;
        asymmfilter.alpha     = 0.1;

        %%
        resp_all = cell(numel(im_all), 1);
        mask_all = cell(numel(im_all), 1);
        parfor i = 1:numel(im_all)
            image = im_all{i};
            image = expand(image);
        %     figure(1); cla; imshow(image);

            [image, mask] = preprocess2(image, 0.15, negative);
        %     figure(2); cla; imshow(image);

            [respimage, oriensmap] = BCOSFIRE_mod(image, symmfilter, asymmfilter, mask);

            respimage = reduce(respimage);
            mask = reduce(mask);
        %     figure(3); cla; imshow(respimage/max(respimage(:))); 
        %     figure(4); cla; imshow(respimage>40); 
        %     figure(5); cla; imshow(mask); 

            resp_all{i} = respimage;
            mask_all{i} = mask;
            fprintf('%d\n', i);
        end

        %%
        for i=1:numel(im_all)
            resp = resp_all{i};
            mask = mask_all{i};
            save(sprintf('%s/%s_%02d_%s_bcosfire.mat', save_path, phase, i-1, which), 'resp', 'mask');
        end
        toc

    end
end

%%
function [nim] = expand(im)
    ss = size(im);
    nim = zeros(ss(1)+200, ss(2)+200, 3);
    nim(101:100+ss(1), 101:100+ss(2), :) = im;
end

function [nim] = reduce(im)
    ss = size(im);
    nim = im(101:ss(1)-100, 101:ss(2)-100, :);
end
