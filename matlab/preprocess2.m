%%
% This file is modified a little bit based on the original one preprocess.m. 
% Please download the original codes from --> 
% https://www.mathworks.com/matlabcentral/fileexchange/49172-trainable-cosfire-filters-for-curvilinear-structure-delineation-in-images
% And please cite the following paper(s) if you use this code. 
%   [1] "George Azzopardi, Nicola Strisciuglio, Mario Vento, Nicolai Petkov, 
%   Trainable COSFIRE filters for vessel delineation with application to retinal images, 
%   Medical Image Analysis, Volume 19 , Issue 1 , 46 - 57, ISSN 1361-8415, 
%   http://dx.doi.org/10.1016/j.media.2014.08.002"

%%
function [img, mask] = preprocess2(img, th, negative)
    
%img = padarray(img,[21 21]);

R = img(:,:,1);
G = img(:,:,2);
B = img(:,:,3);

[L,a,b] = RGB2Lab(R,G,B);
L = L./100;
mask = 1- (L < th);

img = G;
if negative==true
    img = 1 - img;
    img(mask<0.5) = 0;
end

[ignore, img] = getBigimg(img, mask);
img = adapthisteq(img);


function [bigimg, smallimg] = getBigimg(img,mask)

[sizey, sizex] = size(img);

bigimg = zeros(sizey + 100, sizex + 100);
bigimg(51:(50+sizey), 51:(50+sizex)) = img;

bigmask = logical(zeros(sizey + 100, sizex + 100));
bigmask(51:(50+sizey), (51:50+sizex)) = mask;

% Creates artificial extension of image.
bigimg = fakepad(bigimg, bigmask, 5, 10);
smallimg = bigimg(51:(50+sizey), 51:(50+sizex));

function [L,a,b,X,Y,Z] = RGB2Lab(R,G,B)
% function [L, a, b] = RGB2Lab(R, G, B)
% RGB2Lab takes matrices corresponding to Red, Green, and Blue, and 
% transforms them into CIELab.  This transform is based on ITU-R 
% Recommendation  BT.709 using the D65 white point reference.
% The error in transforming RGB -> Lab -> RGB is approximately
% 10^-5.  RGB values can be either between 0 and 1 or between 0 and 255.  
% By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
% Updated for MATLAB 5 28 January 1998.

if (nargin == 1)
  B = double(R(:,:,3));
  G = double(R(:,:,2));
  R = double(R(:,:,1));
end

% if ((max(max(R)) > 1.0) | (max(max(G)) > 1.0) | (max(max(B)) > 1.0))
%   R = R/255;
%   G = G/255;
%   B = B/255;
% end

[M, N] = size(R);
s = M*N;

% Set a threshold
T = 0.008856;

RGB = [reshape(R,1,s); reshape(G,1,s); reshape(B,1,s)];

% RGB to XYZ
MAT = [0.412453 0.357580 0.180423;
       0.212671 0.715160 0.072169;
       0.019334 0.119193 0.950227];
XYZ = MAT * RGB;

X = XYZ(1,:) / 0.950456;
Y = XYZ(2,:);
Z = XYZ(3,:) / 1.088754;

XT = X > T;
YT = Y > T;
ZT = Z > T;

fX = XT .* X.^(1/3) + (~XT) .* (7.787 .* X + 16/116);

% Compute L
Y3 = Y.^(1/3); 
fY = YT .* Y3 + (~YT) .* (7.787 .* Y + 16/116);
L  = YT .* (116 * Y3 - 16.0) + (~YT) .* (903.3 * Y);

fZ = ZT .* Z.^(1/3) + (~ZT) .* (7.787 .* Z + 16/116);

% Compute a and b
a = 500 * (fX - fY);
b = 200 * (fY - fZ);

L = reshape(L, M, N);
a = reshape(a, M, N);
b = reshape(b, M, N);

if ((nargout == 1) | (nargout == 0))
  L = cat(3,L,a,b);
end