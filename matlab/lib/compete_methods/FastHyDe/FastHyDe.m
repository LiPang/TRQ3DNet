function [image_fasthyde, time] = FastHyDe(img_ori,  noise_type, iid, k_subspace)

% Input: -----------
% img_ori        hyperspectral data set with (L x C x N),
%                where L, C, and N are the number of rows,
%                columns, and bands, respectively.

% noise_type     {'white','poisson'}
%
% iid            iid = 1 --  Gaussian i.i.d. noise.
%                iid = 0 --  Gaussian non-i.i.d.
%                if noise_type is set to 'poisson', then the code does not
%                use the value of iid, which can be set to any value.
%
% k_subspace     signal subspace dimension
%
% Output: -----------
% image_fasthyde Denoised hyperspectral data with (L x C x N)
%
% time           Runing time of FastHyDe
%
% ---------------------------- -------------------------------------------
% See more details in papers:
%   [1] L. Zhuang and J. M. Bioucas-Dias, 
%       "Fast hyperspectral image denoising based on low rank and sparse 
%       representations,?in 2016 IEEE International Geoscience and Remote
%       Sensing Symposium (IGARSS 2016), 2016.
%
%   [2] L. Zhuang and J. M. Bioucas-Dias, 
%       "Fast hyperspectral image denoising and inpainting based on low rank 
%       and sparse representations,?Submitted to IEEE Journal of Selected
%       Topics in Applied Earth Observations and Remote Sensing, 2017.
%       URL: http://www.lx.it.pt/~bioucas/files/submitted_ieee_jstars_2017.pdf
%
%% -------------------------------------------------------------------------
%
% Copyright (July, 2017):        
%             Lina Zhuang (lina.zhuang@lx.it.pt)
%             &
%             Jos?Bioucas-Dias (bioucas@lx.it.pt)
%            
%
% FastHyDe is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ---------------------------------------------------------------------




t1=clock;
% addpath('BM3D');
[Lines, Columns, B] = size(img_ori);
N=Lines*Columns;



%% ----------------------------- Data transformation ---------------------------------
% Observed data with additive Gaussian non-iid noise or Poissonian noise are transformed
% in order to  to have additive Gaussian i.i.d. noise before the denoisers are applied.

switch noise_type
    case 'additive'
        if iid==0 %additive Gaussian non-iid noise, applying eq. ?
            Y = reshape(img_ori, N, B)';
            [w Rw] = estNoise(Y,noise_type);
            Rw_ori = Rw;
            Y = sqrt(inv(Rw_ori))*Y;
            img_ori = reshape(Y', Lines, Columns, B);
        end
    case 'poisson'
        % applying the Anscombe transform, which converts Poissonion noise
        % into approximately additive  noise.
        img_ori = 2*sqrt(abs(img_ori+3/8));
end

Y = reshape(img_ori, N, B)';

%% subspace estimation using HySime or SVD

[w Rw] = estNoise(Y,'additive');
[~, E]=hysime(Y,w,Rw);
%[E,~,~]= svd(Y,'econ');

E=E(:,1:k_subspace);

eigen_Y = E'*Y;

%% --------------------------Eigen-image denoising ------------------------------------





eigen_Y_bm3d=[];
for i=1:k_subspace
    % produce eigen-image
    eigen_im = eigen_Y(i,:);
    min_x = min(eigen_im);
    max_x = max(eigen_im);
    eigen_im = eigen_im - min_x;
    scale = max_x-min_x+1e-5;
    
    %scale to [0,1]
    eigen_im = reshape(eigen_im, Lines, Columns)/scale;
    
    
    %estimate noise from Rw
    sigma = sqrt(E(:,i)'*Rw*E(:,i))/scale;
    
    % denoise  with BM3D
    if exist('BM3D.m','file') == 0 %Check existence of  function BM3D
         errordlg({'Function BM3D.m not found! ','Download from http://www.cs.tut.fi/~foi/GCF-BM3D and install it in the folder .../BM3D'});
     error('Function BM3D.m not found!  Download from http://www.cs.tut.fi/~foi/GCF-BM3D and install it in the folder .../BM3D');

    else
    [~, filt_eigen_im] = BM3D(1,eigen_im, sigma*255);
    end
    eigen_Y_bm3d(i,:) = reshape(filt_eigen_im*scale + min_x, 1,N);
    
    
end

% reconstruct data using denoising engin images
Y_reconst = E*eigen_Y_bm3d;

%% ----------------- Re-transform ------------------------------


switch noise_type
    case 'additive'
        if iid==0
            Y_reconst = sqrt(Rw_ori)*Y_reconst;
        end
        
    case 'poisson'
        Y_reconst =(Y_reconst/2).^2-3/8;
end

image_fasthyde=[];

image_fasthyde = reshape(Y_reconst',Lines,Columns,B);


t2=clock;
time = etime(t2,t1);
end