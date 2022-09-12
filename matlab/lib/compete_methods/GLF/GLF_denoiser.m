function  img_denoised  = GLF_denoiser(img_noisy,p, noise_type,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% GLF_denoiser(img_noisy,p, noise_type,varargin)
%
% -- Input arguments
%
%  img_noisy:  noisy image of size rows*columns*bands
%  p:          dimension of subsapce, which can be estimated by subsapce
%              identification methods (like HySime) or be hand-tuned.
%  noise_type: noise type can only be 'additive'. This demo is written for
%              images with additive Gaussian noise.
%
%
%  INPUT ARGUMENTS (OPTIONAL):
%
%  f:     length of a patch, meaning size of a patch is f*f.
%  t:     radius of window for searching similar patches, meaning  size of the search
%         windiow is (2*t+1)*(2*t+1)
%  Nstep: sliding step to process every next reference patch
%
% --- Output
%
% img_denoised: denoised image
%
% USAGE EXAMPLES:
%
%     Case 1) Using the default parameters (i.e., f and t.)
%
%       img_GLF = GLF_denoiser(img_noisy,  p_subspace,  noise_type) ;
%
%     Case 2) Using user-provided argument:
%       img_GLF = GLF_denoiser(img_noisy,  p_subspace,  noise_type, 'f', 10,'t', 39, 'Nstep', 3) ;
%
% ---------------------------- -------------------------------------------
% See more details in papers:
%
%  Lina Zhuang and Jose M. Bioucas-Dias, "Hyperspectral image denoising
%  based on global and non-local low-rank factorizations",  IEEE
%  International Conference on Image Processing, Sep. 2017.
%  URL: https://www.it.pt/Publications/DownloadPaperConference/30727
% 
%% -------------------------------------------------------------------------
%
% Copyright (Nov., 2017):        
%             Lina Zhuang (lina.zhuang@lx.it.pt)
%             &
%             José Bioucas-Dias (bioucas@lx.it.pt)
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


n=numel(varargin);
if n~=0
    for i=1:2:n
        if strcmp(varargin{i},'f')
            f = varargin{i+1};
        end
        if strcmp(varargin{i},'t')
            t = varargin{i+1};
        end
         if strcmp(varargin{i},'Nstep')
            Nstep = varargin{i+1};
        end
    end
    
end
if exist('f') ==0
    f = 10;
end
if exist('t') ==0
    t = 39;
end
if exist('Nstep') ==0
    Nstep = 3;
end

switch noise_type
    case 'additive'
        %do nothing
    otherwise
        %warning:
              errordlg('Warning! This demo is written for images with additive Gaussian noise.');
     error('Warning! This demo is written for images with additive Gaussian noise.');
  
end

[r, c, b] = size(img_noisy);
N=r*c;
Y_noisy = reshape(img_noisy, N, b)';


%------------whiten noise--------------
% Assume that noise is zero-mean, Gaussianm additive, bandwise and 
% pixelwise independent with band dependent variance

[w Rw] = estNoise(Y_noisy,noise_type);
Rw_save=Rw;%+eye(B)*0.000001;
Y_noisy = sqrt(inv(Rw_save))*Y_noisy;


%------------global low-rank matrix factorization--------------
[S,~,~] = svd(Y_noisy*Y_noisy');
Y_noisy = S(:,1:p)'*Y_noisy;
img_noisy = reshape(Y_noisy', r, c, p);

%-----------local low-rank tensor factorization

%noise std of each entry of img_noisy is 1, since noise has been whitened.
img_GLF = NonlocalPatch_local_LR(img_noisy,t,f,Nstep);


Y_GLF= reshape(img_GLF,N,p)';
Y_GLF=S(:,1:p)*Y_GLF;
Y_GLF = sqrt(Rw_save)*Y_GLF;
img_denoised = reshape(Y_GLF', r, c, b);

 