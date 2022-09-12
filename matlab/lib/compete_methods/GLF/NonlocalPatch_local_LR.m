function [output_img]=NonlocalPatch_local_LR(img_noisy,t,f,step)
%Nonlocal patch-based denoising for subsapce coefficients
%
%[output_img]=NonlocalPatch_local_LR(img_noisy,t,f,step  )
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ---- Input
%  img_noisy: input 3D noisy image of size r*c*b, whose noise stand deviation is 1.
%  t:          radius of window for searching similar patches, meaning  size of the search
%              windiow is (2*t+1)*(2*t+1)
%  f:          length of a patch, meaning size of a patch is f*f.
%  step:       sliding step to process every next reference patch
% 
% ---- Output
%
%  output_img: denoised 3D image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N2  =   16;% 36;%  % maximum number of similar blocks (maximum size of the 3rd dimension of a 3D array)
p=1;%4; %apply SVD on img_noisy, then use first p components to compute similarity between two patches.



[r, c, b] = size(img_noisy);
N=r*c;
Y_noisy = reshape(img_noisy, N, b)';
%% Extract features : SVD

[U_ss,D]=svd(Y_noisy,'econ');
% plot(diag(D));
% first p component
U_ss(:,p+1:end) = [];
Y_feature =U_ss'*Y_noisy;
img_feature = reshape(Y_feature', r, c, p);
 

 
length_pat=f;
Num_Wsim = (length_pat)^2;  

% Replicate the boundaries of the input image
img_feature2 = padarray(img_feature,[f-1 f-1],'symmetric','post');


% total patches for the image
for ip = 1:p
    T_pat(:,:,ip) = im2col( img_feature2(:,:,ip), [length_pat,length_pat], 'sliding');
end

  

%index image,
idx_img = reshape( 1: N , [r, c]);
idx_img = padarray( idx_img, [f-1 f-1],'symmetric','post');
idx_img_pat=im2col( idx_img, [length_pat,length_pat], 'sliding');

 
   kernel=ones(length_pat^2,1)/length_pat^2; % each entry in a patch is equally important.

% B=[]; % store pix index
% C=[]; % store pix gray level,
% D=[]; % store weight

no_patches = size(1:step:c,2)*size(1:step:r,2);
length_patch = N2*f^2;
length_total = length_patch*no_patches;
B=zeros( length_total,1) ; % store pix index
C=zeros(b,length_total) ; % store pix gray level,
D=B; % store weight


% N_group_column = (length_pat)^2*N2;
% ones_group_column =ones(1,N_group_column);
%  Y_noisy = [Y_noisy, zeros(b,1)];
i_patch_count=0;
Y_denoised = zeros(size( Y_noisy));
count_Y_denoised = zeros(1,size(Y_noisy,2));
for j=1:step:c % col 
    for i=1:step :r % row    
        i_patch_count = i_patch_count+1;
         i1 = i;
          j1 = j;
        
        % reference patch id
        ref_pat_idx = idx_img(i1, j1);
        ref_pat=T_pat(:,ref_pat_idx,:);
        
       
         rmin = max(i1-t,1);
         rmax = min(i1+t,r+f-1);
         smin = max(j1-t,1);
         smax = min(j1+t,c+f-1);
        
        % other patch in the search window
        patches_idx=idx_img(rmin:1:rmax,smin:1:smax);
        patches_idx=patches_idx(:);
        %patches_idx=patches_idx(patches_idx~=0);
        patches=T_pat(:,patches_idx,:);
        
        
        dis =zeros(p,size(patches,2));
        
        
        for ip=1:p
            dis(ip,:) = sum( kernel * ones(1,size(patches, 2)).* bsxfun(@minus,patches(:,:,ip),ref_pat(:,1,ip)).^2);  
        end
        
        
        w = -sum(dis,1);
        [w_sort, I] = sort(w, 'descend');
        I = I(1: N2);
                
        
        if length(I)>= 1
            
            patches_sel_idx = patches_idx(I,1)';
            
            %non-local cubes:
            idx_similar_cube_2d = idx_img_pat(:,patches_sel_idx);
            
            idx_similar_cube =idx_similar_cube_2d(:)';
            
            
            group =  Y_noisy(:,idx_similar_cube);
            
            group = reshape(group', length_pat^2,N2, b);
            
            
           
            sigma = 1; %noise std of each entry of img_noisy is 1, since noise has been whiten.
            [group_est, weight] = LowRank_tensor( group,idx_similar_cube, sigma);
         

            for ic = 1:N2  %patch by patch
                tmp =  idx_similar_cube_2d(:,ic)';
                Y_denoised(:,tmp) = Y_denoised(:,tmp)+squeeze(group_est(:,ic,:))';
                count_Y_denoised(1,tmp) = count_Y_denoised(1,tmp)+weight;
            end
            
        end
    end
end

output_img = bsxfun(@rdivide,Y_denoised,count_Y_denoised);
output_img=reshape(output_img',r, c, b);
% figure; subplot(1,2,1);imagesc(img_noisy(:,:,1));
% subplot(1,2,2);imagesc(output_img(:,:,1));
end
 



