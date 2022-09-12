%% =========================== Frist part notes ===========================
% ADMM algorithm: tensor denosing
%    solve the following problem 
%           (SSTV regularized low rank tucker decomposition problem)
%    
%      The approximate model:  
%                                min tau*||F||_1 + lambda*||E||_1
%                             s.t. D = X+E, X = Z,DZ = F
%                                  X = Core*U1*U2*U3,rank(Core_i)=ri 
%      where,D is SSTV difference operator in the above model
%
%      The lagrange function is :
%       tau*||F||_1 + lambda*||E||_1 + <M1, D-X-E> + <M2,X-Z> +<Gamma,DZ - F>
%           + beta/2*( ||D-X-E||_F^2 + ||X-Z||_F^2 + ||DZ - F||_F^2 )
%
%% =========================== Second part notes===========================

% Reference paper: Hyperspectral Image Restoration via Total Variation 
%                      Regularized Low-rank Tensor Decomposition
% Author: Yao Wang, Jiangjun Peng
% E-mail addresses: andrew.pengjj@gmail.com
% -------------------------------------------------------------------------

%% =========================== Thrid part notes =========================== 
% INPUT:
%   Noi:     noisy 3-D image of size M*N*p normalized to [0,1] band by band
%   tau:     the trade-off parameter (recommended value 1)              
%   lambda:  sparse noise coefficient
%   rank:    rank constraint,[0.8*M,0.8*N,r]
% OUTPUT:
%  clean_iamge:   3-D denoised image
%  S:             The noise term
%  out_value:     MPSNR and MSSIM and ERGAS valuses of each iteration 
%  ========================================================================
function [clean_image,S] = LRTDTV(Noi, tau,lambda,rank)
sizeD           = size(Noi);
normD           = norm(Noi(:)); 
n               = prod(sizeD);
maxIter         = 50;
epsilon         = 1e-4;  
beta            = 0.01;             % The ascending multiplier value


%%  Initialization 
X               = rand(sizeD);      % X : The clean image
Z               = X;                % Z : auxiliary variable for X
S               = zeros(sizeD);     % S : sparse noise 
F               = zeros(3*n,1);     % F : auxiliary variable for tv
Gamma           = F;                % The multiplier for DZ-F
M1              = zeros(size(Noi)); % The multiplier for          
M2              = M1;

%% main loop

for iter = 1: maxIter
    preX       = X;
    %% - update Core and U_i and X
    temp       = 1/2*(Noi-S+Z+(M1-M2)/beta);
    X          = tucker_hooi(temp,rank);
    %% - update Z
    z          = Z(:);
    z          = myPCG1(z,X,M2,F,Gamma,beta,sizeD);
    Z          = reshape(z,sizeD);
    %% - update F
    diff_Z     = diff3(Z(:), sizeD); 
    F          = softthre(diff_Z+ Gamma/beta, tau/beta );  
    %% - update S 
    S          = softthre(Noi-X+M1/beta,lambda/beta);% sparse
%     S          = beta*(Noi-X+M1/beta)/(beta+2*lambda);
    %% - update M
    M1         = M1 + beta*(Noi-X-S);
    M2         = M2 + beta*(X-Z); 
    %% - update Gamma
    Gamma      = Gamma+beta*(diff_Z-F);            
    beta       = min(beta *1.5,1e5); 
    %% compute the error
    errList    = norm(X(:)-preX(:)) / normD;
%     fprintf('LRTDTV: iterations = %d   difference=%f\n', iter, errList);
    if errList < epsilon
        break;  
    end 

end
%% the final clean image
clean_image = X;
% fprintf('LRTDTV ends: total iterations = %d,difference=%f\n\n', iter, errList);

end

