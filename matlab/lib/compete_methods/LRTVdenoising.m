function X = LRTVdenoising(D,sigma)

%==========================================================================
% Solving
%   || X ||_* + lambda * \sum_j || X_j ||_TV + beta * || X - D ||_F^2
%   
% D ---> input noisy data
% X ---> denoising result
% 
% by Qi Xie
%==========================================================================
r = 3;
sizeD = size(D);
[mu, rho, beta, lambda] = parSetLRTV(sizeD,sigma, r);
muM = mu;
muZ = mu;

LamM  = zeros(sizeD);
LamZ  = zeros([sizeD,2]);
DX    = zeros([sizeD,2]);
fft1D = zeros([sizeD,2]);
for i = 1:2
    fft1D(:,:,:,i)    = permute((psf2otf([+1, -1], sizeD([i:2,1:i-1,3]))),[2-i+2:2,1:2-i+1,3]);
end
allfftn =  sum(fft1D.*conj(fft1D),4);
X = D;
for i = 1:100
    
    % Update M
    tempX    =   Unfold(X+LamM/muM,sizeD,3)';
    [U,S,V]  =   svd(tempX,'econ');
    M        =   Fold(V*diag(max(diag(S)-1/muM,0))*U',sizeD,3);
    
    % Update Z
    for j  = 1:2
        DX(:,:,:,j)  = real(ifft2(fft1D(:,:,:,j).*fft2(X)));
    end
    Z = max(min(0, DX+LamZ/muZ+lambda/muZ), DX+LamZ/muZ-lambda/muZ);
    
    % Update X
    fftnDZ = zeros(sizeD);
    for j = 1:2
        fftnDZ = fftnDZ + conj(fft1D(:,:,:,j)).*fft2(muZ*Z(:,:,:,j)-LamZ(:,:,:,j));
    end
    X = real(ifft2( ( fft2(beta*D + muM*M -LamM)  + fftnDZ  ) ./ ( beta+muM+ muZ*allfftn) ));

    % Update mutiplayers
    LamM = LamM  +  muM*(X-M);
    LamZ = LamZ  +  muZ*(DX-Z);
    
    muM = muM*rho;
    muZ = muZ*rho;
end
end

function [mu, rho, beta, lambda] = parSetLRTV(sizeD,sigma, r)

if sigma<=0.1
%     beta  = 80000/(sigma*prod(sizeD));
    beta  = 120000/(sigma*prod(sizeD));%urban
%     lambda = sqrt(r/(0.01*prod(sizeD)));
    lambda = sqrt(r/(1*prod(sizeD))); %urban
    rho = 1.05;
    mu = 1e-2;
elseif sigma<=0.15
    beta  = 90000/(sigma*prod(sizeD));
    lambda = sqrt(r/(0.01*prod(sizeD)));
    rho = 1.05;
    mu = 1e-2;
elseif sigma<=0.2
    beta  = 90000/(sigma*prod(sizeD));
    lambda = sqrt(r/(0.01*prod(sizeD)));
    rho = 1.05;
    mu = 1e-2;
elseif sigma<=0.25
    beta  = 90000/(sigma*prod(sizeD));
    lambda = sqrt(r/(0.01*prod(sizeD)));
    rho = 1.05;
    mu = 1e-2;
elseif sigma<=0.3
    beta  = 90000/(sigma*prod(sizeD));
    lambda = sqrt(r/(0.01*prod(sizeD)));
    rho = 1.05;
    mu = 1e-2;
end
end

