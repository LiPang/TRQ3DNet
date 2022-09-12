function [Emsi] = tSVD_DeNoising(Nmsi,nSig,memorySaving,Omsi)
% MSI denoising based on tSVD
%
% Input arguments:
%   Nmsi   ...... the MSI corrupted by tensor. Please make sure this MSI is of size height x width x nbands and in range [0, 1].
%   nSig   ...... the variance of noise.
%   Omsi   ...... the clean MSI use to calculate PSNR, this argument is not nesscery.
%
% Output arguments:
%   Emsi   ......  the denoised MSI
%
% by Qi Xie
%==========================================================================


if nargin<4
    OutPSNR = 0;
else
    OutPSNR = 1;
end

if nargin<2
    error('no input nSig')
elseif nargin <3
    memorySaving = 0;
end


sizeData        = size(Nmsi);
[par, parBSM]   = ParSetTSVD(nSig); % set denoising parameters and tensor recvery parameters;
Npatch          = Im2Patch3D(Nmsi, par);
sizePatch       = size(Npatch);
Emsi            = Nmsi;
[Sel_arr]       = nonLocal_arr(sizeData, par); % PreCompute the all the patch index in the searching window
L               = length(Sel_arr);
%% main loop
tic
for  iter = 1 : par.deNoisingIter
    Curmsi      	= Emsi + par.delta*(Nmsi - Emsi);
    Curpatch        = Im2Patch3D(Curmsi, par); % image to FBPs
    if(iter==1)
        Sigma_arr   = par.SigLam*par.nSig * ones(1,L); % First Iteration use the input noise parameter
    else
        temp        = sum(sum((Curpatch(:,:,Sel_arr)-Npatch(:,:,Sel_arr)).^2,1),2)/prod(sizePatch(1:2));
        Sigma_arr   = par.SigLam*sqrt(abs(par.nSig^2*ones(1,L)- temp(:)')); % estimate local noise variance
        parBSM.maxIter = max(parBSM.maxIter-10,15);
        clear Npatch Nmsi
    end
    if memorySaving == 0
        % %     block matching to find samilar FBP goups
        unfoldPatch     = (Unfold(Curpatch,sizePatch,3)');
        patchXpatch     = sum(unfoldPatch.^2,1);
        index     = zeros(par.patnum,L);
        sizePart  =  250; % This can be change according to memory
        numPart   =  floor(L/sizePart)+1;
        fprintf('Block matching of iter %f has been done          ', iter);
        fprintf('\n')
        for i = 1:numPart
            tempInd = (i-1)*sizePart+1:min(L,i*sizePart);
            distenMat       = bsxfun(@plus, patchXpatch(Sel_arr(tempInd)), patchXpatch')-2*(unfoldPatch')*unfoldPatch(:,Sel_arr(tempInd));
            [~,tempindex]       = sort(distenMat);
            index(:,tempInd)           = (tempindex(1:par.patnum,:));
            if i/numPart<0.1
                fprintf('\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            else
                fprintf('\b\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            end
        end
        fprintf('\n')
        clear patchXpatch distenMat unfoldPatch Emsi Curmsi tempInd ;
        Epatch          = zeros(sizePatch);
        W               = zeros(sizePatch(1),sizePatch(3));
        tempPatch = Curpatch(:,:,index(:));
        tempPatch = reshape(tempPatch, [sizePatch(1:2),par.patnum, L]);
        clear Curpatch;
        parfor i = 1:L
%             tempPatch(:,:,:,i) = ipermute(proxF_tSVD_1(permute(tempPatch(:,:,:,i),[1,3,2]),1/Sigma_arr(i),[]),[1,3,2]); % Perform ITS-based tensor recovery on each FBP goup
            tempPatch(:,:,:,i) = proxF_tSVD_1(tempPatch(:,:,:,i),1/Sigma_arr(i),[]); % Perform ITS-based tensor recovery on each FBP goup
        end
        for i = 1:L
            Epatch(:,:,index(:,i))  = Epatch(:,:,index(:,i)) + tempPatch(:,:,:,i);
            W(:,index(:,i))         = W(:,index(:,i))+ones(size(tempPatch(:,:,:,i),1),size(tempPatch(:,:,:,i),3));
        end
    elseif memorySaving == 1
        % %     block matching to find samilar FBP goups
        unfoldPatch     = (Unfold(Curpatch,sizePatch,3)');
        patchXpatch     = sum(unfoldPatch.^2,1);
        index     = zeros(par.patnum,L);
        sizePart  =  250; % This can be change according to memory
        numPart   =  floor(L/sizePart)+1;
        fprintf('Block matching of iter %f has been done          ', iter);
        fprintf('\n')
        for i = 1:numPart
            tempInd = (i-1)*sizePart+1:min(L,i*sizePart);
            distenMat       = bsxfun(@plus, patchXpatch(Sel_arr(tempInd)), patchXpatch')-2*(unfoldPatch')*unfoldPatch(:,Sel_arr(tempInd));
            [~,tempindex]       = sort(distenMat);
            index(:,tempInd)           = (tempindex(1:par.patnum,:));
            if i/numPart<0.1
                fprintf('\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            else
                fprintf('\b\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            end
        end
        fprintf('\n')
        clear patchXpatch distenMat unfoldPatch Emsi Curmsi tempInd ;
        Epatch          = zeros(sizePatch);
        W               = zeros(sizePatch(1),sizePatch(3),'single');
        
        fprintf('tSVD of iter %f has been done          ', iter);
        fprintf('\n')
        sizePart  =  1000;
        numPart   =  floor(L/sizePart)+1;
        for i = 1:numPart
            PattInd = (i-1)*sizePart+1:min(L,i*sizePart);
            tempInd = index(:,PattInd);
            sizeInd = size(tempInd);
            tempPatch = Curpatch(:,:,tempInd(:));
            tempPatch = reshape(tempPatch, [sizePatch(1:2), sizeInd]);
            tempSigma = Sigma_arr(PattInd);
            parfor j = 1:sizeInd(2)
                tempPatch(:,:,:,j) = proxF_tSVD_1(tempPatch(:,:,:,j),1/tempSigma(j),[]); % Perform ITS-based tensor recovery on each FBP goup
            end
            for j = 1:sizeInd(2)
                Epatch(:,:,tempInd(:,j))  = Epatch(:,:,tempInd(:,j)) + tempPatch(:,:,:,j);
                W(:,tempInd(:,j))         = W(:,tempInd(:,j))+ones(size(tempPatch(:,:,:,j),1),size(tempPatch(:,:,:,j),3));
            end
            if i/numPart<0.1
                fprintf('\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            else
                fprintf('\b\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            end
        end
        clear Curpatch;
        fprintf('\n')
        
    end
    clear tempPatch
    time = toc;
    [Emsi, ~]  =  Patch2Im3D( Epatch, W, par, sizeData); % recconstruct the estimated MSI by aggregating all reconstructed FBP goups.
    clear Epatch;
    if OutPSNR
        psnr       =  PSNR3D(Emsi*255,Omsi*255);
        disp(['Iter: ' num2str(iter),' , current PSNR = ' num2str(psnr), ',  already cost time: ', num2str(time)]);
        %         figure;imshow(Emsi(:,:,end));pause(0.5);
    else
        disp(['Iter: ' num2str(iter),'   done,  already cost time: ', num2str(time)]);
    end
end
end

function  [SelfIndex_arr]  =  nonLocal_arr(sizeD, par)
% -SelfIndex_arr is the index of keypatches in the total patch index array
TempR         =   sizeD(1)-par.patsize+1;
TempC         =   sizeD(2)-par.patsize+1;
R_GridIdx	  =   1:par.step:TempR;
R_GridIdx	  =   [R_GridIdx R_GridIdx(end)+1:TempR];
C_GridIdx	  =   1:par.step:TempC;
C_GridIdx	  =   [C_GridIdx C_GridIdx(end)+1:TempC];

temp          = 1:TempR*TempC;
temp          = reshape(temp,TempR,TempC);
SelfIndex_arr = temp(R_GridIdx,C_GridIdx);
SelfIndex_arr = SelfIndex_arr(:)';
end


function  [par,parBSM]=ParSetTSVD(nSig)
% parameters setting for ITS_DeNoising

parBSM.lambda     =   10;
parBSM.rho        =   1.2;
parBSM.mu         =   250;
meanD             =   0.2;
DimUpRank         =   [1,1,1];
par.nSig          =   nSig;                                 % Variance of the noise image
par.delta         =   0.1;                                  % Parameter between each iter
par.SearchWin     =   80;

if nSig <= 0.1
    par.patsize       =   6;                            % Patch size
    par.patnum        =   50;                           % Initial Non-local Patch number
%     par.SigLam        =   0.012*meanD*prod(DimUpRank);  % Noise estimete parameter
    par.SigLam        =   0.012*meanD*prod(DimUpRank);  % For Urban 
    parBSM.maxIter    =   25;                           % max iteration number for ITSReg tensor recovery
    par.deNoisingIter =   2;                            % total iteration numbers
elseif nSig <=0.15
    par.patsize       =   6;                            
    par.patnum        =   55;                           
    par.SigLam        =   0.005*meanD*prod(DimUpRank);
    parBSM.maxIter    =   25;
    par.deNoisingIter =   2;
elseif nSig <=0.2
    par.patsize       =   7;                            
    par.patnum        =   60;                           
    par.SigLam        =   0.00245*meanD*prod(DimUpRank);
    parBSM.maxIter    =   30;
    par.deNoisingIter =   2;
elseif nSig <=0.25
    par.patsize       =   7;                            
    par.patnum        =   65;                           
    par.SigLam        =   0.0015*meanD*prod(DimUpRank);
    parBSM.maxIter    =   30;
    par.deNoisingIter =   2;
else
    par.patsize       =   7;                           
    par.patnum        =   65;                          
    par.SigLam        =   0.0010*meanD*prod(DimUpRank);
    parBSM.maxIter    =   30;
    par.deNoisingIter =   2;
end
par.step          =   floor((par.patsize-1)); 
par.SigLam        =   par.SigLam*1500; 
end

