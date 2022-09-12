function [Emsi,temp_res] = BCTF_DeNoising(Nmsi,nSig,Omsi,par)
    if nargin<3
        OutPSNR = 0;
    else
        OutPSNR = 1;
    end

    if nargin<2
        error('no input nSig')
    end

    [H, W, ~] = size(Nmsi);
    if exist('par', 'var')
        par = ParSet_New(nSig, par); % set denoising parameters and tensor recvery parameters;
    else
        par = ParSet_New(nSig);
    end
    [Neighbor_arr, Num_arr, Self_arr] =	NeighborIndex(H, W, par);
    Emsi = Nmsi;
    
    spmd
        warning('off','MATLAB:nearlySingularMatrix')
    end
    %% main loop
    tic    
    for  iter = 1 : par.deNoisingIter        
        Curmsi      	= Emsi + par.delta*(Nmsi - Emsi);
%         Curmsi          = Emsi;
        [Curpatch, Mat] = Im2Patch3D(Curmsi, par); % image to FBPs
        NL_mat  =  Block_matching(Mat, par.patnum, Neighbor_arr, Num_arr, Self_arr); % Caculate Non-local similar patches for each
        if par.verbose        
            fprintf('BCTF-HSI of iter %f has been done          ', iter);
            fprintf('\n')
        end
        [Epatch, W, maxRank] = NLPatEstimation(NL_mat, Self_arr, Curpatch, par);
        par.maxRank = maxRank;
        temp_res.maxRank(iter) = maxRank;
        
        [Emsi, ~] = Patch2Im3D( Epatch, W, par, size(Nmsi)); % recconstruct the estimated MSI by aggregating all reconstructed FBP goups.
        
        time = toc;
        
        if OutPSNR
            [psnr, ssim, ~, ~] = MSIQA(Emsi * 255, Omsi * 255);
            temp_res.psnr(iter) = psnr;
            temp_res.ssim(iter) = ssim;            
            disp(['Iter: ' num2str(iter),' , current PSNR = ' num2str(psnr), ' SSIM = ', num2str(ssim), ',  already cost time: ', num2str(time)]);
    %         figure;imshow(Emsi(:,:,end));pause(0.5);
        else
            disp(['Iter: ' num2str(iter),'   done,  already cost time: ', num2str(time)]);
        end
    end
    
    spmd
        warning('on','MATLAB:nearlySingularMatrix')
    end    
end

function [Epatch, W, maxRank] = NLPatEstimation(NL_mat, Self_arr, Curpatch, par)
    sizePatch       = size(Curpatch);
    Epatch          = zeros(sizePatch, 'single');
    W               = zeros(sizePatch(1), sizePatch(3), 'single');
    L               = length(Self_arr);    
    maxRank         = 0;
    %% Engineering techniques for parallel computing
    %% Naive version
%     for i = 1:L
%         Temp  = Curpatch(:,:,NL_mat(1:par.patnum,i)); % Non-local similar patches to the keypatch
%         model = BCPF(Temp, 'init', 'rand', 'maxRank', par.maxRank, 'dimRed', 1, 'tol', 1e-3, 'maxiters', par.maxIter, 'verbose', 0);        
%         Epatch(:,:,NL_mat(1:par.patnum,i))  = Epatch(:,:,NL_mat(1:par.patnum,i)) + single(double(model.X));
%         W(:,NL_mat(1:par.patnum,i)) = W(:,NL_mat(1:par.patnum,i)) + 1;
%     end
    %% Enable parallel computing
    sizePart  =  320;
    numPart   =  floor(L/sizePart)+1;
    
    for i = 1:numPart
        PattInd = (i-1)*sizePart+1:min(L,i*sizePart);
        tempInd = NL_mat(1:par.patnum,PattInd);
        sizeInd = size(tempInd);
        tempPatch = Curpatch(:,:,tempInd(:));
        tempPatch = reshape(tempPatch, [sizePatch(1:2), sizeInd]);
        
        TrueRank        = zeros(sizeInd(2),1);
        parfor j = 1:sizeInd(2)       
            model = BCPF(tempPatch(:,:,:,j), 'init', 'rand', 'maxRank', par.maxRank, 'dimRed', 1, 'tol', 1e-3, 'maxiters', par.maxIter, 'verbose', 0);        
            TrueRank(j) = model.TrueRank;
            tempPatch(:,:,:,j) = double(model.X);
        end
        maxRank = maxRank + sum(TrueRank(:));
        for j = 1:sizeInd(2)
            Epatch(:,:,tempInd(:,j))  = Epatch(:,:,tempInd(:,j)) + tempPatch(:,:,:,j);
            W(:,tempInd(:,j))         = W(:,tempInd(:,j))+1;
        end
        if par.verbose
            if i/numPart<0.1
                fprintf('\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            else
                fprintf('\b\b\b\b\b\b\b\b %2.2f%% ', i/numPart*100);
            end
        end
    end
    
    maxRank = ceil(maxRank / L);
    if par.verbose
        fprintf('\nmaxRank is set to %d\n', maxRank);
    end
end

function  [Neighbor_arr, Num_arr, SelfIndex_arr]  =  NeighborIndex(H, W, par)
    % This Function Precompute the all the patch indexes in the Searching window
    % -Neighbor_arr is the array of neighbor patch indexes for each keypatch
    % -Num_arr is array of the effective neighbor patch numbers for each keypatch
    % -SelfIndex_arr is the index of keypatches in the total patch index array 
    SW      	=   par.SearchWin;
    s           =   par.step;
    TempR       =   H-par.patsize+1;
    TempC       =   W-par.patsize+1;
    R_GridIdx	=   [1:s:TempR];
    R_GridIdx	=   [R_GridIdx R_GridIdx(end)+1:TempR];
    C_GridIdx	=   [1:s:TempC];
    C_GridIdx	=   [C_GridIdx C_GridIdx(end)+1:TempC];

    Idx         =   (1:TempR*TempC);
    Idx         =   reshape(Idx, TempR, TempC);
    R_GridH     =   length(R_GridIdx);    
    C_GridW     =   length(C_GridIdx); 

    Neighbor_arr    =   int32(zeros((2*SW+1)*(2*SW+1),R_GridH*C_GridW)); 
    Num_arr         =   int32(zeros(1,R_GridH*C_GridW)); % 1 x keypatch number
    SelfIndex_arr   =   int32(zeros(1,R_GridH*C_GridW));

    for  i  =  1 : R_GridH
        for  j  =  1 : C_GridW    
            OffsetR     =   R_GridIdx(i);
            OffsetC     =   C_GridIdx(j);
            Offset1  	=  (OffsetC-1)*TempR + OffsetR;  % total patch idx
            Offset2   	=  (j-1)*R_GridH + i;  % keypatch idx

            top         =   max( OffsetR-SW, 1 );
            button      =   min( OffsetR+SW, TempR );        
            left        =   max( OffsetC-SW, 1 );
            right       =   min( OffsetC+SW, TempC );     

            NL_Idx      =   Idx(top:button, left:right);
            NL_Idx      =   NL_Idx(:);  % neighbor idx

            Num_arr(Offset2)  =  length(NL_Idx);
            Neighbor_arr(1:Num_arr(Offset2),Offset2)  =  NL_Idx;  
            SelfIndex_arr(Offset2) = Offset1;  % map between total patch idx and keypatch idx
        end
    end
end

function  [Init_Index]  =  Block_matching(X, patnum, Neighbor_arr, Num_arr, SelfIndex_arr)
    L         =   length(Num_arr);
    Init_Index   =  zeros(patnum,L);

    for  i  =  1 : L
        Patch = X(:,SelfIndex_arr(i));
        Neighbors = X(:,Neighbor_arr(1:Num_arr(i),i));    
        Dist = sum((repmat(Patch,1,size(Neighbors,2))-Neighbors).^2);    
        [~, index] = sort(Dist);
        Init_Index(:,i)=Neighbor_arr(index(1:patnum),i);
    end
end

function  [par]=ParSet_New(nSig, par)
    par.verbose       =   true;
    par.maxIter       =   25;
    par.Pstep         =   1;
    par.nSig          =   nSig;                               % Variance of the noise image
    par.delta         =   0.05;
    if ~isfield(par, 'patsize')
        par.patsize       =   6;
    end
    if ~isfield(par, 'patnum')
        par.patnum        =   50;
    end    
    par.SearchWin     =   30;                                 % Non-local patch searching window
    
    if ~isfield(par, 'maxRank')
        par.maxRank       =   15;      
    end
    
    if nSig <= 20
        par.deNoisingIter =   1;                                % total iteration numbers
    elseif nSig <= 30
        par.deNoisingIter =   2;
    elseif nSig <= 60
        par.deNoisingIter =   3;
    elseif nSig <= 85
        par.deNoisingIter =   4;
    else
        par.deNoisingIter =   5;
    end

    par.step          =   floor((par.patsize-1)); 
end
