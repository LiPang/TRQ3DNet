function  [Y, Mat]  =  Im2Patch3D( Video, par)
% get full band patches
patsize     = par.patsize;
if isfield(par,'Pstep')
    step   = par.Pstep;
else
    step   = 1;
end

TotalPatNum = (floor((size(Video,1)-patsize)/step)+1)*(floor((size(Video,2)-patsize)/step)+1);                  %Total Patch Number in the image
Mat         =   zeros(par.patsize*par.patsize, TotalPatNum, 'single');
Y           =   zeros(patsize*patsize, size(Video,3), TotalPatNum, 'single');                                   %Patches in the original noisy image
k           =   0;

for i  = 1:patsize
    for j  = 1:patsize
        k     =  k+1;
        tempPatch     =  Video(i:step:end-patsize+i,j:step:end-patsize+j,:);
        Y(k,:,:)      =  Unfold(tempPatch, size(tempPatch), 3);
        Mat_patch   =  mean(tempPatch,3);
        Mat(k,:)    =  Mat_patch(:)';
    end
end
