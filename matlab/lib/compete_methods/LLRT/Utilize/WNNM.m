
function  [X] =  WNNM( Y, C, NSig)
    [U,SigmaY,V] = svd(full(Y),'econ');    
    PatNum       = size(Y,2);
    TempC  = C*sqrt(PatNum)*2*NSig^2;
    [SigmaX,svp] = ClosedWNNM(SigmaY,TempC,eps); 
    X =  U(:,1:svp)*diag(SigmaX)*V(:,1:svp)';     
return;
