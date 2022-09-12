function [Re_hsi] = NMoG(noisy_hsi,Rank,param)
    [M, N, B] = size(noisy_hsi);
    Y = reshape(noisy_hsi,M*N,B);
    [~,Lr_model] = NMoG_LRMF(Y,Rank,param);
    U = Lr_model.U;
    V = Lr_model.V;
    Re_hsi = reshape(U*V',size(noisy_hsi));
    
end