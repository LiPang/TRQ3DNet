function [varargout]=hysime(varargin);
%
% HySime: Hyperspectral signal subspace estimation
%
% [kf,Ek]=hysime(y,w,Rw);
%
% Input:
%        y：  L x N 高光谱数据集中（每列是一个像元）,L是波段数，N是像元数
%        w：  (L x N) 噪声
%        Rw： 噪声相关系数矩阵 (L x L)
% Output
%        kf signal subspace dimension信号子空间维数，即端元数目
%        Ek matrix which columns are the eigenvectors that span 
%           the signal subspace 由特征向量组成的矩阵，每一列是一个特征向量，这些特征向量的线性组合可以用来表示信号，因此Ek矩阵构成了信号子空间
%

error(nargchk(3, 3, nargin))  %参数检测！输入参数个数一定要为3
if nargout > 2, error('too many output parameters'); end  %输出参数个数不大于2
y = varargin{1}; % y=输入参数列表的第一个参数
[L N] = size(y);  
if ~numel(y),error('the data set is empty');end
n = varargin{2}; % n=输入参数列表的第二个参数 噪声
[Ln Nn] = size(n);  
Rn = varargin{3}; % Rn=输入参数列表的第三个参数 噪声相关系数矩阵
[d1 d2] = size(Rn);

if Ln~=L | Nn~=N,  % n is an empty matrix or with different size  检查 y与 n两个矩阵大小是否相同
   error('empty noise matrix or its size does not agree with size of y\n'),
end
if (d1~=d2 | d1~=L)
   fprintf('Bad noise correlation matrix\n'),
   Rn = n*n'/N; 
end    


x = y - n;  %信号=观测数据-噪声


[L N]=size(y);
Ry = y*y'/N;   % sample correlation matrix 观测数据相关系数矩阵
Rx = x*x'/N;   % signal correlation matrix estimates 信号相关系数矩阵
%计算信号相关系数矩阵Rx的特征向量
[E,D]=svd(Rx); % eigen values of Rx in decreasing order, equation (15)
dx = diag(D);


Rn=Rn+sum(diag(Rx))/L/10^5*eye(L);


Py = diag(E'*Ry*E); %equation (23) 
Pn = diag(E'*Rn*E); %equation (24)
cost_F = -Py + 2 * Pn; %equation (22) 
%估计端元数目 kf
kf = sum(cost_F<0);    %cost_F中元素为负数的数量
[dummy,ind_asc] = sort( cost_F ,'ascend');  %对cost_F做升序排列，得到新的向量dummy，新向量中的每个元素在原始向量中的位置保存在ind_asc
Ek = E(:,ind_asc(1:kf));  %cost_F中为负数的元素对应的特征向量构成子空间 Ek
ind_asc(1:kf)


    
varargout(1) = {kf}; %输出参数列表的第一个参数=端元数目 kf
if nargout == 2, varargout(2) = {Ek};end  %输出参数列表的第二个参数=表示信号子空间的矩阵 Ek
return
%end of function [varargout]=hysime(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

