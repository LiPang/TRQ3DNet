function [varargout]=hysime(varargin);
%
% HySime: Hyperspectral signal subspace estimation
%
% [kf,Ek]=hysime(y,w,Rw);
%
% Input:
%        y��  L x N �߹������ݼ��У�ÿ����һ����Ԫ��,L�ǲ�������N����Ԫ��
%        w��  (L x N) ����
%        Rw�� �������ϵ������ (L x L)
% Output
%        kf signal subspace dimension�ź��ӿռ�ά��������Ԫ��Ŀ
%        Ek matrix which columns are the eigenvectors that span 
%           the signal subspace ������������ɵľ���ÿһ����һ��������������Щ����������������Ͽ���������ʾ�źţ����Ek���󹹳����ź��ӿռ�
%

error(nargchk(3, 3, nargin))  %������⣡�����������һ��ҪΪ3
if nargout > 2, error('too many output parameters'); end  %�����������������2
y = varargin{1}; % y=��������б�ĵ�һ������
[L N] = size(y);  
if ~numel(y),error('the data set is empty');end
n = varargin{2}; % n=��������б�ĵڶ������� ����
[Ln Nn] = size(n);  
Rn = varargin{3}; % Rn=��������б�ĵ��������� �������ϵ������
[d1 d2] = size(Rn);

if Ln~=L | Nn~=N,  % n is an empty matrix or with different size  ��� y�� n���������С�Ƿ���ͬ
   error('empty noise matrix or its size does not agree with size of y\n'),
end
if (d1~=d2 | d1~=L)
   fprintf('Bad noise correlation matrix\n'),
   Rn = n*n'/N; 
end    


x = y - n;  %�ź�=�۲�����-����


[L N]=size(y);
Ry = y*y'/N;   % sample correlation matrix �۲��������ϵ������
Rx = x*x'/N;   % signal correlation matrix estimates �ź����ϵ������
%�����ź����ϵ������Rx����������
[E,D]=svd(Rx); % eigen values of Rx in decreasing order, equation (15)
dx = diag(D);


Rn=Rn+sum(diag(Rx))/L/10^5*eye(L);


Py = diag(E'*Ry*E); %equation (23) 
Pn = diag(E'*Rn*E); %equation (24)
cost_F = -Py + 2 * Pn; %equation (22) 
%���ƶ�Ԫ��Ŀ kf
kf = sum(cost_F<0);    %cost_F��Ԫ��Ϊ����������
[dummy,ind_asc] = sort( cost_F ,'ascend');  %��cost_F���������У��õ��µ�����dummy���������е�ÿ��Ԫ����ԭʼ�����е�λ�ñ�����ind_asc
Ek = E(:,ind_asc(1:kf));  %cost_F��Ϊ������Ԫ�ض�Ӧ���������������ӿռ� Ek
ind_asc(1:kf)


    
varargout(1) = {kf}; %��������б�ĵ�һ������=��Ԫ��Ŀ kf
if nargout == 2, varargout(2) = {Ek};end  %��������б�ĵڶ�������=��ʾ�ź��ӿռ�ľ��� Ek
return
%end of function [varargout]=hysime(varargin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

