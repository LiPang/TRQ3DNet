function C = TensorTimes(A,B)
% ���ټ��� C����������i�� = A(:,:,i)*B(:,:,i);
% ����ͬ��С�ľ���ͬʱ�˻�������������ÿ��������˻����Ȱ�����������ά���������sum����ά
A = permute(A,[2,4,3,1]);
C = bsxfun(@times, A,B) ;
C = permute(sum(C,1),[4,2,3,1]);
end