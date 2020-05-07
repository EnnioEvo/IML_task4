A = readmatrix('../data/graph.csv');


S = A(:,1)'+1; %+1 because 0 is not index of a matrix
D = A(:,2)'+1;
W = A(:,3)';
DG = sparse(S,D,W);
% tril returns the lower triangular part of the matrix.
UG = tril(DG+DG');
[dist,path,pred] = graphshortestpath(UG,1,6,'Directed',false);