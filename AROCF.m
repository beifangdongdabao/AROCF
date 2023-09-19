function [H,U,OBJ] = AROCF(X,H,U,L,lmd)
OBJ = [];
for i=1:10
    % update H
    a = max(max(L)) * lmd;
    A = (ceil(a)+1) * eye(size(L)) - lmd * L;
    B = X' * X * U;
    C = 2 * B + 2 * A * H;
    [UU,TT,WW] = svd(C,'econ');
    H = UU * WW;
    % update U
    U = H;
    obj = norm(X - X * U * H','fro')^2 + lmd * trace(H'*L*H);
    OBJ = [OBJ,obj];
end
end