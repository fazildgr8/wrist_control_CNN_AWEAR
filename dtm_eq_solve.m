clc;
% syms l m n;
% 
% a = 0.15;
% b = 0.2;
% c = 0.96;
% theta = deg2rad(30);
% 
% 
% lv = -1.2;
% mv = 34.2;
% nv = -7.15;
% 
% 
% eq1 = mv*n - nv*m == sin(theta)*a;
% 
% eq2 = nv*l - lv*n == sin(theta)*b;
% 
% eq3 = lv*m - mv*l == sin(theta)*c;
% 
% eq4 = l*lv + m*mv + n*nv == cos(theta);
% 
% sol = solve([eq1, eq2, eq3, eq4],[l ,m n]);

syms x y z x1 y1 z1 x2 y2 z2 l1 m1 n1 l2 m2 n2;

