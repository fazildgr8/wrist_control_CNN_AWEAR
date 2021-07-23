clc;

syms x1 x2 x y1 y2 y z1 z2 z a b c d d1 d2 h

plane_eq = a*x + b*y + c*z - d == 0;

% deq1 = sqrt((x1 - x)^2 + (y1 - y)^2 + (z1 - z)^2) - d1 == 0;
% 
% deq2 = sqrt((x2 - x)^2 + (y2 - y)^2 + (z2 - z)^2) - d1 == 0;

% deq3 = sqrt(x^2 + y^2 + z^2) - h == 0;


sol = solve([plane_eq,deq1,deq2,deq3],[x,y,z]);