function e = ymodulus(grade, temp)
% temp in centigrade
temp = round((temp + 273.15)*100)/100;

E = [
% Elastic constant versus temperature behavior of three hardened maraging steels
% H.M. Ledbettera and M.W. Austina
% Materials Science and Engineering
% Volume 72, Issue 1, June 1985, Pages 65-69

%T(K) E(x10^11 N/m/m)
%     200   250   300
75  1.938 1.976 1.990
100 1.935 1.971 1.986
150 1.920 1.956 1.971
200 1.900 1.935 1.949
250 1.875 1.910 1.925
300 1.849 1.883 1.898
350 1.822 1.855 1.870
400 1.794 1.826 1.841
];

if temp < E(1,1) || temp > E(end,1)
    error('invalid temperature')
end

switch grade
    case 200
        idx3 = 2;
    case 250
        idx3 = 3;
    case 300
        idx3 = 4;
    otherwise
        error('invalid grade')
end

idx2 = find(E(:,1) - temp > 0, 1 );
idx1 = idx2 - 1;
xx=[E(idx1,1),E(idx2,1)];
yy=[E(idx1,idx3),E(idx2,idx3)];
tt=E(idx1,1):0.001:E(idx2,1);
%e = interp1(E(idx1,1),E(idx2,1),E(idx1,idx3),E(idx2,idx3),temp);
etemp = interp1(xx,yy,tt);
e=etemp(find(tt==temp));
e = e*1e11;

