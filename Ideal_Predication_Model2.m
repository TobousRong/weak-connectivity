function [Q,C] =Ideal_Predication_Model2(SC,g,N)
H= SC/max(eig(SC));
% Q=(eye(N)-g*H)^-1;
Q=exp(g*H);
Q=Q*Q';
for i=1:N
    for j=1:N
        C(i,j)=Q(i,j)/sqrt(Q(i,i)*Q(j,j));
    end
end
end

