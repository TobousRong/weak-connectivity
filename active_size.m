function [activesize] = active_size(SC,N,SC1)
H= Structural_network(SC,N);
[SEC SE]=eig(H);
d=repmat(std(SEC.^2),N,1);
H1= Structural_network(SC1,N);
[SEC1 SE1]=eig(H1);
[assigment,angle] = mode_alignment(SEC1,SEC,N);
SEC1=SEC1(:,assigment).^2;
X=SEC1-d;
X(X<=0)=0;
X(X>0)=1;
activesize=mean(X);
end

