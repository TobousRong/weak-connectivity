function [assignment,angle] = mode_alignment(SEC,SEC1,N)
cost=zeros(N,N);
for i=1:N
    for j=1:N
        if acos(SEC(:,i)'*SEC1(:,j))<acos(-SEC(:,i)'*SEC1(:,j))
            cost(i,j)=real(acos(SEC(:,i)'*SEC1(:,j)));
        else
            cost(i,j)=real(acos(-SEC(:,i)'*SEC1(:,j)));
        end
    end
end
[assignment, ~] = assignmentoptimal(cost);
angle=[];
for i=1:N
    angle=[angle;cost(i,assignment (i))];
end
end

