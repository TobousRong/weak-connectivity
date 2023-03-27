clc;clear all;close all
load('Average');
load('7RSN_label')
load('gene_coexp')
load('neuroreceptor.mat');
N=360;

 D=pdist(GN,'cosine');
[Y,e] = cmdscale(D);

D=pdist(receptor,'cosine');
[z,e] = cmdscale(D);
D=[];
for i=1:N-1
    for j=i+1:N
        D=[D;abs(Y(i,1)-Y(j,1)),abs(z(i,1)-z(j,1)),log10(SC(i,j)),receptor(i,j),GN(i,j)];
    end
end
n=find(D(:,3)==-Inf);
D(n,:)=[];
% for i=1:N
%     D=[D;length(find(SC(i,:)<0))/length(find(SC(i,:)~=0))];
% end
% plot(-Y(:,1),D,'o')
% n=find(D(:,2)==-Inf);
% D(n,:)=[];
% n=find(D(:,2)>-3);
% D(n,:)=[];
% X=zscore(D)
% n=find(D(:,3)<0);
% plot(D(n,2),D(n,1),'b.')
% hold on
% n=find(D(:,3)>0);
% plot(D(n,2),D(n,1),'r.')
% [h p]=corr(D)
%
% % SC=SC(b,b);
% X=zeros(18,18);
% for i=1:18
%     for j=1:18
%         X(i,j)=mean(mean(SC((i-1)*20+1:i*20,(j-1)*20+1:j*20)));
%     end
% end
%    Z=[];
%
% for i=1:18-1
%     for j=i+1:18
%         Z=[Z;i,j,X(i,j)];
%     end
% end
