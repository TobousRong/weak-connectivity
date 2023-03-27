clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
Th=0.1:0.1:0.9;
for sub=1:N_sub
    path=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
    MRI=load(char(path));
    SC=MRI.DTI;
    SC(SC<2*10^-8)=0;
    Z=[];
    parfor th=1:length(Th)
        SC1= threshold_proportional(SC, Th(th));
        activesize= active_size(SC.^1,N,SC1.^1);
        Z=[Z;activesize];
    end
    activesize=Z;
    path=strcat('E:\weaklink\output\activesize\',Subj(sub),'_activesize_beta10.mat');
    save(char(path),'activesize')
end

%%===================
Z1=[];Z2=[];
for sub=1:N_sub
    path=strcat('E:\weaklink\output\activesize\',Subj(sub),'_activesize_beta10.mat');
    load(char(path));
    [a b]=sort(activesize(end,:),'descend');
    activesize=activesize(:,b);
    Z1=[Z1;activesize(1,:)];
    Z2=[Z2;activesize(9,:)];
%     for i=1:9
%         Z1=[Z1;mean(activesize(i,2:180))./mean(activesize(end,2:180))];
%         Z2=[Z2;mean(activesize(i,180+1:end)')./mean(activesize(end,180+1:end)')];
%     end
end
%  Z1=reshape(Z1,9,991);Z2=reshape(Z2,9,991);
