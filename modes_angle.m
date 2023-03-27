clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
X=[];
for th=0.1:0.1:0.9
    Z=[];
    parfor sub=1:N_sub
        path=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
        MRI=load(char(path));
        SC=MRI.DTI;
        SC(SC<2*10^-8)=0;
        H= Structural_network(SC.^0.8,N);
        [SEC SE]=eig(H);
        
        SC= threshold_proportional(SC, th);
        H= Structural_network(SC.^0.8,N);
        [SEC1 SE1]=eig(H);
        [assigment,angle] = mode_alignment(SEC1,SEC,N);
        Z=[Z;mean(angle)];
    end
    X=[X,Z];
end
[mean(X)',std(X)']