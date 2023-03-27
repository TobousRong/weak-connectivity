clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
X=[];
for sub=1:N_sub
        path=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
        MRI=load(char(path));
        SC=MRI.DTI;
        SC(SC<2*10^-8)=0;
        H= Structural_network(SC.^0.8,N);
        [SEC SE]=eig(H);
        SEC=SEC.^2;
        X=[X;std(SEC)];
end
[mean(X)',std(X)']