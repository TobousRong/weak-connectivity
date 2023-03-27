clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
load('7RSN_label.mat')
X=[];X1=[];X2=[];
parfor sub=1:N_sub
    path=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
    MRI=load(char(path));
    SC=MRI.DTI;
    SC(SC<2*10^-8)=0;
    P=participation_coef(SC,label);
    X=[X; link_proporation(SC,N,label)];
    SC= threshold_proportional(SC, 0.3);
    X1=[X1;link_proporation(SC,N,label)];
    SC= threshold_proportional(SC, 0.1);
    X2=[X2;link_proporation(SC,N,label)];
end
mean(mean(X))
mean(mean(X1))
mean(mean(X2))
[mean(X);mean(X1);mean(X2)]