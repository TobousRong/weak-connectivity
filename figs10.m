clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
load('7RSN_label')
load('gene_coexp')
load('neuroreceptor.mat');

Z=[];Z1=[];Z2=[];
for sub=1:N_sub
    path=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
    MRI=load(char(path));
    SC=MRI.DTI;
    SC(SC<2*10^-8)=0;
    Z=[Z,participation_coef(SC.^0.2,label)];
    Z1=[Z1,participation_coef(SC.^0.8,label)];
    Z2=[Z2,participation_coef(SC.^1,label)];
end
X=[];
for i=1:7
    n=find(label==i);
    X=[X;mean(mean(Z(n,:))),std(mean(Z(n,:))),mean(mean(Z1(n,:))),std(mean(Z1(n,:))),mean(mean(Z2(n,:))),std(mean(Z2(n,:)))];
end
