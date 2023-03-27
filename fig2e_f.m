clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
D=[];
for p=0.1:0.1:0.9
    X=[];
parfor sub=1:N_sub
    path1=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
    MRI1=load(char(path1));
    SC=MRI1.DTI;
    SC(SC<2*10^-8)=0;
    W = threshold_proportional(SC, p);
    H= Structural_network(W.^0.8,N);
    Z=[];
    for g=50
        [Q,FC] =Ideal_Predication_Model(H,g,N);
        [Clus_num,Clus_size] = Functional_HP(FC,N);
        [Hin,Hse,R_Hin,R_Hse,Hin_inter,Hse_inter] =Seg_Int_component(FC,N,Clus_size,Clus_num);
        HB=Hin-Hse;
        Z=[Z,HB];
    end
    X=[X;Z];
end
D=[D,X];
end
HB=D;
save('HB_beta08_thre.mat','HB')
