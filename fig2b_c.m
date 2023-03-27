clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
%%%===========calculating distance between real and similated FC
for beta=[0.1:0.1:1.2]
    X=[];
    parfor sub=1:N_sub
        path=strcat('E:\Data\HCP\rest_FC\',Subj(sub),'_FC.mat');
        MRI=load(char(path));
        FC_real=MRI.FC;
        FC_real(FC_real<0)=0;
        path1=strcat('E:\Data\HCP\MRI\',Subj(sub),'.mat');
        MRI1=load(char(path1));
        SC=MRI1.DTI;
        SC(SC<2*10^-8)=0;
        H= Structural_network(SC.^beta,N);
        D=[];Z=[];
        for g=0:2:120
            [Q,FC] =Ideal_Predication_Model(H,g,N);
            D=[D,pdist2(FC_real(:)',FC(:)')/N];
        end
        X=[X;D];
    end
    path=strcat('beta_',num2str(beta),'.mat');
    D=X;
    save(char(path),'D')
end
%%%plot fig 2b
Z=[];X=[];
for beta=[0.8]
    path=strcat('beta_',num2str(beta),'.mat');
    load(char(path));
    d=mean(D);
    X=[X;mean(min(D')),std(min(D'))];
    Z=[Z;mean(D)];
end
imagesc(Z)
figure
g=0:2:120;
plot(g,Z,'-*')



