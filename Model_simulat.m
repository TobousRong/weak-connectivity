clc;clear;close all
Subj =textread('H:\EÅÌ\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
X=[];Y=[];Z=[];K=[];
%%%==========================================calculating HB
% Z=[];Z1=[];
% load('model_distance')
% beta=0.1:0.1:1;
% g=0:5:1000;
% parfor sub=1:N_sub
%     c=fusion{sub};
%     [x,y]=find(c==min(min(c)));
%
%     path=strcat('H:\EÅÌ\weaklink\output\fusion\',Subj(sub),'_fusion_DTI_10M.mat');
%     MRI=load(char(path));
%     SC=MRI.SC;
%     [Q,FC] =Ideal_Predication_Model2(SC.^beta(y),g(x),N);
%     [Clus_num,Clus_size] = Functional_HP(FC,N);
%     [Hin,Hse,R_Hin,R_Hse,Hin_inter,Hse_inter] =Seg_Int_component(FC,N,Clus_size,Clus_num);
%     Z=[Z,Hin-Hse];
%
%     for p=[0.3,0.2,0.1]
%         [Q,FC] =Ideal_Predication_Model2(threshold_proportional(SC,p).^beta(y),g(x),N);
%         [Clus_num,Clus_size] = Functional_HP(FC,N);
%         [Hin,Hse,R_Hin,R_Hse,Hin_inter,Hse_inter] =Seg_Int_component(FC,N,Clus_size,Clus_num);
%         Z=[Z,Hin-Hse];
%     end
% end
% fusion=Z;
% save('model_HB.mat','fusion')
dis={};dis1={};dis2={};dis3={};
%%==========================calculate distance.
% parfor sub=1:N_sub
%     path=strcat('H:\EÅÌ\weaklink\output\FC\',Subj(sub),'.mat');
%     fMRI=load(char(path));
%     FC_real=fMRI.FC;
%     FC_real(FC_real<0)=0;
%
%     path=strcat('H:\EÅÌ\weaklink\output\fusion\',Subj(sub),'_fusion_DTI_10M.mat');
%     MRI=load(char(path));
%     fusion=MRI.SC;
%
%     path=strcat('H:\EÅÌ\Data\HCP\Matrix3_DTI\',Subj(sub),'_DTI_10M.mat');
%     MRI=load(char(path));
%     sift2=MRI.Glasser_sift2;
%
%     path=strcat('H:\EÅÌ\Data\HCP\commit2\',Subj(sub),'_commit2_lambda_0.05.mat');
%     MRI=load(char(path));
%     commit2=MRI.SC;
%
%     path=strcat('H:\EÅÌ\weaklink\output\sift2_threshold_commit2\',Subj(sub),'_sift2_threshold_commit2.mat');
%     MRI=load(char(path));
%     thres=MRI.SC;
%     C=[];C1=[];C2=[];C3=[];
%     for beta=0.1:0.1:1
%         for  g=1:10:700
%             [Q,FC] =Ideal_Predication_Model2(fusion.^beta,g,N);
%             %                 [h,p,ks2stat] = kstest2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')
%             C=[C;pdist2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')/N];
%             [Q,FC] =Ideal_Predication_Model2(thres.^beta,g,N);
%             %                 [h,p,ks2stat] = kstest2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')
%             C1=[C1;pdist2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')/N];
%
%             [Q,FC] =Ideal_Predication_Model2(commit2.^beta,g,N);
%             %                 [h,p,ks2stat] = kstest2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')
%             C2=[C2;pdist2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')/N];
%
%             [Q,FC] =Ideal_Predication_Model2(sift2.^beta,g,N);
%             %                 [h,p,ks2stat] = kstest2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')
%             C3=[C3;pdist2(vec_matrix(FC_real,N,2)',vec_matrix(FC,N,2)')/N];
%         end
%     end
%     ks=reshape(C,201,10);ks1=reshape(C1,201,10);ks2=reshape(C2,201,10);ks3=reshape(C3,201,10);
%     dis{sub}=ks;dis1{sub}=ks1;dis2{sub}=ks2;dis3{sub}=ks3;
% end
% fusion=dis;sift2_threshold=dis1;commit2=dis2;sift2=dis3;
% save('model_distance.mat','fusion','sift2_threshold','commit2','sift2')
%%%%%%%%%==================================pos/neg GN
load('H:\EÅÌ\weaklink\Data\7RSN_label.mat')
load('./Data/gene_coexp.mat');
Z1=[];Z=[];
parfor sub=1:N_sub
    path=strcat('H:\EÅÌ\weaklink\output\fusion\',Subj(sub),'_fusion_DTI_10M.mat');
    MRI=load(char(path));
    SC=MRI.SC;
%     W = threshold_proportional(SC, 0.1);
%     th=max(max(SC-W));
%     [P,P1,S,S1] = link_delete(SC,GN,N,th,label,1);
%     
    [Q,FC] =Ideal_Predication_Model2(SC.^0.2,180,N);
    [Clus_num,Clus_size] = Functional_HP(FC,N);
    [Hin,Hse,R_Hin,R_Hse,Hin_inter,Hse_inter] =Seg_Int_component(FC,N,Clus_size,Clus_num);
    Z=[Z,Hin-Hse];
    
%     [Q,FC] =Ideal_Predication_Model2(S1.^0.2,180,N);
%     [Clus_num,Clus_size] = Functional_HP(FC,N);
%     [Hin,Hse,R_Hin,R_Hse,Hin_inter,Hse_inter] =Seg_Int_component(FC,N,Clus_size,Clus_num);
%     Z1=[Z1,Hin-Hse];
end
Neg=Z;Pos=Z1;
% save('Neg_Pos_GN_HB.mat','Neg','Pos')
[h,p,~,stat]=ttest(Neg,Pos)

