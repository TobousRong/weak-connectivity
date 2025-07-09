clc;clear all;close all
Subj =textread('E:\Data\HCP\Subj_list.txt','%s');
N_sub=length(Subj);N=360;
%%%%=============================calculating active size
% parfor sub=1:N_sub
%     path=strcat('H:\EÅÌ\weaklink\output\fusion\',Subj(sub),'_fusion_DTI_10M.mat');
%     MRI=load(char(path));
%     SC=MRI.SC;
%     for beta=0.1:0.1:1;
%         activesize= active_size(SC.^beta,N,SC.^beta);
%         [a b]=sort(activesize,'descend');
%         activesize=activesize(b);
%         Z=[];
%         for th=0.1:0.1:0.4
%             W = threshold_proportional(SC, th);
%             activesize1= active_size(SC.^beta,N,W.^beta);
%             activesize1=activesize1(b);
%             Z=[Z;activesize1];
%         end
%         activesize=Z;
%         path=strcat('H:\EÅÌ\weaklink\output\activesize\',Subj(sub),'_activesize_beta_',num2str(beta),'.mat');
%         par_save(path,activesize)
%     end
% end
%%%%=============================calculating active size excluded Pos-GN
%%%%and Neg-GN
load('.\Data\7RSN_label')
load('.\Data\gene_coexp.mat');
Z=[];Z1=[];Z2=[];
parfor sub=1:N_sub
    path=strcat('H:\EÅÌ\weaklink\output\fusion\',Subj(sub),'_fusion_DTI_10M.mat');
    H=load(char(path));
    SC=H.SC;
    [low_order,high_order,angle] = func_cal_active_size(SC,GN,N,0.0435,label);
    Z=[Z;low_order];Z1=[Z1;high_order];Z2=[Z2;angle];
end
low_order=Z;high_order=Z1;angle=Z2;
save('H:\EÅÌ\weaklink\output\activesize\two_kinds_link_GN_equal_number.mat','low_order','high_order','angle')