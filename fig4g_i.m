clc;clear all;close all
load('Average');
load('7RSN_label')
load('gene_coexp')
load('neuroreceptor.mat');
N=360;

 D=pdist(GN,'cosine');
[Y,e] = cmdscale(D);
[a b]=sort(Y(:,1));
imagesc(log10(SC(b,b)))
caxis([-7,-0.3])
[den] = link_proporation(SC,GN,N,label,2);
figure
plot(Y(:,1),den,'o')
[den,Y(:,1)]

D=pdist(receptor,'cosine');
[z,e] = cmdscale(D);
[den] = link_proporation(SC,receptor,N,label,2);
[a b]=sort(-z(:,1));
figure
imagesc(log10(SC(b,b)))
caxis([-7,-0.3])
figure
plot(den,-z(:,1),'o')
[-z(:,1),den]

