[x1,x2] = textread('mocap.txt','%f%f',5674);
X = [x1';x2'];
k=50;
ini=200;
maxpoint=300;
thr=5.5;
thrnum=2;
dim=2;
nch=200;% no change for first nch-points of each pattern
lsd=100;
[Yre,Ytr,d,sd,seq,patseq,predT,updateT] = TSmodel(X,k,ini,maxpoint,thr,thrnum,dim,nch,lsd);
trendori=Ytr';
trendpre=Yre';
patdiv=[seq' patseq'];
T=[predT' updateT'];
save trendori.txt -ascii trendori;
save trendpre.txt -ascii trendpre;
save patdiv.txt -ascii patdiv;
save T.txt -ascii T;
for i=1:dim
    figure(i);
    plot(1:size(Yre,2)-k,Yre(i,1:end-k));
    hold on;
    plot(1:size(Yre,2)-k,Ytr(i,1:end-k));
end
figure(3);
plot(1:size(d,2),d);
