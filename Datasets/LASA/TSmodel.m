function [Yre,Ytr,d,sd,seq,patseq,predT,updateT] = TSmodel(Xst,k,ini,maxpoint,thr,thrnum,dim,nch,lsd)

X=Xst(:,1:end-k);
Y=Xst(:,1+k:end);
shi=0;
pat=1;
pats=1;
expat=0;
bdnew=0;
nchge=0;
sd=zeros(10,dim);
yshi=zeros(dim,thrnum);
dshi=zeros(1,thrnum);
d=zeros(1,size(X,2)-ini);
dmin=thr;
xinit_train = X(:,1:ini);
yinit_train = Y(:,1:ini);
x_stream = X(:,ini+1:end);
y_stream = Y(:,ini+1:end);
Yre = zeros(dim,size(x_stream,2));
Ytr = y_stream;
maxx=max(X,[],2);
minx=min(X,[],2);
width=maxx-minx+1;
center=(maxx+minx)./2;
seq=[];
patseq=[];
predT=[];
updateT=[];
for s=1:dim
    xinitr = xinit_train;
    yinitr = yinit_train(s,:);
    GPModel{pat*dim+s-dim} = onlineGP(xinitr, yinitr, maxpoint,width,center);
%    sd(pat,s) =  std(yinitr,1);
end
for i=1:size(x_stream,2)
    %disp(i);
    prt=0;
    upt=0;
    for s=1:dim
        xstr = x_stream(:,i);
        tic;
        Yre(s,i) =GPModel{pat*dim+s-dim}.prediv(xstr);
        prt=prt+toc;tic;
        GPModel{pat*dim+s-dim}.updatediv(xstr, y_stream(s,i));
        upt=upt+toc;
    end
    tic;
    if bdnew<thrnum %do not build new model
        nchge=nchge+1;
        if nchge>nch||(pat==1&&i>=lsd)
            for s=1:dim
                sd(pat,s) = std(y_stream(s,i-lsd+1:i),1);
            end
        d(i) = max(abs((Yre(:,i)-Ytr(:,i))./sd(pat,:)'));
        if d(i)>thr
            shi=shi+1;
            if shi==thrnum
                if pats>1
                    for pa=1:pats
%                        if pa~=pat
                            for r=1:thrnum
                                for s=1:dim
                                    xstr = x_stream(:,i-thrnum+r);
                                    yshi(s,r) =GPModel{s+dim*pa-dim}.prediv(xstr);
                                end
                                %之后把yshi改为一维的
                                dshi(r) = max(abs((Ytr(:,i-thrnum+r)-yshi(:,r))./sd(pat,:)'));
                                if r==thrnum&&max(dshi)<dmin
                                    expat=pa;
                                    dmin=max(dshi);
                                end
                            end
                            dshi=zeros(1,thrnum);
%                        end
                    end
                    if expat>0
                        pat=expat;
                        disp("change pat");
                        disp(i);
                        disp(pat);
                        seq=[seq i];
                        patseq=[patseq pat];
                    else
                        bdnew=thrnum;
                        disp("new pat");
                        disp(i);
                        disp(pats+1);
                        seq=[seq i];
                        patseq=[patseq pats+1];
                    end
                    expat=0;
                    dmin=thr;
                else
                    bdnew=thrnum;
                    disp("new pat");
                    disp(i);
                    disp(pats+1);
                    seq=[seq i];
                    patseq=[patseq pats+1];
                end
                shi=0;
            end
        else
            shi=0;
        end
        end
    else %build new model
        if bdnew<ini+2
            bdnew=bdnew+1;
        else
            pats=pats+1;
            pat=pats;
            for s=1:dim
                xistr(:,:) = x_stream(:,i-ini+1:i);
                yistr = y_stream(s,i-ini+1:i);
                GPModel{pat*dim+s-dim} = onlineGP(xistr, yistr, maxpoint,width,center);
%                sd(pat,s) = std(y_stream(s,i:i+ini),1);
            end
            nchge=0;%do not change pattern
            bdnew=0;
        end
    end
    upt=upt+toc;
    predT=[predT prt];
    updateT=[updateT upt];
end