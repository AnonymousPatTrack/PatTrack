classdef onlineGP < handle
    
    properties
        X;
        Y;
        hyper; % hyperparameter
        width; % width of models
        numModels; %number of models in each dimensio
        maxpoint; %maximum number of points in a model
        npoint; % number of total point number
        Xmodel; % data sets of input
        Ymodel; % data sets of output
        model; % array of all models
        subcenter; %centers of submodels
        dim; % dimention of input
        Di; % Di(1,n) means in which dimention model n is divided
        numpoint; % points number for the first time
        broke; % 0 means this model has not been divided, 1 means it is divided
        sizey;% the dimension of y
        lr;%model on the left 1 or right 2
        parent;%the number of parent model
        child;%the number of child model
        n;%the number of all models
        modelID;
        pointID;
    end
    
    
    methods
        function obj = onlineGP(X, Y, maxpoint, width, center)
            cst=80000;
            obj.maxpoint = maxpoint;
            obj.dim = size(X,1);
            obj.Xmodel = cell(1,cst);
            obj.Ymodel = cell(1,cst);
            obj.pointID = cell(1,cst);
            obj.Xmodel{1,1} = X;
            obj.Ymodel{1,1} = Y;
            obj.npoint = zeros(1,cst);
            obj.npoint(1,1) = size(X,2);
            obj.pointID{1,1} = 1:obj.npoint(1,1);
            % hyperparameter estimation
            gpModel = fitrgp(X',Y,...
                'KernelFunction', 'ardsquaredexponential',...
                'FitMethod', 'exact',...
                'PredictMethod', 'exact');
            hyp = gpModel.KernelInformation.KernelParameters;
            obj.hyper.sigmaL = hyp(1:end-1);
            obj.hyper.sigmaF = hyp(end);
            obj.hyper.sigmaN = gpModel.Sigma;
            obj.subcenter = zeros(obj.dim,cst);
            obj.width = zeros(obj.dim,cst);
            obj.lr = zeros(1,cst);
            obj.Di = zeros(1,cst);
            obj.Di(1,1) = 1;
            obj.broke = zeros(1,cst);
            obj.sizey = 1;
            obj.parent=zeros(2,cst);
            obj.child=zeros(2,cst);
            obj.modelID = [];
            obj.modelID = ones(1,obj.npoint(1,1));
            obj.width(:,1) = width';%set according to the data set range
            obj.subcenter(:,1) = center';
            obj.n=1;
            if (obj.npoint(1,1) >= obj.maxpoint)
                obj.divide(1);
            else
                obj.model{1} = MainMODEL(X,Y,obj.hyper);
            end
        end
        
        function divide(obj,seq)
            m=obj.n;
            obj.parent(1,obj.n+1)=seq;
            obj.parent(1,obj.n+2)=seq;
            obj.child(1,seq)=obj.n+1;
            obj.child(2,seq)=obj.n+2;
            obj.n=obj.n+2;
            obj.div(seq,m);
            obj.assign(seq,m);
        end
        
        function updatedivide(obj,seq)
            m=obj.n;
            obj.parent(1,obj.n+1)=seq;
            obj.parent(1,obj.n+2)=seq;
            obj.child(1,seq)=obj.n+1;
            obj.child(2,seq)=obj.n+2;
            obj.n=obj.n+2;
            obj.div(seq,m);
            obj.updateassign(seq,m);
        end
        
        function div(obj,seq,m)
            I=obj.Di(1,seq);
            obj.Di(1,m+1)=I+1;
            if obj.Di(1,m+1)==obj.dim+1
                obj.Di(1,m+1)=1;
            end
            obj.Di(1,m+2)=obj.Di(1,m+1);
            for i=1:obj.dim
                if i==obj.Di(1,seq)
                    wi = obj.width(i,seq)/2;
                    obj.width(i,m+1)=wi;
                    obj.width(i,m+2)=wi;
                    obj.subcenter(i,m+1) = obj.subcenter(i,seq)-wi./2;
                    obj.subcenter(i,m+2) = obj.subcenter(i,seq)+wi./2;
                    obj.lr(1,m+1)=1;
                    obj.lr(1,m+2)=2;
                else
                    obj.width(i,m+1)=obj.width(i,seq);
                    obj.width(i,m+2)=obj.width(i,seq);
                    obj.subcenter(i,m+1) = obj.subcenter(i,seq);
                    obj.subcenter(i,m+2) = obj.subcenter(i,seq);
                end
            end
        end
        
        function assign(obj,seq,m)
            I = obj.Di(1,seq);
            sizeX=size(obj.Xmodel{1,seq},2);
            for i=1:sizeX
                if obj.active(m+1,obj.Xmodel{1,seq}(I,i)) == 1 % point is assigned to the left model
                    obj.npoint(1,m+1) = obj.npoint(1,m+1)+1;
                    obj.Xmodel{1,m+1}(:,obj.npoint(1,m+1)) = obj.Xmodel{1,seq}(:,i);
                    obj.Ymodel{1,m+1}(:,obj.npoint(1,m+1)) = obj.Ymodel{1,seq}(:,i);
                    pID = obj.pointID{1,seq}(:,i);
                    obj.pointID{1,m+1}(:,obj.npoint(1,m+1)) = pID;
                    obj.modelID(pID) = m+1;
                else  % point is assigned to the right model
                    obj.npoint(1,m+2) = obj.npoint(1,m+2)+1;
                    obj.Xmodel{1,m+2}(:,obj.npoint(1,m+2)) = obj.Xmodel{1,seq}(:,i);
                    obj.Ymodel{1,m+2}(:,obj.npoint(1,m+2)) = obj.Ymodel{1,seq}(:,i);
                    pID = obj.pointID{1,seq}(:,i);
                    obj.pointID{1,m+2}(:,obj.npoint(1,m+2)) = pID;
                    obj.modelID(pID) = m+2;
                end
            end
                if obj.npoint(1,m+1)<obj.maxpoint
                    obj.model{m+1} = MainMODEL(obj.Xmodel{1,m+1},obj.Ymodel{1,m+1},obj.hyper);
                else
                    obj.divide(m+1);
                end
                if obj.npoint(1,m+2)<obj.maxpoint
                    obj.model{m+2} = MainMODEL(obj.Xmodel{1,m+2},obj.Ymodel{1,m+2},obj.hyper);
                else
                    obj.divide(m+2);
                end
        end
        
        function updateassign(obj,seq,m)
            I = obj.Di(1,seq);
            sizeX=size(obj.Xmodel{1,seq},2);
            l=0;
            f=0;
            C = zeros(1,sizeX);
            D = zeros(1,sizeX);
            for i=1:sizeX
                if obj.active(m+1,obj.Xmodel{1,seq}(I,i)) == 1 % point is assigned to the left model
                    obj.npoint(1,m+1) = obj.npoint(1,m+1)+1;
                    obj.Xmodel{1,m+1}(:,obj.npoint(1,m+1)) = obj.Xmodel{1,seq}(:,i);
                    obj.Ymodel{1,m+1}(:,obj.npoint(1,m+1)) = obj.Ymodel{1,seq}(:,i);
                    pID = obj.pointID{1,seq}(:,i);
                    obj.pointID{1,m+1}(:,obj.npoint(1,m+1)) = pID;
                    obj.modelID(pID) = m+1;
                    l=l+1;
                    C(l) = i;
                else  % point is assigned to the right model
                    obj.npoint(1,m+2) = obj.npoint(1,m+2)+1;
                    obj.Xmodel{1,m+2}(:,obj.npoint(1,m+2)) = obj.Xmodel{1,seq}(:,i);
                    obj.Ymodel{1,m+2}(:,obj.npoint(1,m+2)) = obj.Ymodel{1,seq}(:,i);
                    pID = obj.pointID{1,seq}(:,i);
                    obj.pointID{1,m+2}(:,obj.npoint(1,m+2)) = pID;
                    obj.modelID(pID) = m+2;
                    f=f+1;
                    D(f) = i;
                end
            end
            B = 1:sizeX;
            C = [C(:,1:l) D(:,1:f)];
            K=obj.model{seq}.getK;
            K(B,:)=K(C,:);
            K(:,B)=K(:,C);
            KL = K(1:l,1:l);
            KF = K(l+1:l+f,l+1:l+f);
            if obj.npoint(1,m+1)<obj.maxpoint
                obj.model{m+1} = GPLMODEL(obj.Ymodel{1,m+1},obj.hyper,KL);
            else
                obj.divide(m+1);
            end
            if obj.npoint(1,m+2)<obj.maxpoint
                obj.model{m+2} = GPLMODEL(obj.Ymodel{1,m+2},obj.hyper,KF);
            else
                obj.divide(m+2);
            end
        end
        
        function pro = active(obj, seq, x)
            I=obj.Di(1,obj.parent(1,seq));
            c=obj.subcenter(I,seq);
            w=obj.width(I,seq);
            pro = ((x-c)<(w/2)).*((x-c)>=(-w/2));
            if (abs((x-c)-(w/2))<(1e-10))&&(randsrc==1)
                pro=1;
            end
        end
        
        function updatediv(obj, x, y)
            %UPDATEDIV inserts the given data point into a model
            obj.npoint(1,1) = obj.npoint(1,1)+1;
            obj.Ymodel{1,1} = [obj.Ymodel{1,1} y];
            pID = obj.pointID{1,1}(end)+1;
            obj.pointID{1,1} = [obj.pointID{1,1} pID];
            obj.modelID = [obj.modelID 1];
            if obj.npoint(1,1)<obj.maxpoint
                obj.model{1}.update(x,obj.Xmodel{1,1},obj.Ymodel{1,1});
                obj.Xmodel{1,1} = [obj.Xmodel{1,1} x];
            elseif obj.npoint(1,1)==obj.maxpoint
                obj.model{1}.update(x,obj.Xmodel{1,1},obj.Ymodel{1,1});
                obj.Xmodel{1,1} = [obj.Xmodel{1,1} x];
                obj.updatedivide(1);
            else
                obj.assignpoint(x,y,1,pID);
            end
        end
        
        function assignpoint(obj,x,y,seqfather,pID)
            seql=obj.child(1,seqfather);
            seqr=obj.child(2,seqfather);
            I = obj.Di(1,seqfather);
            if seqr==0
                pl=1;
            else
                pl= obj.active(seql,x(I,1));
            end
            if pl==1
                obj.npoint(1,seql) = obj.npoint(1,seql)+1;
                obj.Xmodel{1,seql}(:,obj.npoint(1,seql)) = x;
                obj.Ymodel{1,seql}(:,obj.npoint(1,seql)) = y;
                obj.pointID{1,seql}(:,obj.npoint(1,seql)) = pID;
                obj.modelID(pID) = seql;
                if seqr==0
                    obj.assignpoint(x,y,seql,pID);
                else
                    if (obj.child(1,seql)~=0)&&(obj.child(2,seql)==0)
                        obj.assignpoint(x,y,seql,pID);
                    else
                        if obj.npoint(1,seql)<obj.maxpoint
                            obj.model{seql}.update(x,obj.Xmodel{1,seql}(:,1:obj.npoint(1,seql)-1),obj.Ymodel{1,seql});
                        elseif obj.npoint(1,seql) == obj.maxpoint
                            obj.model{seql}.update(x,obj.Xmodel{1,seql}(:,1:obj.npoint(1,seql)-1),obj.Ymodel{1,seql});
                            obj.updatedivide(seql);
                        else
                            obj.assignpoint(x,y,seql,pID);
                        end
                    end
                end
            else
                obj.npoint(1,seqr) = obj.npoint(1,seqr)+1;
                obj.Xmodel{1,seqr}(:,obj.npoint(1,seqr)) = x;
                obj.Ymodel{1,seqr}(:,obj.npoint(1,seqr)) = y;
                obj.pointID{1,seqr}(:,obj.npoint(1,seqr)) = pID;
                obj.modelID(pID) = seqr;
                if obj.child(1,seqr)~=0&&obj.child(2,seqr)==0
                    obj.assignpoint(x,y,seqr,pID);
                else
                    if obj.npoint(1,seqr)<obj.maxpoint
                        obj.model{seqr}.update(x,obj.Xmodel{1,seqr}(:,1:obj.npoint(1,seqr)-1),obj.Ymodel{1,seqr});
                    elseif obj.npoint(1,seqr) == obj.maxpoint
                        obj.model{seqr}.update(x,obj.Xmodel{1,seqr}(:,1:obj.npoint(1,seqr)-1),obj.Ymodel{1,seqr});
                        obj.updatedivide(seqr);
                    else
                        obj.assignpoint(x,y,seqr,pID);
                    end
                end
            end
        end
        
        function delete(obj, pID)
            moseq = obj.modelID(pID);
            locseq = find(obj.pointID{1,moseq}==pID);
            l = obj.npoint(1,moseq);
            C = 1:l;
            C(locseq) = [];
            K=obj.model{moseq}.getK;
            B = 1:size(K,2);
            C = [C locseq];
            K(B,:)=K(C,:);
            K(:,B)=K(:,C);
            KL = K(1:l-1,1:l-1);
            obj.Xmodel{1,moseq}(:,locseq) = [];
            obj.Ymodel{1,moseq}(:,locseq) = [];
            obj.model{moseq}.cut(obj.Ymodel{1,moseq},KL);
            obj.npoint(1,moseq) = obj.npoint(1,moseq)-1;
            obj.pointID{1,moseq}(locseq) = [];
        end
        
        function pred = prediv(obj,x)
            if obj.npoint(1,1)>=obj.maxpoint
                pred = obj.pre(x,1);
            else
                pred = obj.model{1}.predict(x,obj.Xmodel{1,1},obj.sizey);
            end
        end
        
        function pred=pre(obj,x,seq)
            I = obj.Di(1,seq);
            seqlr=obj.child(1,seq);
            prob=obj.active(seqlr,x(I,1));
            if prob==0
                seqlr=obj.child(2,seq);
            end
            if obj.child(1,seqlr)==0&&obj.npoint(1,seqlr)<obj.maxpoint
                pred = obj.model{seqlr}.predict(x,obj.Xmodel{1,seqlr},obj.sizey);
            else
                pred = obj.pre(x,seqlr);
            end
        end
    end
end