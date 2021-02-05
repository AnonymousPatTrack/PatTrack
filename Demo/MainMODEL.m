classdef MainMODEL < handle
    %GPMODEL is a representation of a Gaussian Process Regression
    
    properties
        X; % input training data
        Y; % targets (output training data)
        K; % Kernel(X,X)
        invK; % inverse of Kernel(X,X)
        hyper; % hyperparameters
        alpha; % prediction vector
    end
    
    methods
        function obj = MainMODEL(X, Y, hyper)
            %GPMODEL Construct an instance of this class
            %   fits the GP based on the data
            %   X:      input data [dim x #points]
            %   Y:      output data [1 x #points]
            %   hyper:  hyperparameters of gaussian process, [sigmaL,
            %           sigmaF, sigmaN]
            

            obj.hyper = hyper;
            obj.K= obj.kernel(X, X) + obj.hyper.sigmaN.^2 * eye(size(X,2));
            obj.invK = inv(obj.K);
            obj.alpha = obj.invK*Y';
        end
        
        function update(obj, x, X, updatedY)
            %UPDATE the prediction vector and model
            b = obj.kernel(X,x);
            c = obj.kernel(x,x) + obj.hyper.sigmaN.^2;
            obj.K = [obj.K b;...
                   b' c]; 
            obj.invK = obj.updateInv(b,c);
            obj.alpha = obj.invK * updatedY';
        end
        
        function inv = updateInv(obj, b, c)
            % UPDATEINV adds a row and column to the inverse of K
            k = c - b'*obj.invK*b;
            inv = [obj.invK + obj.invK*(b*b')*obj.invK/k  -obj.invK*b/k;...
                -b'*obj.invK/k                  1/k];
            
        end
        
        function K = getK(obj)
           K = obj.K;           
        end
        
        function cut(obj, Y, K)
            obj.K= K;
            obj.invK = inv(K);
            obj.alpha = obj.invK*Y';
        end
        
        function kern = kernel(obj, Xi, Xj)
            %KERNEL calculates the squared exponential kernel k(xi,xj)
            kern = zeros(size(Xi,2), size(Xj,2));
            for i=1:size(Xi,2)
                for j=1:size(Xj,2)
                    kern(i,j) = obj.hyper.sigmaF.^2 .*...
                        exp(-0.5 .* sum((Xi(:,i)-Xj(:,j)).^2 ...
                        ./ obj.hyper.sigmaL.^2 ));
                end
            end
        end
                
        function pred = predict(obj, x, X, sizey)
            %PREDICT predicts the output with the GP
            pred = zeros(size(x,2),sizey);
            for i = 1:size(x,2)
                if size(obj.alpha,1)==0
                    pred(i)=0;
                else
                    pred(i) = obj.kernel(X, x(:,i))' * obj.alpha;
                end
            end
        end
        
        
    end
end
