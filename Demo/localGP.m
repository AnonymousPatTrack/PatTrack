classdef localGP < handle
    %LOCALGP hold a set of GPMODEL instances for a local GPR
    %approach with a deterministic model choice from 'Model learning with
    %local gaussian process regression', Nguyen-Tuong et.al.
    
    properties
        n; % number of local models
        c; % centers of models
        models; % cell array of models
        distMeasure; % distance measure: 'euclid', 'se'
        hyper; % hyperparameter
        thresh; % threshold for creating a new model
    end
    
    methods
        function obj = localGP(X, Y, distMeasure, thresh)
            %LOCALGP Construct an instance of this class
            %   constructs instances of GPMODEL with x and y, estimates
            %   the hyper parameter
            obj.models = {};
            obj.n = 1;
            obj.distMeasure = distMeasure;
            obj.thresh = thresh;
            
            % hyperparameter estimation
            gpModel = fitrgp(X',Y,...
                'KernelFunction', 'ardsquaredexponential',...
                'FitMethod', 'exact',...
                'PredictMethod', 'exact',...
                'Standardize', false);
            hyp = gpModel.KernelInformation.KernelParameters;
            obj.hyper.sigmaL = hyp(1:end-1);
            obj.hyper.sigmaF = hyp(end);
            obj.hyper.sigmaN = gpModel.Sigma;
            
            obj.c = X(:,1);
            % init local models, start with mean and create a new model for
            % each point with distance > thresh
            Xmodel = {X(:,1)}; Ymodel = {Y(:,1)};
            for i=2:size(X,2)
                if any(obj.distance(X(:,i)) < obj.thresh) && strcmp(obj.distMeasure, 'euclid') || ...
                        any(obj.distance(X(:,i)) > obj.thresh) && strcmp(obj.distMeasure, 'se')
                    dist = obj.distance(X(:,i));
                    if strcmp(obj.distMeasure, 'euclid')
                        [~, near] = min(dist);
                    else
                        [~, near] = max(dist);
                    end
                    Xmodel{near} = [Xmodel{near}, X(:,i)];
                    Ymodel{near} = [Ymodel{near}, Y(:,i)];
                    obj.c(:,near) = mean(Xmodel{near},2);
                    
                else % create new model
                    obj.c(:,end+1) = X(:,i);
                    Xmodel{end+1} = X(:,i);
                    Ymodel{end+1} = Y(:,i);
                    obj.n = obj.n + 1;
                end
            end
            % init models with clustered data
            for i=1:obj.n
                obj.models{i} = GPMODEL(Xmodel{i},Ymodel{i},obj.hyper);
            end
        end
        
        function dist = distance(obj, x)
            %DISTANCE calculates the distance of x to all models
            
            if strcmp(obj.distMeasure, 'euclid')
                dist = zeros(obj.n,1);
                for i=1:obj.n
                    dist(i) = sqrt(sum((x - obj.c(:,i)).^2));
                end
            elseif strcmp(obj.distMeasure, 'se')
                dist = zeros(obj.n,1);
                for i=1:obj.n
                    dist(i) = exp(-0.5.* sum((x - obj.c(:,i)).^2 ./obj.hyper.sigmaL.^2 ));
                end
            else
                error('Unknown distance measure: Use euclid or se.');
            end
        end
        
        function update(obj, x, y)
            %UPDATE inserts the given data point into the nearest model and
            %updates it
            
            % check if distance is smaller than threshold
            if any(obj.distance(x) < obj.thresh) && strcmp(obj.distMeasure, 'euclid') || ...
                        any(obj.distance(x) > obj.thresh) && strcmp(obj.distMeasure, 'se')
                dist = obj.distance(x);
                if strcmp(obj.distMeasure, 'euclid')
                    [~, near] = min(dist);
                else
                    [~, near] = max(dist);
                end
                obj.models{near}.update(x,y);
                obj.c(:,near) = mean(obj.models{near}.X,2);
            else % create new model
                obj.c(:,end+1) = x;
                obj.models{end+1} = GPMODEL(x, y, obj.hyper);
                obj.n = obj.n + 1;
            end
        end
        
        function pred = predict(obj, x)
            %PREDICT predicts the output with all GPs as a weighted average
            dist = obj.distance(x);
            % calc predictions for all models
            predX = zeros(obj.n,1);
            for i=1:obj.n
                predX(i) = obj.models{i}.predict(x);
            end
            % calc the weighted prediction
            pred = sum(dist.*predX)./sum(dist);
        end
        
    end
end

