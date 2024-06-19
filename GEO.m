

function [Leader_pos,Leader_score,ConvergenceCurve] = GEO(fobj,trdata,trlabel,tstdata,tstlabel)

lb = 0;
ub =  10;
dim = 50;
nvars=5;
options.PopulationSize =1;
options1.PopulationSize = (options.PopulationSize);
SearchAgents_no=options.PopulationSize;
options.MaxIterations =10;
options1.MaxIterations=options.MaxIterations;
Max_iteration=options.MaxIterations ;
SearchAgents_no1=SearchAgents_no;
para=[Max_iteration SearchAgents_no];
%% initialization
PopulationSize = options.PopulationSize;
MaxIterations = options.MaxIterations;
options.AttackPropensity = [0.5 ,   2];
options.CruisePropensity = [1   , 0.5];
options1.AttackPropensity = options.AttackPropensity;
options1.CruisePropensity = options.CruisePropensity;
Lb= lb.*ones(1,nvars);    % Lower limit/bounds/     a vector
Ub= ub.*ones(1,nvars);    % Upper limit/bounds/     a vector

Leader_score=inf;
Leader_pos=zeros(1,nvars);

ConvergenceCurve  = zeros (1, MaxIterations);

x = max(round(lb + randi([1,nvars]) .* (ub-lb)));
x;
FitnessScores = fobj(x,trdata,trlabel);


% solver-specific initialization
FlockMemoryF = FitnessScores;
FlockMemoryX = x;

for i=1:PopulationSize
    if FitnessScores(i)<FlockMemoryF(i)
       FlockMemoryX(i,:)= x(i,:);
       FlockMemoryF(i)=FitnessScores(i);
    end
    if FitnessScores(i)<Leader_score
        Leader_score=FitnessScores(i);
        Leader_pos=x(i,:);
    end
end


AttackPropensity = linspace (options.AttackPropensity(1), options.AttackPropensity(2), MaxIterations);
CruisePropensity = linspace (options.CruisePropensity(1), options.CruisePropensity(2), MaxIterations);

%% main loop




for CurrentIteration=1:MaxIterations
    
	% prey selection (one-to-one mapping)
	DestinationEagle = randperm (PopulationSize)';
% 	% calculate AttackVectorInitial (Eq. 1 in paper)
	AttackVectorInitial = FlockMemoryX (DestinationEagle,:) - x;
  

	% calculate Radius
	Radius = norm (AttackVectorInitial, 2);
	
	% determine converged and unconverged eagles
	ConvergedEagles = sum (Radius,2) == 0;
	UnconvergedEagles = ~ConvergedEagles;
	
	% initialize CruiseVectorInitial
	CruiseVectorInitial = 2 .* rand(PopulationSize, nvars) - 1; % [-1,1]
	
	% correct vectors for converged eagles
	AttackVectorInitial (ConvergedEagles,:) = 0;
	CruiseVectorInitial (ConvergedEagles,:) = 0;
	
	% determine constrained and free variables
	for i1 = 1:PopulationSize
		if UnconvergedEagles(i1)
			vConstrained = false([1, nvars]); % mask
			idx = datasample(find(AttackVectorInitial(i1,:)), 1, 2);
			vConstrained(idx) = 1;
			vFree = ~vConstrained;
			CruiseVectorInitial(i1,idx) = - sum(AttackVectorInitial(i1,vFree).*CruiseVectorInitial(i1,vFree), 2) ./ (AttackVectorInitial(i1,vConstrained)); % (Eq. 4 in paper)
		end
	end
	
	% calculate unit vectors
	AttackVectorUnit = AttackVectorInitial ./ norm(AttackVectorInitial,2);
	CruiseVectorUnit = CruiseVectorInitial ./ norm(CruiseVectorInitial, 2);
	
	% correct vectors for converged eagles
	AttackVectorUnit(ConvergedEagles,:) = 0;
	CruiseVectorUnit(ConvergedEagles,:) = 0;
	
	% calculate movement vectors
	AttackVector = randi([1,PopulationSize]) .* AttackPropensity(CurrentIteration) .* Radius .* AttackVectorUnit; % (first term of Eq. 6 in paper)
	CruiseVector = randi([1,PopulationSize]) .* CruisePropensity(CurrentIteration) .* Radius .* CruiseVectorUnit; % (second term of Eq. 6 in paper)
	StepVector = AttackVector + CruiseVector;
	
	% calculate new x
	x = x + StepVector;
	
   
    for i=1:PopulationSize
      x(i,:)=round(simple(x(i,:),Lb,Ub));
      FitnessScores(i)= fobj((x(i)),trdata,trlabel);
    end
	
	% calculate fitness
         
    for i=1:PopulationSize
         if FitnessScores(i)<FlockMemoryF(i)
             FlockMemoryX(i,:)=x(i,:);
             FlockMemoryF(i)=FitnessScores(i);
         end
         
        if  FitnessScores(i)<Leader_score
            Leader_score= FitnessScores(i);
            Leader_pos=x(i,:);
        end
     end

	% update convergence curve
    ConvergenceCurve (CurrentIteration) = Leader_score;

    
    pTemp=x;        
    pTemp2=x;
    for j=1:nvars
        for i=1:PopulationSize
            pTemp2(i,j)=pTemp(i,j)+abs(min(pTemp(:,j)));
        end      
        if max(pTemp2(:,j))>0
            pTemp2(:,j)=pTemp2(:,j)/max(pTemp2(:,j));          
        else
            pTemp2(:,j)=0;
        end           
    end

    
    %%%%%%%%%%%%%%%%%%%%%%%% mean %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% %     for i=1:PopulationSize
% %         % Population diversity of swarm individuals
% %         med=0;
% %         for j=1:nvars
% %             med= med + abs(median(pTemp2(:,j))-pTemp2(i,j));
% %         end
% %     end        
% % 
% %     %%%%%%%%%%%%%%%%% Population diversity as a whole
% %     temp=[];
% %     for j=1:nvars
% %         temp(j)=mean(abs(pTemp2(:,j)-mean(pTemp2(:,j))));
% %     end
% %     Div((CurrentIteration-1)*PopulationSize+1:MaxIterations*PopulationSize)= sum(temp)/nvars;
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%% median %%%%%%%%%%%%%%%%%%%%%%%%%%%%
     for j=1:nvars
             med=0;
         for i=1:PopulationSize
            med= med + abs(median(pTemp2(:,j))-pTemp2(i,j));
         end
        Dev(j)=med/PopulationSize;
    end        

      temp=[];
    for j=1:nvars
        temp(j)=Dev(j);
    end
    Div(CurrentIteration)= sum(temp)/nvars;
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    Xpl2 = ((Div/max(Div))*100);
    Xpt2 =((abs((Div-max(Div)))/max(Div))*100);
end

    Xpl = sum((Div/max(Div))*100);
    Xpt =sum((abs((Div-max(Div)))/max(Div))*100);

%% return values
display(['The best solution obtained by GEO is : ', num2str(Leader_pos,7)]);
%display(['The best optimal value of the objective funciton found by GEO is : ', num2str(Leader_score,7)]);
display(['Population diversity: ', num2str(mean(Div))])
%display(['Exploration-exploitation percentage ratio: ', num2str((Xpl/(Xpl+Xpt))*100),':',num2str((Xpt/(Xpl+Xpt))*100)])
disp(fprintf('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'));  
end


function s = simple( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
end
