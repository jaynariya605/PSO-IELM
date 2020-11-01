function [S,MIN,Globalfitness,E]=PSO(Iteration,Input,target,NumberofTrainingData,NumberofInputNeurons,population,ub,lb)

n=NumberofInputNeurons;
P = population;

x=lb+(ub-lb)*rand(n,P);
b = rand(1,P);


for i = 1:P, % CALCULATING EVERY PARTICLE FITNESS
    InputWeight=x(:,i);
    BiasofHiddenNeurons=b(:,i);
    ind=ones(1,NumberofTrainingData);
    BiasMatrix=BiasofHiddenNeurons(:,ind);  
    H = 1 ./ (1 + exp(-(InputWeight'*Input+BiasMatrix)));
    E=target;
    Beta =pinv(H') * E';      
    E=E-(H' * Beta)'; 
    fitness(i)=sqrt(mse(E));
end

Globalfitness=fitness(1); % Define Golbale fitness
fgmin=fitness(1); % difine personal fitness
s=x+b;
personalS = s(:,1); % define personal solution
S = s(:,1); % define globle solution
for i = 1:P,
    if fitness(i)<Globalfitness, % up date globle finess if personal is better
        Globalfitness=fitness(i);
        S=s(:,i);
    end
end


MIN=[];

for k=1:ceil(Iteration/P),
    if k ==1,
        lbm = lb*ones(n+1,P);
        ubm = ub*ones(n+1,P);
    end
    for j = 1:n,
        for i = 1:P,
            s(j,i)=lbm(j,i)+(ubm(j,i)-lbm(j,i))*abs(sin(2/s(j,i)));
            if s(j,i)>1,
                s(j,i)=1;
            end
            if s(j,i)<-1,
                s(j,i)=-1;
            end
        end
    end
    for i = 1:P,
        InputWeight=x(:,i); % calculate fitness of each
        BiasofHiddenNeurons=b(:,i);
        ind=ones(1,NumberofTrainingData);
        BiasMatrix=BiasofHiddenNeurons(:,ind);  
        H = 1 ./ (1 + exp(-(InputWeight'*Input+BiasMatrix)));
        E=target;
        Beta =pinv(H') * E';      
        E=E-(H' * Beta)'; 
        personalfitness(i)=sqrt(mse(E));
    end
    
    for i = 1:P,
        if personalfitness(i)<fgmin,
            fgmin = personalfitness(i);% update personal fitness
            for j = 1:n+1,
                personalsolution=s(:,i); % update personal solution
            end
        end
    end
    
    
    
   for i = 1:P,
       if personalfitness(i)<fitness(i),
           fitness(i)= personalfitness(i);% update 
           for j = 1:n+1,
               personalS = s(:,i);
           end
       end
   end
   
   if fgmin< Globalfitness, % update globle fitness
       Globalfitness = fgmin;
       for j=1:n+1,
            S = s(:,i); % update globle solution
       end
   end
   for j=1:n,
       lbm(j,:)=personalS(j,:)-(ub-lb)*ones(1,P);
       ubm(j,:)=personalS(j,:)+(ub-lb)*ones(1,P);
   end
   
    for j=1:n+1,
        for i=1:P,
            if lbm(j,i)<lb,
                lbm(j,i)=lb; 
            end
            if ubm(j,i)>ub,   
                ubm(j,i)=ub; 
            end
        end
    end
    
    MIN=[MIN Globalfitness];
end
end
   
   
   
   
   
   
               
       
       
       
  
        
    
      
       

