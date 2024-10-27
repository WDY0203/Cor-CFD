function result = Cor_CFD(budget,lambda,x,e,regress_num,fun,sigma_f)

pd = makedist('Normal','mu',0,'sigma',0.1);
lower = 0.001;
upper = inf;
t = truncate(pd,lower,upper);
BootTime = 100;

n1 = lambda * budget;
bootmean_num = ceil(n1/regress_num);
%delta = normrnd(0,sqrt(0.1),1,regress_num);
delta = random(t,1,regress_num);
delta = delta.*(bootmean_num)^(-1/10);

Y_mid = zeros(bootmean_num,regress_num);
MeanBootVar = zeros(1,regress_num);
MeanBootE = zeros(1,regress_num);
d = length(e);



for i = 1:regress_num
    Y_plus = fun(x+delta(i)*e) + normrnd(0,sigma_f,bootmean_num,1); 
    Y_minus = fun(x-delta(i)*e) + normrnd(0,sigma_f,bootmean_num,1);
    Y_mid(:,i) = (Y_plus - Y_minus)./(2*delta(i));
    [BootStatistics, ~] = bootstrp(BootTime, @mean, Y_mid(:,i));
    BootResult = std(BootStatistics); 
    MeanBootVar(i) = BootResult.^2;
    MeanBootE(i) = mean(BootStatistics);
end

Vtem = 1/2/(bootmean_num^2/(bootmean_num-1)).*ones(1,regress_num)'\(MeanBootVar.*delta.^2)';
sigmaSquared_hat = Vtem;

Btem = [ones(1,regress_num)./sqrt(MeanBootVar);delta.^2./sqrt(MeanBootVar)]'\(MeanBootE./sqrt(MeanBootVar))';
B_hat = Btem(2);
f_one = Btem(1);

n2 = budget - n1;

Delta = (sigmaSquared_hat/4/B_hat^2)^(1/6)/budget^(1/6);
if n2 > 0
    Y1 = fun(x+Delta*e) + normrnd(0,sigma_f,bootmean_num,1); 
    Y2 = fun(x-Delta*e) + normrnd(0,sigma_f,bootmean_num,1);
    Y = (Y1-Y2)/2/Delta;
else
    Y = 0;
end

tranform = (Y_mid - f_one - B_hat.*delta.^2).*abs(delta)./Delta;
tranform = tranform + f_one + B_hat*Delta^2;

theta = (mean(Y)*n2 + mean(mean(tranform))*n1)/budget;
result = theta;
