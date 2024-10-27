%%% This is an optimizer with Cor-CFD-GD
%%% Problem dimension: d
%%% The max function evaluations: n
%%% The initial value: x0
%%% Estimating gradient using Cor-CFD method
%%% Problem function 79
clc
clear

d = 64;

fun = @(x) sum( (10*(x(2:2:end) - x(1:2:end-1)).^2 + (1 - x(1:2:end-1)).^2).^4);

n = 1000 * d * 2;
batch_size = 20;
regress_size = 5;
sigma_f = 1;

rep = 1;
result = zeros(rep,d+1);
file_str = ['function79_GD_dim=', num2str(d), '_noise=', num2str(sigma_f), '_budget=', num2str(int32(n/d/2)), '.txt'];
fid = fopen(file_str,'a');

for k = 1:rep
    k
    tic;

    x0 = ones(1, d);
    x0(1:2:end) = 3;

    e = eye(d);
    
    x = x0;
    g = zeros(1,d);
    g_new = zeros(1,d);

    % Initial gradient estimation
    for i= 1:d
        g(i) = Cor_CFD(batch_size,1,x,e(i,:),regress_size, fun, sigma_f);
    end
    search_direction = -g';
    num_eval = batch_size * d * 2;
    iteration = 0;

    %%% Max function evaluations for line search (not necessary)
    a_max = 20;
    
    %%% Line search parameters
    c1 = 1e-4;
    c2 = 0.5;
    
    while num_eval < n
        iteration = iteration + 1;
        %%% Stochastic line search for step_size
        step_size = 1;
        fun_value_now = fun(x) + normrnd(0,sigma_f);
        num_eval = num_eval + 1;
        while fun(x+step_size*search_direction') + normrnd(0,sigma_f) > fun_value_now + c1*step_size*g*search_direction + 2*sigma_f
            step_size = c2*step_size;
            num_eval = num_eval + 1;
        end

        %% Update x and gradient
        x_new = x + step_size * search_direction';

        for i = 1:d
            g_new(i) = Cor_CFD(fix((batch_size+iteration)/regress_size)*regress_size,1,x_new,e(i,:),regress_size, fun, sigma_f);
        end

        g = g_new;
        x = x_new;

        search_direction = -g';
        num_eval = num_eval + fix((batch_size+iteration)/regress_size) * regress_size * d * 2;
    end
    result(k,:) = [x,fun(x)];
    toc;
end

for k = 1:rep
    for w = 1:(d+1)
        if w == d+1
            fprintf(fid,'%14.10f\n',result(k,w));
        else
            fprintf(fid,'%14.10f\t',result(k,w));
        end
    end
end

fclose(fid);

