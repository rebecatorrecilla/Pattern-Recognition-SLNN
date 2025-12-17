%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ╔═╗┌─┐┌┬┐┬┌┬┐┬┌─┐┌─┐┌┬┐┬┌─┐┌┐┌  ┌─┐┬  ┌─┐┌─┐┬─┐┬┌┬┐┬ ┬┌┬┐┌─┐/  ┌─┐┬ ┬┌┐┌┌─┐┌┬┐┬┌─┐┌┐┌
% ║ ║├─┘ │ │││││┌─┘├─┤ │ ││ ││││  ├─┤│  │ ┬│ │├┬┘│ │ ├─┤│││└─┐   ├┤ │ │││││   │ ││ ││││
% ╚═╝┴   ┴ ┴┴ ┴┴└─┘┴ ┴ ┴ ┴└─┘┘└┘  ┴ ┴┴─┘└─┘└─┘┴└─┴ ┴ ┴ ┴┴ ┴└─┘   └  └─┘┘└┘└─┘ ┴ ┴└─┘┘└┘


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [sol, par] = uoss(P, x, par)

%
% Initializations
%

n = size(x,1);       f = P.f;                                
ldescent = true;     g = P.g;
k = 1;               sol(k).x = x;
sol(k).d  = [];      sol(k).g  = [];
sol(k).ng = [];      sol(k).AC = [];
Xte = par.sg.Xte;    yte = par.sg.yte;
Xtr = par.sg.Xtr;    ytr = par.sg.ytr;

if par.isd == 3
    H = eye(n);
end

%
% Algorithm
%

% SG
if par.isd == 7

    % SGM Variables
    rng(par.sg.seed);             k = 0;
    p = size(Xtr, 2);             al0 = par.sg.al0; 
    m = par.sg.m;                 sg_ke = ceil(p/m);
    s = 0;                        e = 0; 
    sg_kmax = par.sg.emax*sg_ke;  w = x;
    Lte_best = inf;               sg_k = floor(par.sg.be * sg_kmax);
    sg_al = 0.01 * al0;           Lte = 0;

    while (e <= par.sg.emax) && (s < par.sg.eworse)
        rp = randperm(p); 

        for i = 0:ceil(p/m - 1)

            % Determine the indices for the current mini-batch
            S = rp((i*m + 1) : min((i+1)*m, p));

            % Extract the mini_batch
            Xtr_s = Xtr(:, S);
            ytr_s = ytr(:, S);

            d = -g(w, Xtr_s, ytr_s);
            
            % Determine alpha max
            if k <= sg_k
                al = (1-k/sg_k)*al0 + (k/sg_k)*sg_al;
            else 
                al = 0.01*al0;
            end

            % Update
            w = w + al*d;    k = k + 1; 
            sol(k).x = w;    sol(k).AC = 0;
            sol(k).al = al; 
        end

        e = e + 1; Lte = f(w, Xte, yte);

        if Lte < Lte_best
            Lte_best = Lte; 
            bsg.Lte_best = Lte;
            w_star = w;
            s = 0;
            bsg.eo = e;
            bsg.ko = k;
        else
            s = s + 1;
        end
    end
    x = w_star;
    
    % Best sol
    sol = sol(1:bsg.ko);
    sol(bsg.ko).x = w_star;
    
    sol(bsg.ko).g  = g(w_star, Xtr, ytr);
    sol(bsg.ko).ng = norm(sol(bsg.ko).g);

    % Save the final values
    bsg.ktot = k;
    bsg.etot = e;
    sol(k).x = w_star;
    sol(k).iter = k;
    %sol = uosolLog(P, par, sol, bsg);
         
else

    while (norm(g(x)) > par.epsG) && (k < par.maxiter) && (ldescent || par.isd == 4)
        lastx = x;
        if par.isd == 1
            % GM
            d = -g(x);
    
        elseif par.isd == 3
            % BFGS
            d = -H * g(x); 
        end
      
        % Find alpha max
        if k > 1 && abs(g(x)'*d) ~= 0
            par.almax = 2 * abs(f(x)-f(sol(k-1).x)) / abs(g(x)'*d);
        else
            par.almax = 1;  
        end

        % B-LineSearch
        [al, iout] = uoBLSNW32(f, g, x, d, par.almax, par.c1, par.c2); 
        if iout == 0
            ACout = "SWC";
        elseif iout == 1
            ACout = "+30iter";
        elseif iout == 2
            ACout = "eps";
        end
 
        % Update solution
        sol(k).x  = x;
        sol(k).g  = g(x);
        sol(k).ng = norm(g(x));
        sol(k).d  = d;
        sol(k).al = al;
        sol(k).AC = ACout;
        if par.isd == 3
            sol(k).H = H;
        end
    
        % Next iteration
        ldescent = d' * g(x) < 0;
        x = x + al * d; 

        if par.isd == 3
            sx = x - lastx;
            yx = g(x) - g(lastx);
            rho = 1 / (yx' * sx);

            H = (eye(n) - rho * sx * yx') * H * (eye(n) - rho * yx * sx') + rho * (sx * sx');
        end 

        k = k + 1;
    end 
    
    % Final values
    sol(k).x  = x;
    sol(k).g  = g(x);
    sol(k).ng = norm(g(x));
    sol(k).iter = k;
    sol = uosolLog(P, par, sol);

end
end

% [end] Function [uoss] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
