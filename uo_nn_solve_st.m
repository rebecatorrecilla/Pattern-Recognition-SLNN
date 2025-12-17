%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OM / GCED / F.-Javier Heredia https://gnom.upc.edu/heredia
% Function uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Input parameters:
%
% nn:
%          L : loss function.
%         gL : gradient of the loss function.
%        Acc : Accuracy function.
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
% par:
%       epsG : optimality tolerance.
%    maxiter : maximum number of iterations.
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg.al0 : \alpha^{SG}_0.
%      sg.be : \beta^{SG}.
%       sg.m : m^{SG}.
%    sg.emax : e^{SGÇ_{max}.
%   sg.eworse: e^{SG}_{worse}.
%    sg.seed : seed for the first random permutation of the SG.
%
% Output parameters:
%
% nnout
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     Lo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [nnout] = uo_nn_solve_st(nn,par)
Xtr=[];ytr=[];wo=[];Lo=0;tr_acc=0;Xte=[];yte=[];te_acc=0;niter=0;tex=0;Hk = [];
L_star = 0; la = nn.la;

fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n')
fprintf('[uo_nn_solve] %s\n',datetime)
fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('   ')

%
% Training dataset generation
%
fprintf('[uo_nn_solve] Training dataset generation.\n')

fprintf('[uo_nn_solve]    num_target = %d\n', nn.num_target);
fprintf('[uo_nn_solve]    tr_freq    = %.2f\n', nn.tr_freq);
fprintf('[uo_nn_solve]    tr_p       = %d\n', nn.tr_p);
fprintf('[uo_nn_solve]    tr_seed    = %d\n', nn.tr_seed);
[Xtr, ytr]= uo_nn_dataset(nn.tr_seed, nn.tr_p, nn.num_target, nn.tr_freq);
fprintf('   ')

%
% Test dataset generation
%
fprintf('[uo_nn_solve] Test dataset generation.\n');

fprintf('[uo_nn_solve]    te_freq    = %.2f\n', 0.0);
fprintf('[uo_nn_solve]    te_q       = %d\n', nn.te_q);
fprintf('[uo_nn_solve]    te_seed    = %d\n', nn.te_seed);
[Xte, yte] = uo_nn_dataset(nn.te_seed, nn.te_q, nn.num_target, 0.0);
fprintf('   ')

%
% Optimization
%
fprintf('[uo_nn_solve] Optimization\n');

fprintf('[uo_nn_solve]    L2 reg. lambda = %.4f\n', la);
fprintf('[uo_nn_solve]    w1= [0]\n');
fprintf('[uo_nn_solve]    Call uosol.\n');

tic

sig = @(Xtr) 1./(1+exp(-Xtr));
y = @(Xtr,w) sig(w'*sig(Xtr));

w = zeros(35, 1);

L  = nn.L;
gL = nn.gL;

if par.isd == 7
    P.f = @(w, Xtr, ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+(nn.la*norm(w)^2)/2;
    P.g = @(w, Xtr, ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+nn.la*w;
else
    P.f = @(w) L(w, Xtr, ytr); 
    P.g = @(w) gL(w, Xtr, ytr);
end

par.sg.Xtr = Xtr; par.sg.ytr = ytr;
par.sg.Xte = Xte; par.sg.yte = yte;

[sol, ~] = uoss(P, w, par);
tex = toc;

w_star = sol(end).x;
L_star = L(w_star, Xtr, ytr);
niter = sol(end).iter;

fprintf('[uo_nn_solve]    Optimization wall time = %.1e s.\n', tex);
fprintf('[uo_nn_solve]    niter = %d\n', niter);

fprintf('[uo_nn_solve]    wo=[\n');
for i = 1:5:length(w_star)
    idx_end = min(i+4, length(w_star));
    fprintf('[uo_nn_solve]    ');
    fprintf('%+6.1e,', w_star(i:idx_end));
    fprintf('\n');
end
fprintf('[uo_nn_solve]    ]\n');
fprintf('   ')

%
% Training accuracy
%
fprintf('[uo_nn_solve] Training Accuracy.\n');

tr_acc = nn.Acc(Xtr, ytr, w_star);
fprintf('[uo_nn_solve]    tr_accuracy = %.1f\n', tr_acc);
fprintf('   ')

%
% Test accuracy
%
fprintf('[uo_nn_solve] Test Accuracy.\n');

te_acc = nn.Acc(Xte, yte, w_star);
fprintf('[uo_nn_solve]    te_accuracy = %.1f\n', te_acc);
fprintf('   ')

%
nnout.Xtr    = Xtr;
nnout.ytr    = ytr;
nnout.wo     = wo;
nnout.Lo     = L_star;
nnout.niter  = niter;
nnout.tex    = tex;
nnout.tr_acc = tr_acc;
nnout.Xte    = Xte;
nnout.yte    = yte;
nnout.te_acc = te_acc;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
