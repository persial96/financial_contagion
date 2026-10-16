function model = fit_poisson_inhom_loglin(event_times, T)
% MLE of the inhomogeneous Poisson process with log-linear intensity exp(d0 + d1 * t) on [0, T] (Eq. 1).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    t = sort(event_times(:));
    N = numel(t);
    S = sum(t);

    % For a given slope d1 the MLE of d0 is closed form, exp(d0) = N / int_0^T exp(d1 s) ds, so we only maximize the
    % profile log-likelihood over the dimensionless slope u = d1 * T.
    profile_ll = @(u) -N * log(int_exp(u / T, T)) + (u / T) * S;
    u_hat = fminbnd(@(u) -profile_ll(u), -50, 50, optimset('TolX', 1e-12));

    d1 = u_hat / T;
    d0 = log(N / int_exp(d1, T));

    model.beta   = [d0; d1];
    model.loglik = N * d0 + d1 * S - N;

    % Compensator at the event times, input of the time-rescaling diagnostic.
    model.Lambda_events = exp(d0) * int_exp(d1, t);
end


function I = int_exp(d1, x)
% int_0^x exp(d1 * s) ds, element-wise in x.
    if d1 == 0
        I = x;
    else
        I = expm1(d1 * x) ./ d1;
    end
end
