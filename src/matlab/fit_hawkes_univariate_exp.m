function model = fit_hawkes_univariate_exp(event_times, T)
% MLE of the univariate exponential Hawkes process lambda(t) = lambda0 + sum alpha * exp(-beta * (t - t_i)) (Eq. 2-3).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    t = sort(event_times(:));
    N = numel(t);

    % Initial guess: half of the events to the baseline, half to excitation, decay of one mean gap.
    theta0 = [max(0.5 * N / T, 1e-4); max(0.5 * N / T, 1e-4); 1 / max(T / N, 1e-3)];

    % Optimization on log-parameters to keep them positive.
    negll = @(theta_log) -loglik(t, T, exp(theta_log));
    theta_log = fminsearch(negll, log(theta0), optimset('Display', 'off', 'TolX', 1e-7, 'TolFun', 1e-7));
    theta = exp(theta_log);

    model.lambda0 = theta(1);
    model.alpha   = theta(2);
    model.beta    = theta(3);
    model.eta     = theta(2) / theta(3);   % branching ratio
    model.loglik  = loglik(t, T, theta);

    % Compensator at the event times, input of the time-rescaling diagnostic. Between events the excess intensity
    % decays deterministically, so its integral over each gap is closed form.
    Lambda_events = zeros(N, 1);
    Lambda = 0;
    lam    = theta(1);   % intensity right after the last event
    t_prev = 0;
    for k = 1:N
        dt     = t(k) - t_prev;
        Lambda = Lambda + theta(1) * dt + (lam - theta(1)) / theta(3) * (1 - exp(-theta(3) * dt));
        lam    = theta(1) + (lam - theta(1)) * exp(-theta(3) * dt) + theta(2);
        Lambda_events(k) = Lambda;
        t_prev = t(k);
    end
    model.Lambda_events = Lambda_events;
end


function ll = loglik(t, T, theta)
% Exact log-likelihood with the Ozaki recursion A_i = exp(-beta * dt_i) * (1 + A_{i-1}).
    lambda0 = theta(1);
    alpha   = theta(2);
    beta    = theta(3);
    N = numel(t);

    A = zeros(N, 1);
    for i = 2:N
        A(i) = exp(-beta * (t(i) - t(i-1))) * (1 + A(i-1));
    end

    ll = sum(log(lambda0 + alpha * A)) - lambda0 * T - (alpha / beta) * sum(1 - exp(-beta * (T - t)));
end
