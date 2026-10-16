function model = fit_hawkes_multivariate_exp_marks(event_times, event_ids, T, event_marks)
% Penalized MLE of the marked multivariate exponential Hawkes process (Eq. 5, 10-11).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    % Penalty weights and stability threshold of Eq. (11).
    PEN.LAMBDA_B    = 1e-4;
    PEN.LAMBDA_BETA = 1e-6;
    PEN.LAMBDA_S    = 1e6;
    PEN.RHO_BAR     = 0.95;

    %% SORTING AND MARKS
    t   = double(event_times(:));
    ids = double(event_ids(:));
    if isempty(T)
        T = max(t);
    end
    N = numel(t);
    m = max(ids);

    [t, idx] = sort(t);
    ids = ids(idx);
    marks_raw = double(event_marks(:));
    marks_raw = marks_raw(idx);

    % Events with the same timestamp are shifted by a tiny jitter, so that each event only sees strictly earlier
    % events (never triggered with one event per day).
    for i = 2:N
        if t(i) <= t(i-1)
            t(i) = t(i-1) + 1e-5;
        end
    end
    T = max(T, t(end));

    % Transformed marks (Eq. 4) and average mark of each source market, used for B_mark = diag(E[m|j]) * B.
    marks = transform_marks(marks_raw);
    mean_mark = accumarray(ids, marks, [m, 1], @mean, 1);

    %% INITIAL PARAMETERS
    % Baseline at 20% of the naive rate, decay equal to the inverse of the median gap, small branching coefficients.
    mu0 = max(0.20 * accumarray(ids, 1, [m, 1]) / T, 1e-6);

    dt_all = diff(t);
    med_dt = median(dt_all(dt_all > 1e-4));
    if isempty(med_dt) || isnan(med_dt)
        med_dt = 1;
    end
    beta0 = (1 / max(med_dt, 1e-3)) * ones(m, 1);

    B0 = (0.10 / max(m - 1, 1)) * ones(m, m);
    B0(logical(eye(m))) = 0.20;

    % All parameters are optimized on the log scale: theta = [log mu; log B(:); log beta].
    theta0 = [log(mu0); log(B0(:)); log(beta0)];

    %% OPTIMIZATION
    loss = @(theta) penalized_loss(theta, t, ids, marks, T, m, mean_mark, PEN);

    opts = optimoptions('fminunc', ...
        'Display', 'iter-detailed', ...
        'Algorithm', 'trust-region', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxFunctionEvaluations', 10000, ...
        'MaxIterations', 1000, ...
        'OptimalityTolerance', 1e-6, ...
        'StepTolerance', 1e-8);

    theta_hat = fminunc(loss, theta0, opts);

    %% OUTPUT
    [mu, B, beta] = unpack(theta_hat, m);
    B_mark = diag(mean_mark) * B;

    model.lambda   = mu;
    model.alpha    = B .* beta.';
    model.beta     = beta;
    model.B        = B;
    model.B_mark   = B_mark;
    model.marks    = marks;
    model.loglik   = -negloglik(t, ids, marks, T, mu, B, beta);
    model.rho_mark = max(abs(eig(B_mark)));

    fprintf("\nLoglik: %.4f   rho(B_mark): %.4f\n", model.loglik, model.rho_mark);
    if any(sum(B_mark, 1) > PEN.RHO_BAR)
        warning('The stability barrier is binding: some column sums of B_mark exceed %.2f.', PEN.RHO_BAR);
    end
end


function [L, g] = penalized_loss(theta, t, ids, marks, T, m, mean_mark, PEN)
% Objective of Eq. (11) and its gradient w.r.t. the log-parameters.

    [mu, B, beta] = unpack(theta, m);
    [L, g] = negloglik(t, ids, marks, T, mu, B, beta);

    if ~isfinite(L)
        L = 1e10;
        g = zeros(size(theta));
        return;
    end

    % Stability barrier on the column sums of B_mark above RHO_BAR.
    excess = max(sum(diag(mean_mark) * B, 1) - PEN.RHO_BAR, 0);
    L = L + PEN.LAMBDA_S * sum(excess.^2) ...
          + PEN.LAMBDA_B * sum(B(:).^2) ...
          + PEN.LAMBDA_BETA * sum(log(beta).^2);

    % Penalty gradients (chain rule for the log scale: d/dlog(x) = x * d/dx).
    dB    = 2 * PEN.LAMBDA_S * mean_mark * excess + 2 * PEN.LAMBDA_B * B;
    dbeta = 2 * PEN.LAMBDA_BETA * log(beta) ./ beta;

    idx_B    = m + (1:m^2);
    idx_beta = m + m^2 + (1:m);
    g(idx_B)    = g(idx_B)    + dB(:) .* B(:);
    g(idx_beta) = g(idx_beta) + dbeta .* beta;
end


function [nll, grad] = negloglik(t, ids, marks, T, mu, B, beta)
% Negative log-likelihood of Eq. (10) and its gradient w.r.t. [log mu; log B(:); log beta].

    N = numel(t);
    m = numel(mu);

    ll     = 0;
    g_mu   = zeros(m, 1);
    g_B    = zeros(m, m);
    g_beta = zeros(m, 1);

    %% EVENT TERM
    % A(j,i) is the sum over past type-j events of m * exp(-beta_i * (t - t_n)), and A_pr(j,i) the same sum weighted by
    % (t - t_n), needed for the derivative in beta. Both are updated recursively between consecutive events.
    A      = zeros(m, m);
    A_pr   = zeros(m, m);
    t_prev = 0;

    for k = 1:N
        dt = t(k) - t_prev;
        d  = ids(k);

        decay = exp(-beta.' * dt);
        A_pr  = (A_pr + A * dt) .* decay;
        A     = A .* decay;

        % Intensity of the market where the event occurs, just before the event (Eq. 9).
        lambda_d = mu(d) + beta(d) * sum(B(:, d) .* A(:, d));
        if lambda_d <= 0
            nll = Inf;
            grad = zeros(m + m^2 + m, 1);
            return;
        end

        ll = ll + log(lambda_d);
        g_mu(d)   = g_mu(d)   + 1 / lambda_d;
        g_B(:, d) = g_B(:, d) + beta(d) * A(:, d) / lambda_d;
        g_beta(d) = g_beta(d) + sum(B(:, d) .* (A(:, d) - beta(d) * A_pr(:, d))) / lambda_d;

        A(d, :) = A(d, :) + marks(k);
        t_prev  = t(k);
    end

    %% COMPENSATOR TERM (Eq. 8)
    % G(j,i) is the sum over type-j events of m * (1 - exp(-beta_i * (T - t_k))), so that the excitation part of the
    % compensator is sum(G .* B).
    G = zeros(m, m);
    for k = 1:N
        dt = T - t(k);
        e  = exp(-beta.' * dt);
        G(ids(k), :) = G(ids(k), :) + marks(k) * (1 - e);
        g_beta = g_beta - (marks(k) * B(ids(k), :) .* e * dt).';
    end

    ll   = ll - sum(mu) * T - sum(G(:) .* B(:));
    g_mu = g_mu - T;
    g_B  = g_B - G;

    nll  = -ll;
    grad = -[g_mu .* mu; g_B(:) .* B(:); g_beta .* beta];
end


function [mu, B, beta] = unpack(theta, m)
% Back from the log-parameter vector to (mu, B, beta).
    mu   = exp(theta(1:m));
    B    = reshape(exp(theta(m + (1:m^2))), m, m);
    beta = exp(theta(m + m^2 + (1:m)));
end
