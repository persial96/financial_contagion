function lam_q = hawkes_univ_intensity_at_times(t_query, t_events, mu, alpha, beta)
% Intensity of the univariate exponential Hawkes process at the query times (events strictly before each query).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    [tq, ord] = sort(t_query(:));
    t_ev = sort(t_events(:));
    N = numel(t_ev);

    lam_sorted = zeros(numel(tq), 1);
    lam_prev = mu;   % intensity right after the last processed event
    t_prev = 0;
    k = 1;

    % Single sweep over queries and events, decaying the excess intensity between points.
    for i = 1:numel(tq)
        while k <= N && t_ev(k) < tq(i)
            lam_prev = mu + (lam_prev - mu) * exp(-beta * (t_ev(k) - t_prev)) + alpha;
            t_prev = t_ev(k);
            k = k + 1;
        end
        lam_sorted(i) = mu + (lam_prev - mu) * exp(-beta * (tq(i) - t_prev));
    end

    lam_q = zeros(numel(tq), 1);
    lam_q(ord) = lam_sorted;
end
