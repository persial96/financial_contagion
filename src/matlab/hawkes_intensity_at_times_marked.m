function lambda_q = hawkes_intensity_at_times_marked(t_query, t_events, ids, marks, lambda_vec, alpha, beta)
% Intensities lambda_i(t) of the marked multivariate exponential Hawkes process at the query times (Eq. 5).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland
%
% Only events strictly before a query time contribute, and events with the same timestamp are added as one block:
% same-day events are co-arrivals, not causes of each other. Marks must be on the estimation scale (transform_marks).

    t_query    = double(t_query(:));
    lambda_vec = double(lambda_vec(:));
    beta       = double(beta(:));
    m = numel(lambda_vec);

    [t_events, idx] = sort(double(t_events(:)));
    ids   = double(ids(idx));
    marks = double(marks(idx));
    N = numel(t_events);

    [tq, ord] = sort(t_query);
    lambda_sorted = zeros(numel(tq), m);
    lambda_prev = lambda_vec;   % intensities right after the last processed block
    t_prev = 0;
    k = 1;

    for q = 1:numel(tq)

        % Absorb the event blocks strictly before the query time.
        while k <= N && t_events(k) < tq(q)
            tk = t_events(k);
            lambda_t = lambda_vec + (lambda_prev - lambda_vec) .* exp(-beta * (tk - t_prev));
            jump = zeros(m, 1);
            while k <= N && t_events(k) == tk
                jump = jump + marks(k) * alpha(ids(k), :).';
                k = k + 1;
            end
            lambda_prev = lambda_t + jump;
            t_prev = tk;
        end

        % Decay from the last block up to the query time.
        lambda_sorted(q, :) = (lambda_vec + (lambda_prev - lambda_vec) .* exp(-beta * (tq(q) - t_prev))).';
    end

    lambda_q = zeros(numel(tq), m);
    lambda_q(ord, :) = lambda_sorted;
end
