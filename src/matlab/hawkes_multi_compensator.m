function dLambda = hawkes_multi_compensator(t_events, ids, marks, lambda_vec, alpha, beta)
% Increments of the superposed compensator of the marked multivariate Hawkes process at the pooled events (Eq. 12).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    t_events   = double(t_events(:));
    ids        = double(ids(:));
    marks      = double(marks(:));
    lambda_vec = double(lambda_vec(:));
    beta       = double(beta(:));

    % Baseline part mu_i * t_k, then the excitation of every event strictly before t_k on all markets (Eq. 8).
    Lambda = t_events * lambda_vec.';
    for k = 1:numel(t_events)
        past = t_events < t_events(k);
        if any(past)
            dt = t_events(k) - t_events(past);
            W  = marks(past) .* (alpha(ids(past), :) ./ beta.');
            Lambda(k, :) = Lambda(k, :) + sum(W .* (1 - exp(-dt * beta.')), 1);
        end
    end

    dLambda = diff([0; sum(Lambda, 2)]);
end
