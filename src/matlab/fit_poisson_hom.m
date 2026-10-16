function model = fit_poisson_hom(event_times, T)
% Closed-form MLE of the homogeneous Poisson process on [0, T] (Definition 1).
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

    t = sort(event_times(:));
    N = numel(t);

    model.lambda = N / T;
    model.loglik = N * log(model.lambda) - model.lambda * T;

    % Compensator at the event times, input of the time-rescaling diagnostic.
    model.Lambda_events = model.lambda * t;
end
