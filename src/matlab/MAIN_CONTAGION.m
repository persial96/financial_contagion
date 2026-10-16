%% MAIN - FINANCIAL CONTAGION
% Estimates the Poisson, univariate Hawkes and marked multivariate Hawkes models on the sector stress events of one
% crisis window, and exports the tables and figure data of the paper to data/results.
% Author: Luca Persia (2026), Università della Svizzera italiana, Lugano, Switzerland

clear; clc; close all;

%% SETTINGS
% Crisis window: "GFC" or "COVID". The script is run once per window.
WINDOW = "GFC";

% Minimum number of events for a sector to enter the multivariate fit.
MIN_EVENTS = 10;

% Paths relative to the repository root (this file is in src/matlab).
SRC_DIR = fileparts(mfilename('fullpath'));
if isempty(SRC_DIR)
    SRC_DIR = pwd;
end
ROOT_DIR    = fileparts(fileparts(SRC_DIR));
RESULTS_DIR = fullfile(ROOT_DIR, 'data', 'results');
addpath(SRC_DIR);
if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR);
end

% Event file (5% left tail, one event per day), observation window and tag of the output files.
switch upper(WINDOW)
    case "GFC"
        CSV_EVENTS = fullfile(ROOT_DIR, 'data', 'events', 'events_daily_GFC_2008_q95_oneperday.csv');
        OBS_START  = datetime("2007-06-01");
        OBS_END    = datetime("2011-01-01");
        OUT_TAG    = "";
    case "COVID"
        CSV_EVENTS = fullfile(ROOT_DIR, 'data', 'events', 'events_daily_COVID_2020_q95_oneperday.csv');
        OBS_START  = datetime("2019-06-01");
        OBS_END    = datetime("2024-01-01");
        OUT_TAG    = "_covid";
    otherwise
        error('WINDOW must be "GFC" or "COVID".');
end

%% LOAD EVENTS
fprintf("\n=== WINDOW: %s ===\n", WINDOW);
df = readtable(CSV_EVENTS, 'VariableNamingRule', 'preserve');
df.date   = datetime(df.date);
df.market = string(df.market);
df = df(df.date >= OBS_START & df.date <= OBS_END, :);
df = sortrows(df, {'date', 'market'});

%% EVENT SUMMARY (TABLE 1)
% Computed before the sparsity filter, so sectors excluded from the multivariate fit are still reported.
sector_names   = ["Materials"; "Energy"; "Financials"; "Industrials"; ...
                  "Technology"; "Utilities"; "Health Care"; "Consumer Discretionary"];
sector_tickers = ["XLB"; "XLE"; "XLF"; "XLI"; "XLK"; "XLU"; "XLV"; "XLY"];
n_sectors = numel(sector_tickers);

summary_table = table(sector_names, sector_tickers, zeros(n_sectors, 1), NaT(n_sectors, 1), NaT(n_sectors, 1), ...
    nan(n_sectors, 1), nan(n_sectors, 1), nan(n_sectors, 1), 'VariableNames', ...
    {'Sector', 'Ticker', 'No_events', 'Start', 'End', 'Mean_mark', 'Max_mark', 'Mean_gap'});

for i = 1:n_sectors
    tmp = df(df.market == sector_tickers(i), :);
    if isempty(tmp)
        continue;
    end
    summary_table.No_events(i) = height(tmp);
    summary_table.Start(i)     = min(tmp.date);
    summary_table.End(i)       = max(tmp.date);
    summary_table.Mean_mark(i) = mean(tmp.mark);
    summary_table.Max_mark(i)  = max(tmp.mark);
    if height(tmp) >= 2
        % Mean gap between consecutive events, in calendar days.
        summary_table.Mean_gap(i) = mean(days(diff(tmp.date)));
    end
end

disp(summary_table);
writetable(summary_table, fullfile(RESULTS_DIR, "event_summary" + OUT_TAG + ".csv"));

%% SPARSITY FILTER AND MODEL INPUTS
[G, market_names] = findgroups(df.market);
counts = splitapply(@numel, df.date, G);
valid_markets = market_names(counts >= MIN_EVENTS);
if any(counts < MIN_EVENTS)
    fprintf("Dropping markets with fewer than %d events: %s\n", MIN_EVENTS, strjoin(market_names(counts < MIN_EVENTS), ", "));
end

df = df(ismember(df.market, valid_markets), :);
risk_types = cellstr(valid_markets');
m = numel(risk_types);

% Time in calendar days since the first event of the window; market ids follow the alphabetical order of risk_types.
time_origin = min(df.date);
t_events  = days(df.date - time_origin);
T_horizon = max(t_events);
ids = double(categorical(df.market, risk_types));
marks_raw = df.mark;

% Marks on the estimation scale (Eq. 4), used for intensities and compensators after the fit.
marks_fit = transform_marks(marks_raw);

% Univariate models: sector events aggregated to unique stress days.
t_uni = days(unique(df.date) - time_origin);

%% UNIVARIATE BENCHMARKS
poisson_hom   = fit_poisson_hom(t_uni, T_horizon);
poisson_inhom = fit_poisson_inhom_loglin(t_uni, T_horizon);
hawkes_uni    = fit_hawkes_univariate_exp(t_uni, T_horizon);

fprintf("\nLog-likelihoods: Poisson %.2f, inhomogeneous Poisson %.2f, univariate Hawkes %.2f (eta = %.3f)\n", ...
    poisson_hom.loglik, poisson_inhom.loglik, hawkes_uni.loglik, hawkes_uni.eta);

%% MULTIVARIATE MARKED HAWKES (TABLES 2-3)
hawkes_multi = fit_hawkes_multivariate_exp_marks(t_events, ids, T_horizon, marks_raw);

% The fit sorts the events by time; with one event per day the order is unchanged and the marks must coincide.
assert(max(abs(hawkes_multi.marks - marks_fit)) < 1e-12, 'Marks of the fit and of the main script are not aligned.');

% Contagion matrix B_mark: rows = source, columns = target.
B_table = array2table(hawkes_multi.B_mark, 'VariableNames', risk_types, 'RowNames', risk_types);
disp(B_table);
writetable(B_table, fullfile(RESULTS_DIR, "branching_matrix" + OUT_TAG + ".csv"), 'WriteRowNames', true);

%% GOODNESS OF FIT (FIGURES 1 AND 3)
% Time-rescaling: the compensator increments between consecutive events should be i.i.d. Exp(1) (Eq. 12).
Y = {diff([0; poisson_hom.Lambda_events]), ...
     diff([0; poisson_inhom.Lambda_events]), ...
     diff([0; hawkes_uni.Lambda_events]), ...
     hawkes_multi_compensator(t_events, ids, marks_fit, hawkes_multi.lambda, hawkes_multi.alpha, hawkes_multi.beta)};
model_names = ["Homogeneous Poisson"; "Inhomogeneous Poisson"; "Univariate Hawkes"; "Multivariate Superposed"];

qq_stats = table(model_names, nan(4, 1), nan(4, 1), nan(4, 1), 'VariableNames', {'Model', 'KS_stat', 'p_value', 'N'});
rng(123);   % reproducible envelopes

for k = 1:4
    % The first increment is zero (the first event is the time origin) and is dropped.
    y = sort(Y{k}(Y{k} > 0));
    n = numel(y);

    % Theoretical Exp(1) quantiles, KS test and pointwise 95% envelope from 2000 simulated samples.
    theo = -log(1 - ((1:n)' - 0.5) / n);
    [~, p_ks, ks] = kstest(y, "CDF", makedist("Exponential", "mu", 1));
    qq_stats{k, 2:4} = [ks, p_ks, n];
    sim = sort(exprnd(1, [n, 2000]), 1);

    writetable(table(theo, y, prctile(sim, 2.5, 2), prctile(sim, 97.5, 2), 'VariableNames', {'theo', 'emp', 'lo', 'hi'}), ...
        fullfile(RESULTS_DIR, sprintf("qq%s_%d.csv", OUT_TAG, k)));
end

disp(qq_stats);
writetable(qq_stats, fullfile(RESULTS_DIR, "qq_stats" + OUT_TAG + ".csv"));

%% INTENSITY DECOMPOSITION (FIGURES 2 AND 4, INPUT OF TABLE 4)
% One point per calendar day d, evaluated at the close of the day (t = d + 1), i.e. including the events of day d.
day_idx = (0:floor(T_horizon))';
grid_t  = day_idx + 1;

intens_multi = hawkes_intensity_at_times_marked(grid_t, t_events, ids, marks_fit, ...
    hawkes_multi.lambda, hawkes_multi.alpha, hawkes_multi.beta);
lambda_uni = hawkes_univ_intensity_at_times(grid_t, t_uni, hawkes_uni.lambda0, hawkes_uni.alpha, hawkes_uni.beta);

dates = time_origin + days(day_idx);
dates.Format = 'yyyy-MM-dd';
intensity_table = table(dates, sum(intens_multi, 2), lambda_uni, 'VariableNames', {'date', 'lambda_tot', 'lambda_uni'});
for j = 1:m
    intensity_table.(lower(risk_types{j})) = intens_multi(:, j);
end

writetable(intensity_table, fullfile(RESULTS_DIR, "daily_intensity_decomp" + OUT_TAG + ".csv"));
fprintf("\nDone (%s). Results saved in %s\n", WINDOW, RESULTS_DIR);
