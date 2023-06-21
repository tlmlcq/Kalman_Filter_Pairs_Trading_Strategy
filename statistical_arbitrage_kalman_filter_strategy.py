import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from pykalman import KalmanFilter
import statsmodels.api as sm
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
import yfinance as yf
from sklearn.cluster import DBSCAN
import statsmodels.tsa.stattools as ts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from kneed import KneeLocator
import heapq
import operator
from pandas_datareader import data as web
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


class GetData:
    """
    Class GetData to retrieve market prices
    Initialisation: start and end date of the period to be retrieved
    """
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    """
    dl_close_data function retrieves daily prices from a list of tickers thanks to pandas_datareader
    :list_tickers: a list of tickers of type string
    :return: a dataframe with the adjusted closing price for each ticker in the list
    """
    def dl_close_data(self, list_tickers):
        print(f'Retrieve market data for {len(list_tickers)} tickers')
        res = {}
        for tick in list_tickers:
            yf.pdr_override()
            df = web.get_data_yahoo(tick, start=self.start_date, end=self.end_date, progress=False)
            df['ticker'] = tick
            res[tick] = df
        data = pd.concat(res)
        data.reset_index(inplace=True)
        data = data.pivot(index='Date', columns='ticker', values='Adj Close')
        return data

    """
    twelve_data function retrieves market prices from a list of tickers thanks to the Twelve Data API
    :list_tickers: a list of tickers of type string
    :interval: interval between each price collected : 1min, 5min, 30min, 1h, 4h, 1day
    :return: a dataframe with the adjusted closing price for each ticker in the list
    """
    @staticmethod
    def twelve_data(list_tickers, interval):
        print(f'Retrieve market data for {len(list_tickers)} tickers')
        url = "https://twelve-data1.p.rapidapi.com/time_series"
        full_df = pd.DataFrame(columns=list_tickers, index=range(5000))
        for ticket in list_tickers:
            querystring = {"symbol": ticket, "interval": interval, "outputsize": 5000, "format": "csv"}
            headers = {
                'x-rapidapi-key': "d48890a58emsh5e7affca7714a71p1ab6e1jsn4ba14ffa0b6c",
                'x-rapidapi-host': "twelve-data1.p.rapidapi.com"
            }
            response = requests.request("GET", url, headers=headers, params=querystring)
            test_data = StringIO(response.text)
            df = pd.read_csv(test_data, sep=";")
            df = df.assign(datetime=pd.to_datetime(df['datetime']))
            df = df.sort_values(by='datetime')
            df = df.reset_index().drop('index', axis=1)
            full_df[ticket] = df['close']
        return full_df


class GetOptPair:
    """
    Class GetOptPair to find cointegrated pairs from a list of tickers
    :significance: critical value for the ADF test, generally 5%
    :df_prices: dataframe containing all the prices for the defined period for each ticker
    """
    def __init__(self, significance, df_prices):
        self.significance = significance
        self.df_prices = df_prices

    """
    clustering function to find the different clusters among all the assets
    :data: dataframe containing all the prices for the defined period for each ticker
    :return: clusters found using the Kmeans algorithm
    """
    @staticmethod
    def clustering(data):
        list_k = range(2, 7)
        silhouettes = []
        for k in list_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='random')
            kmeans.fit(data)
            silhouettes.append(silhouette_score(data, kmeans.labels_))
        kl = KneeLocator(list_k, silhouettes, curve="convex", direction="decreasing")
        k_means = KMeans(n_clusters=kl.elbow)
        k_means.fit(data)
        clustered_series = pd.Series(index=data.index, data=k_means.labels_.flatten())
        clustered_series = clustered_series[clustered_series != -1]
        return clustered_series

    """
    clustering function to find the different clusters among all the assets using DBSCAN
    :data: dataframe containing all the prices for the defined period for each ticker
    :return: clusters found using the DBSCAN algorithm
    """
    @staticmethod
    def db_scan_clustering(data):
        clf = DBSCAN()
        clf.fit(data)
        labels = clf.labels_
        print(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print("Clusters discovered: %d" % n_clusters_)
        clustered = clf.labels_
        clustered_series = pd.Series(index=data.index, data=clustered.flatten())
        clustered_series_all = pd.Series(index=data.index, data=clustered.flatten())
        clustered_series = clustered_series[clustered_series != 1]
        cluster_size_limit = 500
        counts = clustered_series.value_counts()
        ticker_count_reduced = counts[(counts > 1) & (counts <= cluster_size_limit)]
        counts = clustered_series.value_counts()
        clusters_viz_list = list(counts[(counts < 500) & (counts > 1)].index)[::-1]
        print("Clusters formed: %d" % len(ticker_count_reduced))
        print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced - 1)).sum())
        return clustered_series

    """
    cointegration function to find the cointegrated pairs in each cluster
    :data: dataframe containing all the prices for the defined period for each ticker
    :return: clusters found using the Kmeans algorithm
    """
    def cointegration(self, cluster):
        pair_coin = []
        p_value = []
        n = cluster.shape[0]
        keys = cluster.keys()
        for i in range(n):
            for j in range(i + 1, n):
                asset_1 = self.df_prices[keys[i]]
                asset_2 = self.df_prices[keys[j]]
                results = sm.OLS(asset_1, asset_2).fit()
                predict = results.predict(asset_2)
                error = asset_1 - predict
                adf_test = ts.adfuller(error)
                if adf_test[1] < self.significance:
                    pair_coin.append([keys[i], keys[j]])
                    p_value.append(adf_test[1])
        return p_value, pair_coin

    """
    pair_selection function to find the cointegrated pairs
    :clustered_series: dataframe containing all the prices for the period for each ticker
    :return: a list of all cointegrated pairs
    """
    def pair_selection(self, clustered_series, e_selection=False):
        opt_pairs = []
        counts = clustered_series.value_counts()
        clusters_viz_list = list(counts[(counts < 500) & (counts > 1)].index)[::-1]
        if e_selection:
            for i in clusters_viz_list:
                cluster = clustered_series[clustered_series == i]
                result = GetOptPair.cointegration(self, cluster)
                if len(result[0]) > 0:
                    if np.min(result[0]) < self.significance:
                        index = np.where(result[0] == np.min(result[0]))[0][0]
                        opt_pairs.append([result[1][index][0], result[1][index][1]])
        else:
            p_value_contval = []
            pairs_contval = []
            for i in clusters_viz_list:
                cluster = clustered_series[clustered_series == i]
                result = GetOptPair.cointegration(self, cluster)
                if len(result[0]) > 0:
                    p_value_contval += result[0]
                    pairs_contval += result[1]
            opt_pair_index = heapq.nsmallest(25, range(len(p_value_contval)), key=p_value_contval.__getitem__)
            opt_pairs = operator.itemgetter(*opt_pair_index)(pairs_contval)
        return opt_pairs


class GetSpread:
    """
    Class GetSpread to compute the spread between each pair of cointegrated assets
    :df: dataframe containing all the prices for the defined period for each ticker
    :tick1: the ticker of the asset 1
    :tick2: the ticker of the asset 2
    """
    def __init__(self, df, tick1, tick2):
        self.df = df
        self.tick1 = tick1
        self.tick2 = tick2

    """
    kalman_filter function to initialize the Kalman Filter
    :return: the evolution of the alpha and beta parameters, the initialized Kalman Filter, the history of the state of the parameters and covariance
    """
    def kalman_filter(self):
        x = self.df[self.tick1][:10]
        y = self.df[self.tick2][:10]
        delta = 1e-3
        # How much random walk wiggles
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
        # y is 1-dimensional, (alpha, beta) is 2-dimensional
        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                          initial_state_mean=[0, 0],
                          initial_state_covariance=np.ones((2, 2)),
                          transition_matrices=np.eye(2),
                          observation_matrices=obs_mat,
                          observation_covariance=2,
                          transition_covariance=trans_cov)
        # Use the observations y to get running estimates and errors for the state parameters
        state_means, state_covs = kf.filter(y.values)
        res_param = pd.DataFrame(state_means,
                                 index=range(len(x)),
                                 columns=['beta', 'alpha'])
        return res_param, kf, state_means, state_covs

    """
    update_kalman_filter function to update the Kalman Filter thanks to the previous states computed during the initialization
    :kf: the initialized Kalman Filter
    :history_state_means: the history of the states of the parameters alpha and beta
    :history_state_covs: the history of the states of the covariance
    :return: the value of alpha and beta parameters 
    """
    def udpate_kalman_filter(self, kf, history_state_means, history_state_covs):
        n = self.df.shape[0]
        n_dim_state = 2
        new_state_means = np.zeros((n, n_dim_state))
        new_state_covs = np.zeros((n, n_dim_state, n_dim_state))
        for i in range(self.df.shape[0]):
            obs_mat = np.asarray([[self.df[self.tick2].iloc[i], 1]])
            if i == 0:
                new_state_means[i], new_state_covs[i] = kf.filter_update(history_state_means[-1],
                                                                                 history_state_covs[-1],
                                                                                 observation=self.df[self.tick1].iloc[i],
                                                                                 observation_matrix=obs_mat)
            else:
                new_state_means[i], new_state_covs[i] = kf.filter_update(new_state_means[-1],
                                                                                 new_state_covs[-1],
                                                                                 observation=self.df[self.tick1].iloc[i],
                                                                                 observation_matrix=obs_mat)

        means = pd.DataFrame(new_state_means,
                                  index=range(self.df.shape[0]),
                                  columns=['beta', 'alpha'])
        return means

    """
    update_estim_beta function to initialize, update the Kalman Filter and compute the spread, called misp_ptf
    :roll_scale_period: the period over which the spread will be centred and reduced  
    :return: a dataframe containing the price of the two assets, the value of alpha and beta and the value of the spread
    """
    def update_estim_beta(self, roll_scale_period):
        df = self.df.copy()
        states_param, kf, st_m, st_c = GetSpread.kalman_filter(self)
        states_param = GetSpread.udpate_kalman_filter(self, kf, st_m, st_c)
        df['beta'] = states_param['beta'].to_list()
        df['alpha'] = states_param['alpha'].to_list()
        df['misp_ptf'] = [df.reset_index().loc[i, self.tick1] - df.reset_index().loc[i, 'beta'] * df.reset_index().loc[
            i, self.tick2] - df.reset_index().loc[i, 'alpha'] for i in range(df.shape[0])]
        df['misp_ptf'] = [
            (df.reset_index().loc[i, 'misp_ptf'] - df.reset_index()['misp_ptf'].rolling(roll_scale_period).mean()[i]) /
            df.reset_index()['misp_ptf'].rolling(roll_scale_period).std()[i] for i in range(df.shape[0])]
        return df


class Backtest:
    """
    Class Backtest to backtest our strategy over a defined period
    """
    """
    simple_backtest function to backtest our strategy, taking positions according to the signal 
    :df: dataframe containing the price of the two assets, the value of alpha and beta and the value of the spread
    :tick1: the ticker of the asset 1
    :tick2: the ticker of the asset 2
    :params: a list of parameters to define the signal
    :return: the dataframe, the final value of our pnl, the evolution of our pnl
    """
    @staticmethod
    def simple_backtest(df, tick1, tick2, params):
        coeff_const = params[0]
        coeff_std = params[1]
        roll_window = params[2]
        coeff_exit = params[3]
        pos = 'nul'
        ptf = 100
        fees = 0.001
        nb_asset2 = 0
        nb_asset1 = 0
        val_trade = 0
        pnl = []
        df['pos_asset1'] = 0
        df['pos_asset2'] = 0
        df['roll_vol'] = df['misp_ptf'].rolling(roll_window).std()
        df['seuil_signal'] = coeff_const + coeff_std * df['roll_vol']
        df['pnl_strat'] = 0
        df['exit_signal'] = coeff_exit
        for i in range(roll_window, df.shape[0]):
            if pos == 'nul':
                if df['misp_ptf'].iloc[i] < -df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] >= - \
                df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_asset2'
                    df.loc[i, 'pos_asset2'] = 1
                    df.loc[i, 'pos_asset1'] = -1
                    pos_size = ptf/(df['beta'].iloc[i]*df[tick2].iloc[i]+df[tick1].iloc[i])
                    nb_asset2 = pos_size*df['beta'].iloc[i]
                    nb_asset1 = pos_size
                    val_trade = nb_asset2 * df[tick2].iloc[i] - 1.0 * nb_asset1 * df[tick1].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
                elif df['misp_ptf'].iloc[i] > df['seuil_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] <= \
                        df['seuil_signal'].iloc[i - 1]:
                    pos = 'short_asset1'
                    df.loc[i, 'pos_asset2'] = -1
                    df.loc[i, 'pos_asset1'] = 1
                    pos_size = ptf/(df['beta'].iloc[i]*df[tick2].iloc[i]+df[tick1].iloc[i])
                    nb_asset2 = pos_size*df['beta'].iloc[i]
                    nb_asset1 = pos_size
                    val_trade = -1.0 * nb_asset2 * df[tick2].iloc[i] + nb_asset1 * df[tick1].iloc[i]
                    ptf = 0
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']

                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i-1, 'pnl_strat']
            elif pos == 'short_asset2':
                if df['misp_ptf'].iloc[i] > -df['exit_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] <= - \
                df['exit_signal'].iloc[i - 1]:
                    pos = 'nul'
                    # long asset2, short asset1 to reverse position
                    df.loc[i, 'pos_asset2'] = -1
                    df.loc[i, 'pos_asset1'] = 1
                    pnl.append(
                        -1.0 * nb_asset2 * df[tick2].iloc[i] + nb_asset1 * df[tick1].iloc[i] + val_trade - (ptf * fees * 2))
                    df.loc[i, 'pnl_strat'] = df.loc[i - 1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i - 1, 'pnl_strat']
            else:
                if df['misp_ptf'].iloc[i] < df['exit_signal'].iloc[i] and df['misp_ptf'].iloc[i - 1] >= \
                        df['exit_signal'].iloc[i - 1]:
                    pos = 'nul'
                    # short asset2, long asset1 to reverse position
                    df.loc[i + 1, 'pos_asset2'] = 1
                    df.loc[i + 1, 'pos_asset1'] = -1
                    pnl.append(
                        nb_asset2 * df[tick2].iloc[i] - 1.0 * nb_asset1 * df[tick1].iloc[i] + val_trade - (ptf * fees * 2))
                    df.loc[i, 'pnl_strat'] = df.loc[i - 1, 'pnl_strat'] + pnl[-1]
                    ptf = 100
                    val_trade = 0
                else:
                    df.loc[i, 'pnl_strat'] = df.loc[i - 1, 'pnl_strat']

        if len(pnl) != 0:
            final_pnl = np.cumsum(pnl)[-1]
        else:
            final_pnl = 0

        return df, final_pnl, pnl


class OptimizeSignals:
    """
    Class OptimizeSignals to optimize the position-taking signal in our strategy
    Two kinds of optimization: Bayesian Search and Grid Search
    """
    """
    objective function to define the function to be minimized in the Bayesian Search
    :params: a list of parameters to define the signal
    :return: the final value of the pnl of the strategy with these specific parameters
    """
    @staticmethod
    def objective(params):
        _, final_pnl, _ = Backtest.simple_backtest(params['data'], params['tick1'], params['tick2'],
                                                   [params['const_coeff'], params['const_std'], int(params['roll_vol']),
                                                    params['coeff_exit']])
        return -1.0 * final_pnl

    """
    bayesian_opti function to minimize the objective function
    :objective: the objective function we wish to minimize 
    :space_param: the parameter space for each parameter to be optimized
    :max_evals: the number of iterations to be performed during optimization
    :return: the best value obtained over a specified number of iterations, the value of the optimal parameters 
    """
    @staticmethod
    def bayesian_opti(objective, space_param, max_evals):
        # Algorithm
        tpe_algorithm = tpe.suggest

        # Trials object to track progress
        bayes_trials = Trials()

        # Optimize
        best = fmin(fn=objective, space=space_param, algo=tpe_algorithm, max_evals=max_evals, trials=bayes_trials,
                    return_argmin=False)
        df_best = pd.DataFrame.from_dict(best, orient='index')
        print(bayes_trials.best_trial['result']['loss'])
        return df_best, bayes_trials.best_trial['result']['loss']

    """
    multi_bay_opti function to minimize the objective function over a number of iterations
    :objective: the objective function we wish to minimize 
    :space_param: the parameter space for each parameter to be optimized
    :nb: the number of iterations to be performed during optimization
    :return: a dataframe with the optimized value obtained and the corresponding parameters 
    """
    @staticmethod
    def multi_bay_opti(objective, space_param, nb=1):
        final = pd.DataFrame(
            columns=['const_coeff', 'const_std', 'roll_vol', 'coeff_exit', 'SCORE'], index=range(nb))
        for i in range(nb):
            optim = OptimizeSignals.bayesian_opti(objective, space_param, 100)
            final['const_std'].iloc[i] = optim[0].loc['const_std'][0]
            final['roll_vol'].iloc[i] = optim[0].loc['roll_vol'][0]
            final['const_coeff'].iloc[i] = optim[0].loc['const_coeff'][0]
            final['coeff_exit'].iloc[i] = optim[0].loc['coeff_exit'][0]
            final['SCORE'].iloc[i] = optim[1]
        return final

    """
    grid_search function to find the optimum parameters in specific spaces for each parameter
    :df: dataframe containing the price of the two assets, the value of alpha and beta and the value of the spread
    :return: a list of the optimum parameters 
    """
    @staticmethod
    def grid_search(df):
        # 570 iterations
        space_param0 = np.arange(0.5, 2.5, 0.2)
        space_param1 = np.arange(0.1, 1, 0.2)
        space_param2 = np.arange(15, 55, 10)
        pnl_res = []
        val_param0 = []
        val_param1 = []
        val_param2 = []
        print(f'nb iterations : {len(space_param0)*len(space_param1)*len(space_param2)}')
        for i in space_param0:
            for j in space_param1:
                for h in space_param2:
                    pnl_res.append(Backtest.simple_backtest(df, [i, j, h])[1])
                    val_param0.append(i)
                    val_param1.append(j)
                    val_param2.append(h)

        idx = pnl_res.index(max(pnl_res))
        return [pnl_res[idx], val_param0[idx], val_param1[idx], val_param2[idx]]


def main_cac40():
    cac40 = pd.read_html('https://en.wikipedia.org/wiki/CAC_40#Composition')
    cac40_list = np.array(cac40[4]['Ticker'])
    full_tickers = list(cac40_list)
    start = dt.datetime(2018, 1, 1)
    end = dt.datetime(2023, 6, 5)
    get_data = GetData(start, end)
    data = get_data.dl_close_data(full_tickers)

    df_nan = pd.DataFrame(data.isnull().sum(), columns=['nb_nan'])
    l_tick_to_del = df_nan.sort_values('nb_nan')[-2:].reset_index()['ticker'].to_list()
    data = data.drop(columns=l_tick_to_del)
    data.drop(columns=['WLN.PA'], inplace=True)
    df_prices = data.fillna(method='ffill')
    returns = data.pct_change().mean() * 252
    returns = pd.DataFrame(returns)
    returns.columns = ['returns']
    returns['volatility'] = data.pct_change().std() * np.sqrt(252)
    df_returns = returns.copy()
    scale = StandardScaler().fit(df_returns)
    scaled_data = pd.DataFrame(scale.fit_transform(df_returns), columns=df_returns.columns, index=df_returns.index)

    significance = 0.05
    find_opt_pairs = GetOptPair(significance, df_prices)
    clustered_series = find_opt_pairs.clustering(scaled_data)
    opt_pairs = find_opt_pairs.pair_selection(clustered_series)
    print("Number of cointegrated pairs: ", len(opt_pairs))
    print("Pairs with lowest p-value among all the clusters:", list(opt_pairs))
    all_pairs = []
    for i in range(len(opt_pairs)):
        all_pairs.append(opt_pairs[i][0])
        all_pairs.append(opt_pairs[i][1])
    all_pairs = np.unique(all_pairs)
    final_data = data[all_pairs]

    full_df = final_data.dropna().copy()
    dict_df_res_bay = {}
    dict_mx_bnf_bay = {}
    dict_bnf_bay = {}
    dict_full_bnf_bay = {}
    for elem in opt_pairs:
        this_df = full_df[elem]
        name = f'{elem[0]}_{elem[1]}'
        df_train = this_df[100:950]
        df_test = this_df[950:]
        get_spread_train = GetSpread(df_train, elem[0], elem[1])
        df_train = get_spread_train.update_estim_beta(50)
        space_param = {
            "const_coeff": hp.choice("const_coeff", np.arange(0.8, 2.2, 0.01)),
            "const_std": hp.choice("const_std", np.arange(0, 1, 0.01)),
            "roll_vol": hp.choice("roll_vol", np.arange(20, 60, 1, dtype=int)),
            "coeff_exit": hp.choice("coeff_exit", np.arange(-1, 1, 0.01)),
            "data": df_train.reset_index(),
            "tick1": elem[0],
            "tick2": elem[1]
        }
        opti = OptimizeSignals.multi_bay_opti(OptimizeSignals.objective, space_param, 1)
        get_spread_test = GetSpread(df_test, elem[0], elem[1])
        df_test = get_spread_test.update_estim_beta(50)
        df_res, mx_bnf, bnf = Backtest.simple_backtest(df_test.reset_index(), elem[0], elem[1], params=[
            opti.sort_values('SCORE').reset_index()['const_coeff'][0],
            opti.sort_values('SCORE').reset_index()['const_std'][0],
            int(opti.sort_values('SCORE').reset_index()['roll_vol'][0]),
            opti.sort_values('SCORE').reset_index()['coeff_exit'][0]])
        print(f'Pnl on test data : {round(mx_bnf, 2)} for strat {name}')
        print(f"ADF test on df_test for {name} : {ts.adfuller(df_res['misp_ptf'].dropna())[1]}")
        dict_df_res_bay[name] = df_res
        dict_mx_bnf_bay[name] = mx_bnf
        dict_bnf_bay[name] = bnf
        dict_full_bnf_bay[name] = df_res['pnl_strat'].values

    df_cac40 = yf.download('^FCHI', start=start, end=end)['Adj Close'][1766:].reset_index()
    df_cac40['daily_ret'] = df_cac40['Adj Close'].pct_change()
    df_cac40['cum_ret'] = (1 + df_cac40['daily_ret']).cumprod()*100 - 100
    full_return_ptf_bay = pd.DataFrame.from_dict(dict_full_bnf_bay, orient='index').transpose().dropna()
    results_df_bay = full_return_ptf_bay / len(full_return_ptf_bay.columns)
    final_res_bay = results_df_bay.sum(axis=1)
    plt.figure(figsize=(15, 10))
    plt.title('Comparison of cumulative returns between the strategy and the CAC40 target index')
    plt.plot(final_res_bay[50:].to_list(), label='Strategy')
    plt.plot(df_cac40['cum_ret'].to_list(), label='CAC40 index')
    plt.legend()
    plt.show()


def main_nrg():
    oil_products = ['CL=F', 'HO=F', 'NG=F', 'RB=F', 'BZ=F']
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime(2023, 6, 1)
    get_data = GetData(start, end)
    data = get_data.dl_close_data(oil_products)
    data.dropna(inplace=True)
    pairs = list(combinations(oil_products, 2))
    all_pairs = []
    for pair in pairs:
        if pair[::-1] not in all_pairs:
            all_pairs.append(list(pair))

    opt_pairs = []
    for pair in all_pairs:
        asset1 = data[pair[0]]
        asset2 = data[pair[1]]
        results = sm.OLS(asset1, asset2).fit()
        predict = results.predict(asset2)
        error = asset1 - predict
        adf_test = ts.adfuller(error)
        if adf_test[1] < 0.05:
            opt_pairs.append(pair)

    full_df = data.copy()
    dict_df_res_bay = {}
    dict_mx_bnf_bay = {}
    dict_bnf_bay = {}
    dict_full_bnf_bay = {}
    for elem in opt_pairs:
        this_df = full_df[elem]
        name = f'{elem[0]}_{elem[1]}'
        df_train = this_df[1000:1700]
        df_test = this_df[1700:]
        get_spread_train = GetSpread(df_train, elem[0], elem[1])
        df_train = get_spread_train.update_estim_beta(50)
        space_param = {
            "const_coeff": hp.choice("const_coeff", np.arange(1, 2.5, 0.01)),
            "const_std": hp.choice("const_std", np.arange(0, 1, 0.01)),
            "roll_vol": hp.choice("roll_vol", np.arange(20, 60, 1, dtype=int)),
            "coeff_exit": hp.choice("coeff_exit", np.arange(-1, 1, 0.01)),
            "data": df_train.reset_index(),
            "tick1": elem[0],
            "tick2": elem[1]
        }
        opti = OptimizeSignals.multi_bay_opti(OptimizeSignals.objective, space_param, 1)
        get_spread_test = GetSpread(df_test, elem[0], elem[1])
        df_test = get_spread_test.update_estim_beta(50)
        df_res, mx_bnf, bnf = Backtest.simple_backtest(df_test.reset_index(), elem[0], elem[1], params=[
            opti.sort_values('SCORE').reset_index()['const_coeff'][0],
            opti.sort_values('SCORE').reset_index()['const_std'][0],
            int(opti.sort_values('SCORE').reset_index()['roll_vol'][0]),
            opti.sort_values('SCORE').reset_index()['coeff_exit'][0]])
        print(f'Pnl on test data : {round(mx_bnf, 2)} for strat {name}')
        print(f"ADF test on df_test for {name} : {ts.adfuller(df_res['misp_ptf'].dropna())[1]}")
        dict_df_res_bay[name] = df_res
        dict_mx_bnf_bay[name] = mx_bnf
        dict_bnf_bay[name] = bnf
        dict_full_bnf_bay[name] = df_res['pnl_strat'].values

    full_return_ptf_bay = pd.DataFrame.from_dict(dict_full_bnf_bay, orient='index').transpose().dropna()
    results_df_bay = full_return_ptf_bay / len(full_return_ptf_bay.columns)
    final_res_bay = results_df_bay.sum(axis=1)
    plt.figure(figsize=(15, 10))
    plt.title('Cumulative returns of the strategy for a list of energy products')
    plt.plot(final_res_bay[50:].to_list())
    plt.show()

    for elem in opt_pairs:
        name = elem[0] + '_' + elem[1]
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 1, 1)
        plt.plot(dict_df_res_bay[name]['misp_ptf'].dropna())
        plt.annotate(dict_mx_bnf_bay[name], (200, 2))
        plt.plot(dict_df_res_bay[name]['seuil_signal'].dropna())
        plt.plot(-dict_df_res_bay[name]['seuil_signal'].dropna())
        plt.plot(dict_df_res_bay[name]['exit_signal'].dropna())
        plt.plot(-dict_df_res_bay[name]['exit_signal'].dropna())
        plt.subplot(2, 1, 2)
        plt.plot(full_return_ptf_bay[name].dropna())
        plt.show()


if __name__ == '__main__':
    main_nrg()


