# %%
import pandas as pd
import numpy as np

import jax.numpy as jnp
from sklearn.metrics import mean_absolute_percentage_error

from lightweight_mmm import preprocessing

from lightweight_mmm import utils
from lightweight_mmm import lightweight_mmm
from lightweight_mmm import plot
# %%
sim_data = pd.read_csv("./input/simulated_data.csv")
sim_data = sim_data.dropna(axis=0)

sim_data.loc[sim_data['clicks_Search']<0,'clicks_Search'] = 0

# %% config: set variables
spend_columns = ['spend_Channel_01', 'spend_Channel_02',
                 'spend_Channel_03', 'spend_Search']
media_columns = ['impressions_Channel_01',
                 'impressions_Channel_02',
                 'impressions_Channel_03',
                 'clicks_Search']
sales = ['total_revenue']

spend_data = sim_data[spend_columns]
media_data = sim_data[media_columns]
sales_data = sim_data[sales]

# %% convert jax format
cost_jax = jnp.array(
    spend_data.values
)

media_data_jax = jnp.array(
    media_data.values
)

sales_jax = jnp.array(
    sales_data.values
)

# %% train test split
split_point = len(media_data_jax) - 20

media_data_train = media_data_jax[:split_point]
media_data_test = media_data_jax[split_point:]

target_train = sales_jax[:split_point]
target_test = sales_jax[split_point:]

costs_train = cost_jax[:split_point].sum(axis=0)
costs_test = cost_jax[split_point:].sum(axis=0)
media_names = media_data.columns

# %% scaling
media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
costs_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)

media_data_train = media_scaler.fit_transform(media_data_train)
media_data_test = media_scaler.transform(media_data_test)

target_train = target_scaler.fit_transform(target_train)
target_test = target_scaler.transform(target_test)

costs_train = costs_scaler.fit_transform(costs_train)
cost_test = costs_scaler.transform(costs_test)

# %% find best model
# no scaling fitting to compare to Robyn coefs
model_name = 'hill_adstock'

mmm = lightweight_mmm.LightweightMMM(model_name=model_name)
mmm.fit(
    media=media_data_train,
    media_names=['Channel_01', 'Channel_02', 'Channel_03', 'Search'],
    media_prior=costs_train,
    target=target_train,
    number_warmup=1000,
    number_samples=10_000,
    number_chains=5,
    weekday_seasonality=True,
    seasonality_frequency=52,
    seed=42
)

#%%
pred = mmm.predict(
    media=media_data_test,
    target_scaler=target_scaler
)

# %% 
mmm.print_summary()

# %%
media_effect_hat, roi_hat = mmm.get_posterior_metrics(
    cost_scaler=costs_scaler,
    target_scaler=target_scaler
)


# %%
utils.save_model(mmm, "./output/lightweight_mmm_20240725_model.pkl")

# how to get media coefs mean?
pd.DataFrame(
    {
        'medial_name': mmm.media_names,
        'true_ROI': [2.90889964244781, 67.6262704285043,
                    14.2904166234131, 5.25751787085714],
        'lmmm_ROI_hat': roi_hat.mean(axis=0),
        'robyn_ROI_hat': [5.557, 20.778, 16.09, 0.494]
    }
)

# 	medial_name	true_ROI	lmmm_ROI_hat    robyn_ROI_hat
# 	Channel_01	2.908900	17.849827   	5.557
# 	Channel_02	67.626270	25.900499   	20.778
# 	Channel_03	14.290417	34.662693   	16.090
#   	Search	5.257518	51.653549   	0.494