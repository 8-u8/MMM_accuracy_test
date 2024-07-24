# load packages
# https://facebookexperimental.github.io/siMMMulator/docs/step_by_step_guide

library(siMMMulator)
set.seed(1234)

# define campaign settings
my_variables <- step_0_define_basic_parameters(

  # since it is defact data volume to implement MMMs, 
  # we assume the 2 years data are provided.
  years = 2,

  # 3 channels impressions
  channels_impressions = c("Channel_01", "Channel_02", "Channel_03"),
  
  # 1 channels click.
  channels_clicks = c("Search"),
  
  # we assume campaigns coccur every 2 weeks and last 2 weeks.
  frequency_of_campaigns = 2,
  
  # we assume CVR for each media are same, for simply.
  true_cvr = c(
    0.10,  # CVR of Channel_01 is the smallest in our media strategy.
    0.10,  # CVR of Channel_02
    0.10,  # CVR of Channel_03
    0.10   # CVR of Search is the highest in our media strategy.
  ),

  # we assume unit of revenue is Japanese yen.
  revenue_per_conv = 1000, 
  # we start from 2023.
  start_date = "2023/1/1"
)

# generate baseline
df_baseline <- step_1_create_baseline(
  my_variables = my_variables,

  # baseline revenue not due to ads or any trends.
  # we assume unit of revenue is Japanese yen
  base_p = 500000,

  # we assume our revenue has no trend(stable).
  trend_p = 0,

  # seasonality
  temp_var = 0.3,

  # average seasonality mean
  temp_coef_mean = 50000,
  temp_coef_sd = 5000,
  
  # error term
  error_std = 5000)

optional_step_1.5_plot_baseline_sales(df_baseline = df_baseline)


# generate add spends
channel_proportion <- c(
  0.45, 0.50, # Channel_01
  0.25, 0.30, # Channel_02
  0.15, 0.20  # Channel_03
)

df_ads_step2 <- step_2_ads_spend(
    my_variables = my_variables,

    # we assume campaign spend as much as revenue.
    campaign_spend_mean = 500000,
    campaign_spend_std = 10000,
    max_min_proportion_on_each_channel <- channel_proportion
)
df_ads_step2
optional_step_2.5_plot_ad_spend(df_ads_step2)

# generate media variables as independent variables.
df_ads_step3 <- step_3_generate_media(
  # initial input
  my_variables = my_variables,
  df_ads_step2 = df_ads_step2,

  # Cost par impression each media.
  # we assume its unit is Japanese yen.
  true_cpm = c(
      20000,  # Channel_01 is high cost par impression.
      1000,  # Channel_02 
      5000,   # Channel_03 is low cost par impression of all media
      NA),
  # Cost par click for search media
  true_cpc = c(
      NA,
      NA,
      NA,
      3 # 1 conversion costs 3 yen.
    ),

  # noise parameter(normal distribution.)
  mean_noisy_cpm_cpc = c(
      100,    # Channel_01
      100,    # Channel_02
      50,     # Channel_03
      10      # Clicks

    ),
  std_noisy_cpm_cpc = c(
      50,   # Channel_01
      15,   # Channel_02
      10,   # Channel_03
       5    # Clicks
    )
)

# generate noisy cvr
df_ads_step4 <- step_4_generate_cvr(
  # init
  my_variables = my_variables,
  df_ads_step3 = df_ads_step3,

  # noise of conversion rate
  # for simple, we assume all same noises.
  mean_noisy_cvr = c(
    0,      # Channel_01
    0, # Channel_02
    0, # Channel_03
    0       # Click
    ), 
  std_noisy_cvr = c(
    0.01, # Channel_01
    0.01, # Channel_02
    0.01, # Channel_03
    0.01  # Click
  )
)

# pivot data from long form to wide form
df_ads_step5a_before_mmm <- step_5a_pivot_to_mmm_format(
    my_variables = my_variables,
    df_ads_step4 = df_ads_step4
    )

# apply adstock function.
df_ads_step5b <- step_5b_decay(
      my_variables = my_variables,
      df_ads_step5a_before_mmm = df_ads_step5a_before_mmm,
      # for simple, we assume all lambda is same.
      # in Robyn is formalize as theta, maybe.
      true_lambda_decay = c(
        0.3, # Channel_01
        0.3, # Channel_02
        0.3, # Channel_03
        0.3  # Clicks
      )
    )

# apply saturation for media efficiency.
df_ads_step5c <- step_5c_diminishing_returns(
      my_variables = my_variables,
      df_ads_step5b = df_ads_step5b,

      # we assume all media have same satulation rate.
      # satulation equation is:
      # x_decay_{t, media}^alpha / (x_decay_{t, media}^alpha + gamma^alpha)
      # if alpha = 1, the calculation above is more simplified.
      # x_decay_{t, media} / (x_decay_{t, media} + gamma)
      alpha_saturation = c(
        2, # Channel_01
        2, # Channel_02
        2, # Channel_03
        2  # Click
      ),
      gamma_saturation = c(
        0.5, # Channel_01
        0.5, # Channel_02
        0.5, # Channel_03
        0.5  # Click
      )
    )

df_ads_step6 <- step_6_calculating_conversions(
      my_variables = my_variables,
      df_ads_step5c = df_ads_step5c
    )

df_ads_step7 <- step_7_expanded_df(
      my_variables = my_variables,
      df_ads_step6 = df_ads_step6,
      df_baseline = df_baseline
    )

# negative value occured, for simulate, convert absolute values.
df_ads_step7$conv_Search <- abs(df_ads_step7$conv_Search)

step_8_calculate_roi(
      my_variables = my_variables,
      df_ads_step7 = df_ads_step7
      )

list_of_df_final <- step_9_final_df(
        my_variables = my_variables,
        df_ads_step7 = df_ads_step7
      )

plot(list_of_df_final[[2]]$spend_Search, type = "l")
write.csv(list_of_df_final[[2]], "./input/simulated_data.csv", row.names = FALSE,
 fileEncoding="utf-8")