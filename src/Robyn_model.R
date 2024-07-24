library(Robyn)
library(tidyverse)
library(reticulate)

system("pipenv --venv", inter = TRUE)
# reticulate::use_virtualenv(venv, required = TRUE) 
output <- "./output/"

sim_data <- read_csv("./input/simulated_data.csv") |> 
  na.omit() |> 
  dplyr::mutate(
    clicks_Search = if_else(clicks_Search <= 0, 0, clicks_Search)
  )
holiday_data <- read_csv("./input/generated_holidays.csv") |> 
  na.omit()

InputCollect <- Robyn::robyn_inputs(
  dt_input = sim_data,
  dt_holidays = holiday_data,
  # assume Japanese holiday
  prophet_country = "JP",
  date_var = "DATE",
  dep_var = "total_revenue",
  dep_var_type = "revenue",
  prophet_vars = c("trend", "season"),
  paid_media_spends = c("spend_Channel_01",
                        "spend_Channel_02",
                        "spend_Channel_03",
                        "spend_Search"),
  paid_media_vars = c("impressions_Channel_01",
                      "impressions_Channel_02",
                      "impressions_Channel_03",
                      "clicks_Search"),
  adstock = "geometric"
)


hyperparameters <- list(
  spend_Channel_01_alphas = c(0.5, 4.0), # simulated: 2
  spend_Channel_01_gammas = c(0.3, 1.0), # simulated: 0.5
  spend_Channel_01_thetas = c(0.1, 0.5), # simulated: 0.3

  spend_Channel_02_alphas = c(0.5, 3.0), # simulated: 2
  spend_Channel_02_gammas = c(0.3, 1.0), # simulated: 0.5
  spend_Channel_02_thetas = c(0.1, 0.5), # simulated: 0.3
  
  spend_Channel_03_alphas = c(0.5, 4.0), # simulated: 2
  spend_Channel_03_gammas = c(0.3, 1.0), # simulated: 0.5
  spend_Channel_03_thetas = c(0.1, 0.5), # simulated: 0.3
  
  spend_Search_alphas = c(0.5, 4.0),  # simulated: 2
  spend_Search_gammas = c(0.3, 1.0),  # simulated: 0.5
  spend_Search_thetas = c(0, 0.5),    # simulated: 0.3
  train_size = c(0.8, 0.9)
)


InputCollect <- robyn_inputs(InputCollect = InputCollect,
   hyperparameters = hyperparameters)
print(InputCollect)

OutputModels <- robyn_run(
  InputCollect = InputCollect, # feed in all model specification
  cores = NULL, # NULL defaults to (max available - 1)
  iterations = 10000, # 2000 recommended for the dummy dataset with no calibration
  trials = 5, # 5 recommended for the dummy dataset
  ts_validation = TRUE, # 3-way-split time series for NRMSE validation.
  add_penalty_factor = FALSE # Experimental feature. Use with caution.
)
print(OutputModels)

OutputCollect <- robyn_outputs(
  InputCollect, OutputModels,
  pareto_fronts = "auto", # automatically pick how many pareto-fronts to fill min_candidates (100)
  # min_candidates = 100, # top pareto models for clustering. Default to 100
  # calibration_constraint = 0.1, # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  export = TRUE, # this will create files locally
  plot_folder = output, # path for plots exports and files creation
  plot_pareto = TRUE # Set to FALSE to deactivate plotting and saving model one-pagers
)
print(OutputCollect)

model_no <- "4_473_5"
ExportedModel <- robyn_write(InputCollect, OutputCollect,
                             model_no, export = FALSE)

# if each parameter is closer to simulated data parameters,
# Robyn is accurate the media parameters.
print(ExportedModel)

# [1] "roi of Channel_01 is: 2.90889964244781"
# [1] "roi of Channel_02 is: 67.6262704285043"
# [1] "roi of Channel_03 is: 14.2904166234131"
# [1] "roi of Search is     : 5.25751787085714"

# Summary Values on Selected Model:
#           variable    coef decompPer decompAgg    ROI mean_response mean_spend
# 1      (Intercept)  1.714M     3.86%   178.22M      -             -          -
# 2            trend   0.518    51.91%    2.397B      -             -          -
# 3           season    0.35     0.00%   175.49K      -             -          -
# 4 spend_Channel_01  8.313M    10.40%   480.53M  5.557        4.629M    831.52K
# 5 spend_Channel_02 22.272M    22.59%    1.043B 20.788       10.016M    482.51K
# 6 spend_Channel_03 10.396M    11.09%    512.4M  16.09        4.931M     306.2K
# 7     spend_Search 173.93K     0.14%    6.602M  0.494       77.849K    128.42K

# Hyper-parameters:
#     Adstock: geometric   2       0.5        0.3
#            channel    alphas    gammas    thetas
# 1 spend_Channel_01 0.6428098 0.3599642 0.4958000
# 2 spend_Channel_02 2.9734200 0.9671084 0.1004090
# 3 spend_Channel_03 1.0330570 0.9809159 0.1006910
# 4     spend_Search 3.9117300 0.8677973 0.4999715