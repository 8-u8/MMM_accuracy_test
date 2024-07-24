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

model_no <- "1_618_4"
ExportedModel <- robyn_write(InputCollect, OutputCollect,
                             model_no, export = FALSE)

# if each parameter is closer to simulated data parameters,
# Robyn is accurate the media parameters.
print(ExportedModel)
