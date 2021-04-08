library("checkpoint")
checkpoint("2021-01-01") # all packages in the same R project will be taken from this date
library("mlr3")
library("mlr3viz")
library("mlr3verse")
library("tidyverse")
library("precrec")

####################################################################
# Import dataframe with both genotype info and metabolite abundances
# Outliers removed: IL26
####################################################################

peaks <- read.csv("01.metabolic_candidate_selection/genotype_and_peak_data.csv")

sample_info <- read.csv("01.metabolic_candidate_selection/sample_genotype_phenotype.csv")


# finding the row index of IL27_6
row_index_of_outliers = c(
  match(x = "IL27_6", table = sample_info$sample),
  match(x = "s_ch_1", table = sample_info$sample)
)

# remove this line from the peaks data
peaks = peaks[-row_index_of_outliers,]

# remove it from the sample to genotype df
sample_info <- sample_info[-row_index_of_outliers,]


df <- bind_cols(sample_info, peaks) %>% 
  select(- sample, - genotype) # not required for ML classification



#########################
# Define Task and Learner
#########################
task_metabolites <- TaskClassif$new(id = "peaks", 
                                    backend = df, 
                                    target = "phenotype", 
                                    positive = "resistant")


rf_learner = lrn(.key = "classif.ranger", 
              id = "rf", 
              predict_type = "response",
              importance = "permutation",
              num.trees = 10000)



# filter_ranger = flt("importance", learner = rf_learner)
# filter_ranger$calculate(task_metabolites)
# 
# feature_importances = as.data.table(filter_ranger) %>% 
#   arrange(desc(score)) %>% 
#   head(n = 10)
# 
# ggplot(feature_importances, aes(x = feature, y = score)) + 
#   geom_histogram(stat = "identity") +
#   coord_flip()

##############################################################################################
# Cross-validation (to improve generalisation capacity and test stability of different splits)
##############################################################################################

# parameters for each methods
# as.data.table(mlr_resamplings)
cv_strategy = rsmp(.key = "cv", folds = 10)
cv_strategy$instantiate(task_metabolites)

results_from_cv_models = resample(task = task_metabolites, 
                     learner = rf_learner, 
                     resampling = cv_strategy, 
                     store_models = TRUE)

measures_performance <- list("classif.acc","classif.bacc")
results_from_cv_models$aggregate(measures_performance)   # accuracy
results_from_cv_models$aggregate(msr("classif.bacc"))  # balanced accuracy


##############
# Feature selection
##################
results_from_cv_models$filter("importance")

filter = flt("importance", learner = rf_learner)
filter$param_set$values = list(importance = "permutation", num.trees= 10000) 

filter$calculate(task_metabolites)
filter

################
# Compare models
################


learners = lrns(c("classif.rpart", "classif.ranger"), predict_type = "prob")

bm_design = benchmark_grid(
  tasks = task_metabolites,
  learners = learners,
  resamplings = rsmp("cv", folds = 10)
)
bmr = benchmark(bm_design, store_models = TRUE)

# as.data.table(mlr_measures) %>% filter(task_type == "classif")
measures = msrs(c("classif.ce", "classif.acc"))
performances = bmr$aggregate(measures)


autoplot(bmr, type = "prc")
