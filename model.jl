# Setup
using CSV
using DataFrames
using MixedModels

speeded_comprehension = CSV.read("speeded_comprehension.csv", DataFrame)


# 1) Model fitting ----

## Replicating model from R
model = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
                      zerocorr(1 + PitchAccent | Subject) +
                      (1| Item)),
  speeded_comprehension,
  Bernoulli()
)

## R model with max RE
model_max = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
                      (1 + PitchAccent + SemanticFit + TransitivityBias | Subject) +
                      (1 + PitchAccent | Item)),
  speeded_comprehension,
  Bernoulli();
  progress = false
)
issingular(model_max)
VarCorr(model_max)
MixedModels.rePCA(model_max)

## Model with best RE
model_final = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
                      (1 + PitchAccent | Subject) +
                      (1 | Item)),
  speeded_comprehension,
  Bernoulli()
)
issingular(model_final)
VarCorr(model_final)

## Model including two-way interaction
@time model_interactions = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
                      PitchAccent & SemanticFit +
                      PitchAccent & TransitivityBias +
                      SemanticFit & TransitivityBias +
                      (1 + PitchAccent | Subject) +
                      (1 | Item)),
  speeded_comprehension,
  Binomial()
)
issingular(model_interactions)

## Theoretically maximal model (not in Colab notebook)
@time model_interaction_max = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent * SemanticFit * TransitivityBias +
                      (1 + PitchAccent * SemanticFit * TransitivityBias | Subject) +
                      (1 + PitchAccent | Item)),
  speeded_comprehension,
  Bernoulli()
)
issingular(model_interaction_max)
MixedModels.rePCA(model_interaction_max)


# 2) Model selection & diagnostics ----

## Likelihood ratio test between models with no singular fit:
MixedModels.likelihoodratiotest(model_final, model_interactions)

## Visualization (not in Colab notebook)
using CairoMakie
using MixedModelsMakie
VarCorr(model_final)
caterpillar(model_final, :Subject)
shrinkageplot(model_final, :Subject)

# 3) Transferring model ----

## Write out model estimates
coef_table = coeftable(model_final)
coef_datatable = DataFrame(coef_table)
CSV.write("model_output.csv", coef_datatable)

## Convert Julia MixedModels model to R lmer model (not in Colab notebook)
using RCall
using JellyMe4

julia_model = (model_final, speeded_comprehension);
@rput julia_model;
R"julia_model"
R"saveRDS(julia_model, 'julia_model.rds')"
