# Setup

## Load packages
using CSV
using DataFrames
using MixedModels

# Read data
speeded_comprehension = CSV.read("speeded_comprehension.csv", DataFrame)

## Decimal printing options
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.3f", f)

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

## R model but with max RE
model_max = fit(
  MixedModel,
  @formula(Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
                      (1 + PitchAccent + SemanticFit + TransitivityBias | Subject) +
                      (1 + PitchAccent | Item)),
  speeded_comprehension,
  Bernoulli()
)
issingular(model_max)
VarCorr(model_max)

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

## Model including two-way interactions in FE
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
fm_max = @formula(
  Accuracy ~ PitchAccent * SemanticFit * TransitivityBias +
             (1 + PitchAccent * SemanticFit * TransitivityBias | Subject) +
             (1 + PitchAccent | Item)
)

@time model_interaction_max = fit(
  MixedModel, fm_max, speeded_comprehension, Bernoulli()
)
length(model_interaction_max.θ) # unicode!
issingular(model_interaction_max)
MixedModels.rePCA(model_interaction_max)

@time model_interaction_max_fast = fit(
  MixedModel, fm_max, speeded_comprehension, Bernoulli();
  fast=true # slightly less accurate, but useful for model selection
);
deviance(model_interaction_max_fast)
deviance(model_interaction_max)
MixedModels.rePCA(model_interaction_max_fast)
VarCorr(model_interaction_max_fast)

# 2) Model selection & diagnostics ----

## Likelihood ratio test between models with no singular fit:
MixedModels.likelihoodratiotest(model_final, model_interactions)

## Visualization (not in Colab notebook)
using CairoMakie
using MixedModelsMakie
VarCorr(model_final).σρ.Subject
caterpillar(model_final, :Subject)
shrinkageplot(model_final, :Subject)

# 3) Transferring model ----

## Write out model estimates
coef_table = coeftable(model_final)
coef_datatable = DataFrame(coef_table)
CSV.write("model_output.csv", coef_datatable)

## Convert Julia MixedModels model to R lmer model (not in Colab notebook)
using RCall

### -- R REPL demo --

### Model conversion
using JellyMe4
julia_model = (model_final, speeded_comprehension);

@rput julia_model;
R"julia_model"
R"saveRDS(julia_model, 'julia_model.rds')"
