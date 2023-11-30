library(tidyverse)

# Preprocessing ----

experiment <- read_csv("https://raw.githubusercontent.com/yjunechoe/Semantic-Persistence/master/processed.csv")
norming <- read_csv("https://raw.githubusercontent.com/yjunechoe/Semantic-Persistence/master/data/Item_norms.csv")

speeded_comprehension <- experiment %>% 
  filter(Type == "Critical") %>% 
  select(Item, PitchAccent = Cond, Subject, Accuracy) %>% 
  mutate(Subject = fct_anon(Subject, "S")) %>% 
  left_join(norming, by = "Item") %>% 
  rename(SemanticFit = plaus_bias, TransitivityBias = trans_bias) %>% 
  mutate(PitchAccent = ifelse(PitchAccent == "Subject", "Noun", "Verb")) %>% 
  mutate(PitchAccent = ifelse(PitchAccent == "Verb", 1, -1)) %>% 
  select(Accuracy, Subject, Item, PitchAccent, SemanticFit, TransitivityBias) %>% 
  arrange(Subject, Item, PitchAccent) %>% 
  filter(!is.na(Accuracy))

# Write ----
write_csv(speeded_comprehension, "speeded_comprehension.csv")

# Model ----
library(lme4)
model <- glmer(
  Accuracy ~ PitchAccent + SemanticFit + TransitivityBias +
             (1 + PitchAccent || Subject) +
             (1 | Item),
  data = speeded_comprehension,
  family = "binomial"
)
summary(model)

## Theoretical maximal model: will take long and not converge - run at your own risk!
system.time({
  model_max <- glmer(
    Accuracy ~ PitchAccent * SemanticFit * TransitivityBias +
               (1 + PitchAccent * SemanticFit * TransitivityBias | Subject) +
               (1 + PitchAccent | Item),
    data = speeded_comprehension,
    family = "binomial"
  )
})

# Read in Julia model (not in Colab notebook) ----
julia_model <- readRDS("julia_model.rds")
summary(julia_model)

## Ex: report model
sjPlot::tab_model(julia_model)

## Ex: plot predictions
marginaleffects::plot_predictions(
  julia_model,
  condition = list(SemanticFit = -2:2, TransitivityBias = -2:2, PitchAccent = unique),
) +
  scale_y_continuous(limits = 0:1)
