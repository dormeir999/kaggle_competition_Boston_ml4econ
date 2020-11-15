## ----setup, include=FALSE-------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## ---- echo=TRUE, warning=FALSE, results=FALSE-----------------------------------------------------------------------------
#install.packages("dplyr", repos = "http://cran.us.r-project.org", dependencies = TRUE)
#install.packages("devtools", repos = "http://cran.us.r-project.org")
library(devtools)
#devtools::install_github("itamarcaspi/experimentdatar")
library(experimentdatar)
data(vouchers)
dat <- vouchers


## ---- echo=TRUE,message=FALSE, warning=FALSE, results=FALSE---------------------------------------------------------------
dat_table3<-subset(dat, (BOG95SMP==1 | BOG97SMP==1 | JAM93SMP==1))
dat_table3<-dat_table3[c("ID","VOUCH0","BOG95SMP","BOG97SMP","JAM93SMP","AGE2","SCYFNSH","INSCHL","FINISH6","FINISH7","FINISH8","PRSCHA_1","PRSCHA_2","PRSCH_C","REPT6","REPT","NREPT","TOTSCYRS","USNGSCH","SVY","HSVISIT","DJAMUNDI","PHONE","AGE","SEX2","DBOGOTA","D1993","D1995","D1997","DMONTH1","DMONTH2","DMONTH3","DMONTH4","DMONTH5","DMONTH6","DMONTH7","DMONTH8","DMONTH9","DMONTH10","DMONTH11","DMONTH12","STRATA1","STRATA2","STRATA3","STRATA4","STRATA5","STRATA6","STRATAMS","DAREA1","DAREA10","DAREA11","DAREA12","DAREA13","DAREA14","DAREA15","DAREA16","DAREA17","DAREA18","DAREA19","DAREA2","DAREA3","DAREA4","DAREA5","DAREA6","DAREA7","DAREA8","DAREA9","SEX_MISS")]
sort(dat_table3$VOUCH0)


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
Bogota_95_Losers <- subset(dat_table3, VOUCH0==0 & BOG95SMP==1)
#detach(dat_table3)
attach(Bogota_95_Losers)
df <- data.frame()
options(digits=3)
means_sd_n <- c(round(mean(SCYFNSH,na.rm = TRUE),digits = 3),paste("(",round(sd(SCYFNSH,na.rm = TRUE),digits = 3),")"),round(mean(INSCHL,na.rm = TRUE),digits = 3),paste("(",round(sd(INSCHL,na.rm = TRUE),digits = 3),")"),round(mean(FINISH6,na.rm = TRUE),digits = 3),paste("(",round(sd(FINISH6,na.rm = TRUE),digits = 3),")"),round(mean(FINISH7,na.rm = TRUE),digits = 3),paste("(",round(sd(FINISH7,na.rm = TRUE),digits = 3),")"),round(mean(FINISH8,na.rm = TRUE),digits = 3),paste("(",round(sd(FINISH8,na.rm = TRUE),digits = 3),")"),round(mean(PRSCHA_1,na.rm = TRUE),digits = 3),paste("(",round(sd(PRSCHA_1,na.rm = TRUE),digits = 3),")"),round(mean(PRSCHA_2,na.rm = TRUE),digits = 3),paste("(",round(sd(PRSCHA_2,na.rm = TRUE),digits = 3),")"),round(mean(PRSCH_C,na.rm = TRUE),digits = 3),paste("(",round(sd(PRSCH_C,na.rm = TRUE),digits = 3),")"),round(mean(REPT6,na.rm = TRUE),digits = 3),paste("(",round(sd(REPT6,na.rm = TRUE),digits = 3),")"),round(mean(NREPT,na.rm = TRUE),digits = 3),paste("(",round(sd(NREPT,na.rm = TRUE),digits = 3),")"),round(mean(TOTSCYRS,na.rm = TRUE),digits = 3),paste("(",round(sd(TOTSCYRS,na.rm = TRUE),digits = 3),")"),round(mean(STRATAMS,na.rm = TRUE),digits = 3),paste("(",round(sd(STRATAMS,na.rm = TRUE),digits = 3),")"),round(mean(USNGSCH,na.rm = TRUE),digits = 3),paste("(",round(sd(USNGSCH,na.rm = TRUE),digits = 3),")"),nrow(Bogota_95_Losers))
df <- means_sd_n
table_3<-`.rowNamesDF<-`(df,make.names=TRUE,c("Highest grade
completed","sd","Currently in school","sd","Finished 6th grade","sd","Finished 7th grade
(excludes Bogota´ 97)","sd","Finished 8th grade
(excludes Bogota´ 97)","sd","Started 6th grade in
private","sd","Started 7th grade in
private","sd","Currently in private
school","sd","Repetitions of 6th grade","sd","Ever repeated after
lottery","sd","Years in school since
lottery","sd","Total repetitions since
lottery","sd","Using any scholarship
in survey year","sd","Sample Size"))
table_3 <- data.frame(table_3)
# Rename the column:
detach(Bogota_95_Losers)


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
#install.packages("car", repos = "http://cran.us.r-project.org")
library(car)
attach(dat_table3)
# The list of dependents (FINISH7-8 will be estimated again excluding bogota 97):
df_dependents <- data.frame(SCYFNSH,INSCHL,FINISH6,FINISH7,FINISH8,PRSCHA_1,PRSCHA_2,PRSCH_C,REPT6,REPT,NREPT,TOTSCYRS,USNGSCH)
# The new table column:
table_3 <- cbind(table_3, "Combined basic controls"=0)
table_3 <- data.frame(table_3)
options(digits=3)
# Run all models:
for (i in 1:length(df_dependents)){
  model <- lm(formula = df_dependents[,i] ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS, data = dat_table3)
  # The estimated effect of Winning a voucher on the dependent:
  vouch0_effect <- model$coefficients["VOUCH0"]
  vouch0_effect <- round(vouch0_effect, digits = 3)
  table_3[i*2-1,"Combined.basic.controls"] <- vouch0_effect
  # The heteroscedasticity-corrected covariance matrix:
  vcov <- data.frame(hccm(model, type = "hc0"))
  # Obtain the standart error of the dependent, as they are given by the square root of the element in the main diagonal of the matrix:
  vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
  vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
  table_3[i*2,"Combined.basic.controls"] <- vouch0_se
}
table_3["Sample.Size","Combined.basic.controls"] <- nobs(model)

## Calculating FINISH7-8 from the Bogota97 excluded data subset:

# FINISH7:
Bogota_97_excluded <- subset(dat_table3, (BOG95SMP==1 | JAM93SMP))
model <- lm(formula = FINISH7 ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS, data = Bogota_97_excluded)
vouch0_effect <- model$coefficients["VOUCH0"]
vouch0_effect <- round(vouch0_effect, digits = 3)
table_3["Finished.7th.grade..excludes.Bogota..97.","Combined.basic.controls"] <- vouch0_effect
vcov <- data.frame(hccm(model, type = "hc0"))
vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
table_3["sd.3","Combined.basic.controls"] <- vouch0_se
# FINISH8:
model <- lm(formula = FINISH8 ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS, data = Bogota_97_excluded)
vouch0_effect <- model$coefficients["VOUCH0"]
vouch0_effect <- round(vouch0_effect, digits = 3)
table_3["Finished.8th.grade..excludes.Bogota..97.","Combined.basic.controls"] <- vouch0_effect
vcov <- data.frame(hccm(model, type = "hc0"))
vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
table_3["sd.4","Combined.basic.controls"] <- vouch0_se
# Rename the column:
colnames(table_3)[1] <- "Bogota 95 Losers means (1)"
colnames(table_3)[2] <- "Combined - Basic controls (5)"


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
# The list of dependents (FINISH7-8 will be estimated again excluding bogota 97):
df_dependents <- data.frame(SCYFNSH,INSCHL,FINISH6,FINISH7,FINISH8,PRSCHA_1,PRSCHA_2,PRSCH_C,REPT6,REPT,NREPT,TOTSCYRS,USNGSCH)
# The new table column:
table_3 <- cbind(table_3, "temporary_name"=0)
options(digits=3)
# Run all models:
for (i in 1:length(df_dependents)){
  model <- lm(formula = df_dependents[,i] ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 +  SEX_MISS + DAREA1 + DAREA2 + DAREA3+ DAREA4 + DAREA5 + DAREA6 + DAREA7 + DAREA8 + DAREA9 + DAREA10 + DAREA11 + DAREA12 + DAREA13 + DAREA14 + DAREA15 + DAREA16 + DAREA17 + DAREA18 + DAREA19, data = dat_table3)
  # The estimated effect of Winning a voucher on the dependent:
  vouch0_effect <- model$coefficients["VOUCH0"]
  vouch0_effect <- round(vouch0_effect, digits = 3)
  table_3[i*2-1,"temporary_name"] <- vouch0_effect
  # The heteroscedasticity-corrected covariance matrix:
  vcov <- data.frame(hccm(model, type = "hc0"))
  # Obtain the standart error of the dependent, as they are given by      the square root of the element in the main diagonal of the matrix:
  vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
  vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
  table_3[i*2,"temporary_name"] <- vouch0_se
}
table_3["Sample.Size","temporary_name"] <- nobs(model)

## Calculating FINISH7-8 from the Bogota97 excluded data subset:

# FINISH7:
model <- lm(formula = FINISH7 ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS + DAREA1 + DAREA2 + DAREA3+ DAREA4 + DAREA5 + DAREA6 + DAREA7 + DAREA8 + DAREA9 + DAREA10 + DAREA11 + DAREA12 + DAREA13 + DAREA14 + DAREA15 + DAREA16 + DAREA17 + DAREA18 + DAREA19, data = Bogota_97_excluded)
vouch0_effect <- model$coefficients["VOUCH0"]
vouch0_effect <- round(vouch0_effect, digits = 3)
table_3["Finished.7th.grade..excludes.Bogota..97.","temporary_name"] <- vouch0_effect
vcov <- data.frame(hccm(model, type = "hc0"))
vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
table_3["sd.3","temporary_name"] <- vouch0_se
# FINISH8:
model <- lm(formula = FINISH8 ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS + DAREA1 + DAREA2 + DAREA3+ DAREA4 + DAREA5 + DAREA6 + DAREA7 + DAREA8 + DAREA9 + DAREA10 + DAREA11 + DAREA12 + DAREA13 + DAREA14 + DAREA15 + DAREA16 + DAREA17 + DAREA18 + DAREA19, data = Bogota_97_excluded)
vouch0_effect <- model$coefficients["VOUCH0"]
vouch0_effect <- round(vouch0_effect, digits = 3)
table_3["Finished.8th.grade..excludes.Bogota..97.","temporary_name"] <- vouch0_effect
vcov <- data.frame(hccm(model, type = "hc0"))
vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
table_3["sd.4","temporary_name"] <- vouch0_se
# Rename the column:
colnames(table_3)[3] <- "Combined - Basic + 19 barrio controls (6)"


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
detach(dat_table3)
Bogota_95 <- subset(dat_table3, BOG95SMP==1)
attach(Bogota_95)
# The list of dependents (FINISH7-8 will be estimated again excluding bogota 97):
df_dependents <- data.frame(SCYFNSH,INSCHL,FINISH6,FINISH7,FINISH8,PRSCHA_1,PRSCHA_2,PRSCH_C,REPT6,REPT,NREPT,TOTSCYRS,USNGSCH)
# The new table column:
table_3 <- cbind(table_3, "temporary_name"=0)
options(digits=3)
# Run all models:
for (i in 1:length(df_dependents)){
  model <- lm(formula = df_dependents[,i] ~ VOUCH0, data = Bogota_95)
  # The estimated effect of Winning a voucher on the dependent:
  vouch0_effect <- model$coefficients["VOUCH0"]
  vouch0_effect <- round(vouch0_effect, digits = 3)
  table_3[i*2-1,"temporary_name"] <- vouch0_effect
  # The heteroscedasticity-corrected covariance matrix:
  vcov <- data.frame(hccm(model, type = "hc0"))
  # Obtain the standart error of the dependent, as they are given by      the square root of the element in the main diagonal of the matrix:
  vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
  vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
  table_3[i*2,"temporary_name"] <- vouch0_se
}
table_3["Sample.Size","temporary_name"] <- nobs(model)

# Rename the column:
colnames(table_3)[4] <- "Bogota 95 - No controls (2)"


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
# The list of dependents (FINISH7-8 will be estimated again excluding bogota 97):
df_dependents <- data.frame(SCYFNSH,INSCHL,FINISH6,FINISH7,FINISH8,PRSCHA_1,PRSCHA_2,PRSCH_C,REPT6,REPT,NREPT,TOTSCYRS,USNGSCH)
# The new table column:
table_3 <- cbind(table_3, "temporary_name"=0)
options(digits=3)
# Run all models:
for (i in 1:length(df_dependents)){
  model <- lm(formula = df_dependents[,i] ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS, data = Bogota_95)
  # The estimated effect of Winning a voucher on the dependent:
  vouch0_effect <- model$coefficients["VOUCH0"]
  vouch0_effect <- round(vouch0_effect, digits = 3)
  table_3[i*2-1,"temporary_name"] <- vouch0_effect
  # The heteroscedasticity-corrected covariance matrix:
  vcov <- data.frame(hccm(model, type = "hc0"))
  # Obtain the standart error of the dependent, as they are given by      the square root of the element in the main diagonal of the matrix:
  vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
  vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
  table_3[i*2,"temporary_name"] <- vouch0_se
}
table_3["Sample.Size","temporary_name"] <- nobs(model)

# Rename the column:
colnames(table_3)[5] <- "Bogota 95 - Basic controls (3)"


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
# The list of dependents (FINISH7-8 will be estimated again excluding bogota 97):
df_dependents <- data.frame(SCYFNSH,INSCHL,FINISH6,FINISH7,FINISH8,PRSCHA_1,PRSCHA_2,PRSCH_C,REPT6,REPT,NREPT,TOTSCYRS,USNGSCH)
# The new table column:
table_3 <- cbind(table_3, "temporary_name"=0)
options(digits=3)
# Run all models:
for (i in 1:length(df_dependents)){
  model <- lm(formula = df_dependents[,i] ~ VOUCH0 + SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 +  SEX_MISS + DAREA1 + DAREA2 + DAREA3+ DAREA4 + DAREA5 + DAREA6 + DAREA7 + DAREA8 + DAREA9 + DAREA10 + DAREA11 + DAREA12 + DAREA13 + DAREA14 + DAREA15 + DAREA16 + DAREA17 + DAREA18 + DAREA19, data = Bogota_95)
  # The estimated effect of Winning a voucher on the dependent:
  vouch0_effect <- model$coefficients["VOUCH0"]
  vouch0_effect <- round(vouch0_effect, digits = 3)
  table_3[i*2-1,"temporary_name"] <- vouch0_effect
  # The heteroscedasticity-corrected covariance matrix:
  vcov <- data.frame(hccm(model, type = "hc0"))
  # Obtain the standart error of the dependent, as they are given by      the square root of the element in the main diagonal of the matrix:
  vouch0_se <- sqrt(vcov["VOUCH0","VOUCH0"])
  vouch0_se <- paste("(",round(vouch0_se, digits = 3),")")
  table_3[i*2,"temporary_name"] <- vouch0_se
}
table_3["Sample.Size","temporary_name"] <- nobs(model)

# Rename the column:
colnames(table_3)[6] <- "Bogota 95 - Basic + 19 barrio controls (4)"
detach(Bogota_95)


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
table_3 <- table_3[c(25,26,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,17,18,19,20,23,24,21,22,27),]
table_3 <- table_3[c(1,4,5,6,2,3)]
rownames(table_3) <- c("Using any scholarship","in survey  year","Started 6th grade ","in private","Started 7th grade"," in private","Currently in ","private school","Highest grade","completed","Currently in","school","Finished 6th","grade","Finished 7th","grade (excludes Bogota´ 97)","Finished 8th"," grade (excludes Bogota´ 97)","Repetitions of 6th"," grade","Ever repeated after","lottery","Total repetitions since"," lottery","Years in school since "," lottery ","Sample size")


## ---- echo=FALSE,message=FALSE, warning=FALSE-----------------------------------------------------------------------------
#install.packages("kableExtra", repos = "http://cran.us.r-project.org")
#devtools::install_github("haozhu233/kableExtra")
library(knitr)
#library(kableExtra)
#knitr::kable(table_3, caption = "The replicated table 3", format.args = )
#kable(table_3)
kable(table_3,format = "html" )


## ---- echo=FALSE, fig.cap="The original table 3", out.width = '80%'-------------------------------------------------------
knitr::include_graphics("table_3_paper.PNG")


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
#devtools::install_github("susanathey/causalTree")
library(causalTree)


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
attach(dat_table3)
Y <- "FINISH8"
D <- "VOUCH0"
X <- c("SVY","HSVISIT","DJAMUNDI","PHONE","AGE","SEX2","STRATA1","STRATA2","STRATA3","STRATA4","STRATA5","STRATA6","STRATAMS","DBOGOTA","D1993","D1995","D1997","DMONTH1","DMONTH2","DMONTH3","DMONTH4","DMONTH5","DMONTH6","DMONTH7","DMONTH8","DMONTH9","DMONTH10","DMONTH11","DMONTH12","SEX_MISS","DAREA1","DAREA2","DAREA3","DAREA4","DAREA5","DAREA6","DAREA7","DAREA8","DAREA9","DAREA10","DAREA11","DAREA12","DAREA13","DAREA14","DAREA15","DAREA16","DAREA17","DAREA18","DAREA19")


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)
#install.packages("tidymodels", repos = "http://cran.us.r-project.org")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyvers, svglite, kabelExtra, RefManageR, truncnorm, tidymodels, knitr, mlbench, ggdag, causalTree, huxtable)
#install.packages("rstan", repos = "http://cran.us.r-project.org")
library(tidymodels)


## ---- echo=TRUE,message=FALSE, warning=FALSE------------------------------------------------------------------------------
set.seed(654)
# removing NAs from the data since honest.causalTree doesn't deal with them properly
filtered_dat <- dat_table3 %>% 
  filter(!is.na(AGE)) %>%
  filter(!is.na(FINISH8))
n <- nrow(filtered_dat)
trIdx <- which(filtered_dat$VOUCH0 == 1)
conIdx <- which(filtered_dat$VOUCH0 == 0)

train_idx <- c(sample(trIdx, length(trIdx) / 2),
               sample(conIdx, (length(conIdx) / 2) + 1))

train_data <- filtered_dat[train_idx, ]
est_data <- filtered_dat[-train_idx, ]
honestTree <- honest.causalTree(
               formula = FINISH8 ~ SVY + HSVISIT + DJAMUNDI + PHONE + AGE + SEX2 + STRATA1 + STRATA2 + STRATA3 + STRATA4 + STRATA5 + STRATA6 + STRATAMS + DBOGOTA + D1993 + D1995 + D1997 + DMONTH1 + DMONTH2 + DMONTH3 + DMONTH4 + DMONTH5 + DMONTH6 + DMONTH7 + DMONTH8 + DMONTH9 + DMONTH10 + DMONTH11 + DMONTH12 + SEX_MISS + DAREA1 + DAREA2 + DAREA3 + DAREA4 + DAREA5 + DAREA6 + DAREA7 + DAREA8 + DAREA9 + DAREA10 + DAREA11 + DAREA12 + DAREA13 + DAREA14 + DAREA15 + DAREA16 + DAREA17 + DAREA18 + DAREA19,
               data = train_data,
               treatment = train_data$VOUCH0,
               est_data = est_data,
               est_treatment = est_data$VOUCH0,
               split.Rule = "CT", split.Honest = T,
               HonestSampleSize = nrow(est_data),
               split.Bucket = T, cv.option = "fit",
               cv.Honest = F, minsize = 5, na.action=na.omit
               )


## ---- echo=FALSE, out.width = "50%", fig.align='center'-------------------------------------------------------------------
rpart.plot(honestTree)
#rpart.plot(honestTree, type = 3, clip.right.labs = TRUE, branch = .3)


## ---- echo=TRUE-----------------------------------------------------------------------------------------------------------
opcp <- honestTree$cptable[,1][which.min(honestTree$cptable[,4])]
pruned_tree <- prune(honestTree, opcp)


## ---- echo=FALSE, out.width = "50%", fig.align='center'-------------------------------------------------------------------
rpart.plot(pruned_tree)
#rpart.plot(pruned_tree, type = 3, clip.right.labs = TRUE, branch = .3)


## -------------------------------------------------------------------------------------------------------------------------
table(AGE)


## -------------------------------------------------------------------------------------------------------------------------
est_data$leaf <- predict(pruned_tree, est_data, type="vector")
est_data$leaf_fct <- as.factor(round(est_data$leaf, 3))

reg_model <- lm(formula = FINISH8 ~ -1 + leaf_fct + leaf_fct*VOUCH0 - VOUCH0,
                data = est_data)

huxtable::huxreg(reg_model, coefs=c("age>=19" ="leaf_fct-0.492:VOUCH0",
                                    "age<15_1997" ="leaf_fct0:VOUCH0",
                                    "15<age<=19_notDAREA12_notFeburary_notDAREA7" ="leaf_fct0.077:VOUCH0",
                                    "age<15_not1997" ="leaf_fct0.09:VOUCH0",
                                    "15<age<=19_notDAREA12_Feburary" ="leaf_fct0.321:VOUCH0",
                                    "15<age<=19_notDAREA12" ="leaf_fct0.5:VOUCH0",
                                    "15<age<=19_notDAREA12_notFeburary_DAREA7" ="leaf_fct0.6:VOUCH0"))


