
diretorio <- getwd()
setwd(direotrio)

library(tidyverse)
library(xgboost)

prime <- ggplot2movies::movies

dados <- ggplot2movies::movies %>% 
  filter(!is.na(budget), budget > 0) %>% 
  select( year, budget, rating, Action, Animation, Comedy, Drama, Documentary, Romance, Short) %>% 
  arrange(desc(year))

## 75% of the sample size
smp_size <- floor(0.75 * nrow(dados))

## set the seed to make your partition reproductible
train_ind <- sample(seq_len(nrow(dados)), size = smp_size)

train <- dados[train_ind, ]
test <- dados[-train_ind, ]

#using one hot encoding 
labels <- train$rating 
ts_label <- test$rating

dados_matrix <- data.matrix(dados, rownames.force = NA)

new_tr <- model.matrix(~.+0,train[,-3])
new_ts <- model.matrix(~.+0,test[,-3])


#Transformando em xgb.mtx

dtrain <- xgb.DMatrix(data = new_tr, label = labels)
dtest <- xgb.DMatrix(data = new_ts, label = ts_label)



#Modelo XGBoost Random Forest
#default parameters
params <- list(booster = "gbtree", objective = "reg:linear", 
                 eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
                 subsample=1, colsample_bytree=1)



#XGBoost:
xgbcv <- xgb.cv( params = params, data = dtrain, 
                 nrounds = 100, nfold = 5, showsd = T, stratified = T, 
                 print_every_n = 10, early_stop_round = 20, maximize = F)

min(xgbcv$evaluation_log$train_rmse_mean)


xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 90, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10, 
                   early_stop_round = 10, maximize = F , eval_metric = "error")
#model prediction
xgbpred <- predict (xgb1,dtest)

#confusion matrix
library(caret)

labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

RMSE(xgbpred, ts_label)

test0 <- xgbpred
test1 <- test$rating
testao <- cbind(test1,test0)
testao <- data.frame(testao)
testao <- dplyr::mutate(testao, testinho = test1 - test0)

test2 <- chisq.test(test$rating, xgbpred)
test3 <- RMSE(test$rating, xgbpred)


#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)

xgb.plot.importance (importance_matrix = mat[1:9], col=2) 

levels(xgbpred)
levels(ts_label)


#Primeiro modelo
modelo <- lm(rating ~ budget + year, data = dados)
summary(modelo)

#Um modelo Melhor
modelo2 <- lm(rating ~ budget * year, data = dados)
summary(modelo2)

#Fun??o preditiva:

funcao_que_preve <- function(orcamento, ano) {
  predict(modelo, newdata = data.frame(budget = orcamento, year = ano))
}

#Fun??o que apresenta uma amostra de 10 obs aleatorias
solta_10 <- function() {
  dados %>% 
    sample_n(10)
}

#Para criar uma API com o plumber, voc? precisa de tr?s coisas:
  #1. Uma fun??o que executa uma a??o
  #2. Uma documenta??o da fun??o
  #3. Selecionar uma porta para disponibilizar sua api

library("plumber")

p <- plumber::plumb('meuscript.R')
p$run(port = 8888)

#PLOTANDO DADOS
ggplot(dados, aes(y = rating, x = budget)) + 
  geom_point(aes(colour = year)) +
  geom_smooth(method = "lm")+
  geom_point(x=10000,y=my_rt,colour="red", size=2)

ggplot(dados, aes(y = budget, x = year)) + 
  geom_point(aes(colour = year)) +
  geom_smooth(method = "lm")+
  geom_point(x=10000,y=my_rt,colour="red", size=2)

#E se trasformarmos a variave budget
library("dplyr")

dados2 <- dplyr::mutate(dados, lnbget = log(budget))
ggplot(dados2, aes(y = lnbget, x = year)) + 
  geom_point(aes(colour = year)) +
  geom_smooth(method = "lm")+
  geom_point(x=10000,y=my_rt,colour="red", size=2)

ggplot()+  
geom_point(aes(y=my_rt, x=10000, size=2),colour="blue")

my_rt <- as.numeric(funcao_que_preve(100000,2011))
my_rt
