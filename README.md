
---
title: "Uso e comparação de XGBoost na predição de notas IMBD"
output:
  html_document:
    toc: true
    toc_float: true
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Documento

Nessa continuação iremos usar outro método, muito poderoso e rápido, para a predição das notas e compará-las aos resultados obtidos no artigo anterior.

Caso não tenha visto,
[Veja aqui!](http://rpubs.com/viniciuslucenasouza/316278)

Usaremos os mesmo banco de dados, com diferentes inputs. Significa que usaremos mais variáveis para o nosso modelo. O XGBoost está entre os algoritmos com maior destaque, pois sua maior simplicidade de calculo e rapidez de processamento o torna popular, não só isso, está entre os melhores, e ganhando inúmeras competições.

[Saiba mais aqui](https://xgboost.readthedocs.io/en/latest/)



Porém o modelo necessita de alguma particularidades e é em cima disso que iremos trabalhar, se forma simples, pois nosso banco de dados facilita nosso trabalho.

##Inicio

definimos o diretório (etapa não obrigatória, pois importaremos o banco de dados de um pacote)

```{r warning=FALSE}
diretorio <- getwd()
setwd(diretorio)
```

Importaremos os pacotes necessários para realizar nosso modelo:

```{r warning=FALSE}
library(tidyverse)
library(xgboost)
```

De maneira simples e igualmente ao exemplo anterior, importamos o banco de dados do pacote:
```{r}
prime <- ggplot2movies::movies
```

Da mesma maneira selecionamos e limpamos os dados.  
Note que, ao contrário do exemplo anterior, selecionamos ainda dummy variables referentes ao tipo de filme (drama, ação, comédia, etc)

```{r}
dados <- ggplot2movies::movies %>%
  filter(!is.na(budget), budget > 0) %>%
  select( year, budget, rating, Action, Animation, Comedy, Drama, Documentary, Romance, Short) %>%
  arrange(desc(year))
```



## Treinamento e teste
Para validação do modelo é necessário saber o quão bom nosso modelo está sendo.  
Então faremos a seleção de 75% dos dados para treinamento e 25% para teste.  
Definimos train e test.
```{r}
smp_size <- floor(0.75 * nrow(dados))
train_ind <- sample(seq_len(nrow(dados)), size = smp_size)

train <- dados[train_ind, ]
test <- dados[-train_ind, ]
```


###Definindo a variável x

```{r}
labels <- train$rating
ts_label <- test$rating
```

O XGBoost necessita que os dados estejam em matrix ou no formato proprio xgb.mtx para isso transformamos o nosso data.frame em matrix.

Note que aqui, obrigatoriamente devemos ter dados numéricos. Pode ser um empasse se tiver fatores, comuns em data.frames. Porém podem ser facilmente convertidos em dummy variables. No nosso caso, já possuimos o formato correto, porém pode ser facilmente conseguido atravéz de vários pacotes em R.  
Poderíamos ter feito antes e depois ter dividido os dados, nos evitaria gastar algumas linhas de códigos.

```{r}
new_tr <- model.matrix(~.+0,train[,-3])
new_ts <- model.matrix(~.+0,test[,-3])

dados_matrix <- data.matrix(dados, rownames.force = NA)
```

###Transformando em xgb.mtx

Para melhor desempenho, como recomenda o desenvolvedor. Então faremos isso:

```{r}
dtrain <- xgb.DMatrix(data = new_tr, label = labels)
dtest <- xgb.DMatrix(data = new_ts, label = ts_label)
```

##Modelo XGBoost Random Forest

Vamo agora criar nosso modelo como `reg:linear`, `eta=0.3`, `gamma=0`, `max_depth=6`, `min_child_weig=1`,`subsample=1` e `colsample_bytree=1`  
Esses são nossos parametros padrões


```{r}
params <- list(booster = "gbtree", objective = "reg:linear",
                 eta=0.3, gamma=0, max_depth=6, min_child_weight=1,
                 subsample=1, colsample_bytree=1)
```



##XGBoost melhores parametros:

Usando a função nativa do modelo `xgb.cv`, vamos calcular o melhor numero de rounds `nrounds` para esse modelo. E mais, essa função também retorna o erro de CV que é uma estimativa do erro do modelo.

Chamamos nossos parametro `params`, os dados para treinamento no formato xgb `dtrain`, `nround=100`, `nfold=5` e com `early.stop.round=20`. Imprimindo a cada 10 rounds.

```{r}
xgbcv <- xgb.cv( params = params, data = dtrain,
                 nrounds = 100, nfold = 5, showsd = T, stratified = T,
                 print_every_n = 10, early_stop_round = 20, maximize = F)
```

Também avaliamos o RMSE mínimo.

```{r}
min(xgbcv$evaluation_log$train_rmse_mean)
```

##Treinamento

Dos dados resultantes da função de validação cruzada, temos `nrounds=90`

```{r}
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 90,
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10,
                   early_stop_round = 10, maximize = F , eval_metric = "error")
```


###Predição do modelo
De forma simples escrevemos apenas:
```{r}
xgbpred <- predict(xgb1,dtest)

```

##Analisando o resultado
Calculamos simplesmento o RMSE entre o valor predito e o valor real do filme.
```{r}
library(caret)

xgb1_rmse <- RMSE(test$rating, xgbpred)
xgb1_rmse

```

Sendo esse o valor do erro médio quadrátido.

Para ficar bonito, plotamos o grafico que nos diz o grau de importancia das variáveis:
```{r}
library(caret)
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:9], col=3)
```

Assim vemos que ao adicionar nota a um filme, o fator mais importante é o orçamento, seguido do ano de lançamento. Tiramos a importante informação que se o filme for drama e curtametragem também tem peso significativo para o valor da nota.

É isso. Qualquer dúvida fique a vontade de enviar um e-mail para vinicius.lucena.souza@gmail.com

##Referencia
Vinicius Lucena
[Github](https://github.com/viniciuslucenasouza)
[Linkedin](https://www.linkedin.com/in/vinicius-lucena/)
