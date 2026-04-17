#################################################################################################
#  This is an empirical application for the paper Chen, Chernozhukov and Fernandez-Val (2018),
#  "Causal Impact of Democracy on Growth: An Applied Econometrics Perspective" 
#  based on Acemoglu, Naidu, Restrepo and Robinson (2005), "Democracy Does
#  Cause Growth," forthcoming JPE
#
#  Authors: S. Chen,  V. Chernozhukov and I. Fernandez-Val
#
# Data source: Daron Acemoglu (MIT), N = 147 countries, T = 23 years (1987-2009), balanced panel
#
# Description of the data: the sample selection and variable contruction follow ANRR
#
# The variables in the data set include:
#
# country_name    = Country name
# wbcode          = World Bank country code
# year            = Year 
# id              = Generated numeric country code
# dem             = Democracy measure by ANRR
# lgdp            = log of GDP per capita in 2000 USD from World Bank

#################################################################################################


# Updated on 01/02/2019


##################################################################################################




### set working directory
filepathIvan<-"/Users/Ivan/Dropbox/Shared/Democracy"
setwd(filepathIvan)

###  read-in TestScores data

library(foreign);
library(xtable);
library(plm);
library(gmm);
library(readstata13);

data <- read.dta13("Data/democracy-balanced-l4.dta")
data <- pdata.frame(data, index = c("id","year"));

attach(data);


########## Descriptive statistics

options(digits=2);
dstat <- cbind(sapply(data[, c(5:6)], mean), sapply(data[,c(5:6)], sd), apply(data[data$dem==1, c(5:6)],2, mean), apply(data[data$dem==0, c(5:6)], 2, mean));
dstat <- rbind(dstat, c(nrow(data), nrow(data), sum(data$dem==1), sum(data$dem==0)));
dimnames(dstat) <- list(c("Democracy", "Log(GDP)", "Number Obs."), c("Mean", "SD", "Dem = 1", "Dem = 0"));
xtable(dstat);


########## Fixed Effects estimation

form.fe <- lgdp ~ dem + lag(lgdp, 1:4) - 1;

fe.fit    <- plm(form.fe, data, model = "within", effect = "twoways", index = c("id","year"));
coefs.fe  <- coef(fe.fit); 
# se.fe     <- summary(fe.fit)$coef[ ,2];
HCV.coefs <- vcovHC(fe.fit, cluster = 'group');
cse.fe    <- sqrt(diag(HCV.coefs)); # Clustered std errors
lr.fe     <- coefs.fe[1]/(1-sum(coefs.fe[2:5]));
jac.lr <- c(1,rep(lr.fe,4))/(1-sum(coefs.fe[2:5]));
cse.lr.fe <- sqrt(t(jac.lr) %*% HCV.coefs[1:5,1:5] %*% jac.lr);

######### Split Panel Jackknife Bias correction

fe.fit1   <- plm(form.fe, data, subset = (as.double(year) <= 14), model = "within", effect = "twoways", index = c("id","year"));
fe.fit2   <- plm(form.fe, data, subset = (as.double(year) >= 14), model = "within", effect = "twoways", index = c("id","year"));
coefs.jbc  <- 19*coef(fe.fit)/9 - 10*(coef(fe.fit1) + coef(fe.fit2))/18; 
lr.fe1     <- coef(fe.fit1)[1]/(1-sum(coef(fe.fit1)[2:5]));
lr.fe2     <- coef(fe.fit2)[1]/(1-sum(coef(fe.fit2)[2:5]));
lr.jbc     <- 19*lr.fe/9 - 10*(lr.fe1 + lr.fe2)/18;


######### Analytical Bias correction

abc <- function(data, form, lags, N) {
  data$l1lgdp <- lag(data$lgdp,1); 
  data$l2lgdp <- lag(data$lgdp,2); 
  data$l3lgdp <- lag(data$lgdp,3); 
  data$l4lgdp <- lag(data$lgdp,4); 
  fit <- lm(form, data, x=TRUE, na.action = na.omit);
  res <- fit$residuals;
  jac <- solve(t(fit$x) %*% fit$x / length(res))[2:6,2:6];
  indexes <- c(1:length(res));
  bscore <- rep(0, 5)
  T <- length(res)/N;
  for (i in 1:lags) {
    indexes   <- indexes[-c(1+c(0:(N-1))*T)];
    lindexes  <- indexes - i;
    bscore  <- bscore + t(fit$x[indexes, 2:6]) %*% res[lindexes] / length(indexes);
  }
  bias <- - jac %*% bscore;
  return(as.vector(bias/T));
}


form.abc <- lgdp ~ dem + l1lgdp + l2lgdp + l3lgdp + l4lgdp + factor(year) + factor(id);
bias.l4 <- abc(data, form.abc, lags = 4, N = length(levels(id))); 
coefs.abc4 <- coefs.fe - bias.l4; 
jac.lr <- c(1,rep(lr.fe,4))/(1-sum(coefs.fe[2:5]));
lr.abc4 <- lr.fe - crossprod(jac.lr, bias.l4);

########## Arellano-Bond estimation - All lags


form.ab <- lgdp ~ dem + lag(lgdp, 1:4) | lag(lgdp, 2:99) + lag(dem, 1:99);
ab.fit <- pgmm(form.ab, data, model = "onestep", effect = "twoways" );
coefs.ab  <- coef(ab.fit); 
# se.ab     <- summary(ab.fit)$coef[ ,2];
HCV.coefs <- vcovHC(ab.fit, cluster = 'group');
cse.ab    <- sqrt(diag(HCV.coefs)); # Clustered std errors
# Jtest.ab  <- sargan(ab.fit)$statistic;
# Jdof.ab  <- sargan(ab.fit)$parameter;
lr.ab     <- coefs.ab[1]/(1-sum(coefs.ab[2:5]));
jac.lr <- c(1,rep(lr.ab,4))/(1-sum(coefs.ab[2:5]));
cse.lr.ab <- sqrt(t(jac.lr) %*% HCV.coefs[1:5,1:5] %*% jac.lr);

## Split sample bias correction (1 partition)

N         <- length(levels(id));
S2 <- 1;
acoeff.ab <- 0*coef(ab.fit);
alr.ab    <- 0;
for (s in 1:S2) {
  sample1   <- sample(N, ceiling(N/2), replace = FALSE);
  ab.fit1 <- pgmm(form.ab, data[as.double(id) %in% sample1, ], model = "onestep", effect = "twoways" );
  ab.fit2 <- pgmm(form.ab, data[!(as.double(id) %in% sample1), ], model = "onestep", effect = "twoways" );
  lr.ab1     <- coef(ab.fit1)[1]/(1-sum(coef(ab.fit1)[2:5]));
  lr.ab2     <- coef(ab.fit2)[1]/(1-sum(coef(ab.fit2)[2:5]));
  acoeff.ab <- acoeff.ab + ((coef(ab.fit1) + coef(ab.fit2))/2)/S2;
  alr.ab    <- alr.ab + ((lr.ab1 + lr.ab2)/2)/S2;
}
coefs.ab.jbc  <- 2*coef(ab.fit) - acoeff.ab;
lr.ab.jbc     <- 2*lr.ab - alr.ab;

## Split sample bias correction (5 partitions)

S2 <- 5;
acoeff.ab <- 0*coef(ab.fit);
alr.ab    <- 0;
for (s in 1:S2) {
  sample1   <- sample(N, ceiling(N/2), replace = FALSE);
  ab.fit1 <- pgmm(form.ab, data[as.double(id) %in% sample1, ], model = "onestep", effect = "twoways" );
  ab.fit2 <- pgmm(form.ab, data[!(as.double(id) %in% sample1), ], model = "onestep", effect = "twoways" );
  lr.ab1     <- coef(ab.fit1)[1]/(1-sum(coef(ab.fit1)[2:5]));
  lr.ab2     <- coef(ab.fit2)[1]/(1-sum(coef(ab.fit2)[2:5]));
  acoeff.ab <- acoeff.ab + ((coef(ab.fit1) + coef(ab.fit2))/2)/S2;
  alr.ab    <- alr.ab + ((lr.ab1 + lr.ab2)/2)/S2;
}
coefs.ab.jbc5  <- 2*coef(ab.fit) - acoeff.ab;
lr.ab.jbc5     <- 2*lr.ab - alr.ab;



########## Panel bootstrap std errors

R <- 500;

# Function to generate bootstrap data sets;

data.rg <- function(data, mle)
{
  N                  <- length(unique(data$id));
  T                  <- length(unique(data$year));
  ids                <- kronecker(sample.int(N, N, replace = TRUE), rep(1,T));
  data.b             <- data[(ids-1)*T + rep(c(1:T),N), ];
  data.b$id          <- kronecker(c(1:N), rep(1,T));
  data.b$year        <- rep(c(1987:2009),N);
  data.b             <- data.frame(data.b);
  data.b             <- pdata.frame(data.b, index = c("id","year"));         # reset indexes of the panel 
  return(data.b);
}

## FIXED EFFECTS APPROACH #######################################################################
########## statistics to be computed in each bootstrap draw #####################################
boot.SE.fe<- function(data, form.fe, form.abc){

# Fixed Effects
  fe.fit    <- plm(form.fe, data, model = "within", effect = "twoways", index = c("id","year"));
  coefs.fe  <- coef(fe.fit); 
  lr.fe     <- coefs.fe[1]/(1-sum(coefs.fe[2:5]));
  
# Split-sample bias correction
  fe.fit1   <- plm(form.fe, data, subset = (as.double(year) <= 14), model = "within", effect = "twoways", index = c("id","year"));
  fe.fit2   <- plm(form.fe, data, subset = (as.double(year) >= 14), model = "within", effect = "twoways", index = c("id","year"));
  coefs.jbc  <- 19*coef(fe.fit)/9 - 10*(coef(fe.fit1) + coef(fe.fit2))/18; 
  lr.fe1     <- coef(fe.fit1)[1]/(1-sum(coef(fe.fit1)[2:5]));
  lr.fe2     <- coef(fe.fit2)[1]/(1-sum(coef(fe.fit2)[2:5]));
  lr.jbc     <- 19*lr.fe/9 - 10*(lr.fe1 + lr.fe2)/18;
  
# Analytical bias correction
  bias.l4 <- abc(data, form.abc, lags = 4, N = length(levels(id))); 
  coefs.abc4 <- coefs.fe - bias.l4; 
  jac.lr <- c(1,rep(lr.fe,4))/(1-sum(coefs.fe[2:5]));
  lr.abc4 <- lr.fe - crossprod(jac.lr, bias.l4);

  return(c(coefs.fe, coefs.jbc, coefs.abc4, lr.fe, lr.jbc, lr.abc4));
}

library(boot); # library to do bootstrap with paralell computing

set.seed(888);
result.boot.SE.fe <- boot(data = data, statistic=boot.SE.fe, sim = "parametric", ran.gen = data.rg, mle = 0, form.fe = form.fe, form.abc = form.abc, 
                          parallel="multicore", ncpus = 20, R=R);


rsd <- function(x) { return((quantile(x,.75,na.rm=TRUE)-quantile(x,.25,na.rm=TRUE))/(qnorm(.75) - qnorm(.25)))} # robust estimator of std deviation based on IQR

# Robust bootstrap std errors;

result      <- structure(vapply(result.boot.SE.fe$t, as.double, numeric(1)), dim=dim(result.boot.SE.fe$t)); # transforms "Error in La.svd(x, nu, nv) : error code 1 from Lapack routine 'dgesdd'\n" to NA
bse.fe      <- apply(result[,1:5], 2, rsd);
bse.jbc     <- apply(result[,6:10], 2, rsd);
bse.abc4    <- apply(result[,11:15], 2, rsd);
bse.lr.fe   <- rsd(result[,16]);
bse.lr.jbc  <- rsd(result[,17]);
bse.lr.abc4 <- rsd(result[,18]);

## ARELLANO-BOND APPROACH #######################################################################
########## statistics to be computed in each bootstrap draw #####################################
boot.SE.ab<- function(data, form.ab){
  
  # Arellano-Bond
  N         <- length(unique(data$id));
  ab.fit <- pgmm(form.ab, data, model = "onestep", effect = "twoways" );
  coefs.ab  <- coef(ab.fit); 
  lr.ab     <- coefs.ab[1]/(1-sum(coefs.ab[2:5]));
  
  # Split sample bias correction - 1 partition
  S2        <- 1;
  acoeff.ab <- 0*coef(ab.fit);
  alr.ab    <- 0;
  for (s in 1:S2) {
    sample1   <- sample(N, ceiling(N/2), replace = FALSE);
    ab.fit1 <- pgmm(form.ab, data[as.double(id) %in% sample1, ], model = "onestep", effect = "twoways" );
    ab.fit2 <- pgmm(form.ab, data[!(as.double(id) %in% sample1), ], model = "onestep", effect = "twoways" );
    lr.ab1     <- coef(ab.fit1)[1]/(1-sum(coef(ab.fit1)[2:5]));
    lr.ab2     <- coef(ab.fit2)[1]/(1-sum(coef(ab.fit2)[2:5]));
    acoeff.ab <- acoeff.ab + ((coef(ab.fit1) + coef(ab.fit2))/2)/S2;
    alr.ab    <- alr.ab + ((lr.ab1 + lr.ab2)/2)/S2;
  }
  coefs.ab.jbc  <- 2*coef(ab.fit) - acoeff.ab;
  lr.ab.jbc     <- 2*lr.ab - alr.ab;
  
  # Split sample bias correction - 5 partitions
  S2        <- 5;
  acoeff.ab <- 0*coef(ab.fit);
  alr.ab    <- 0;
  for (s in 1:S2) {
    sample1   <- sample(N, ceiling(N/2), replace = FALSE);
    ab.fit1 <- pgmm(form.ab, data[as.double(id) %in% sample1, ], model = "onestep", effect = "twoways" );
    ab.fit2 <- pgmm(form.ab, data[!(as.double(id) %in% sample1), ], model = "onestep", effect = "twoways" );
    lr.ab1     <- coef(ab.fit1)[1]/(1-sum(coef(ab.fit1)[2:5]));
    lr.ab2     <- coef(ab.fit2)[1]/(1-sum(coef(ab.fit2)[2:5]));
    acoeff.ab <- acoeff.ab + ((coef(ab.fit1) + coef(ab.fit2))/2)/S2;
    alr.ab    <- alr.ab + ((lr.ab1 + lr.ab2)/2)/S2;
  }
  coefs.ab.jbc5  <- 2*coef(ab.fit) - acoeff.ab;
  lr.ab.jbc5     <- 2*lr.ab - alr.ab;
  
  
  return(c(coefs.ab[1:5], coefs.ab.jbc[1:5], coefs.ab.jbc5[1:5], lr.ab, lr.ab.jbc, lr.ab.jbc5));
}

library(boot); # library to do bootstrap with paralell computing

set.seed(888);
result.boot.SE.ab <- boot(data = data, statistic=boot.SE.ab, sim = "parametric", ran.gen = data.rg, mle = 0, form.ab = form.ab, 
                          parallel="multicore", ncpus = 20, R=R);


rsd <- function(x) { return((quantile(x,.75,na.rm=TRUE)-quantile(x,.25,na.rm=TRUE))/(qnorm(.75) - qnorm(.25)))} # robust estimator of std deviation based on IQR

# Robust bootstrap std errors;

result      <- structure(vapply(result.boot.SE.ab$t, as.double, numeric(1)), dim=dim(result.boot.SE.ab$t)); # transforms "Error in La.svd(x, nu, nv) : error code 1 from Lapack routine 'dgesdd'\n" to NA
bse.ab      <- apply(result[,1:5], 2, rsd);
bse.ab.jbc  <- apply(result[,6:10], 2, rsd);
bse.ab.jbc5 <- apply(result[,11:15], 2, rsd);
bse.lr.ab   <- rsd(result[,16]);
bse.lr.ab.jbc  <- rsd(result[,17]);
bse.lr.ab.jbc5  <- rsd(result[,18]);



######## Table of results;

options(digits=2);
table.all <- matrix(NA, nrow = 18, ncol = 6, dimnames = list(c("Democracy", "CSE", "BSE", "L1.log(gdp)",  "CSE1", "BSE1", "L2.log(gdp)",  "CSE2", "BSE2","L3.log(gdp)",  "CSE3", "BSE3", "L4.log(gdp)",  "CSE4", "BSE4", "LR-Democracy","CSE5","BSE5"), c("FE", "SBC", "ABC4", "AB", "SBC1", "SBC5" )));


table.all[c(1,4,7,10,13), 1] <- coefs.fe[1:5];
table.all[c(2,5,8,11,14), 1] <- cse.fe[1:5];
table.all[c(3,6,9,12,15), 1] <- bse.fe[1:5];

table.all[c(1,4,7,10,13), 2] <- coefs.jbc[1:5];
table.all[c(3,6,9,12,15), 2] <- bse.jbc[1:5];

table.all[c(1,4,7,10,13), 3] <- coefs.abc4[1:5];
table.all[c(3,6,9,12,15), 3] <- bse.abc4[1:5];

table.all[c(1,4,7,10,13), 4] <- coefs.ab[1:5];
table.all[c(2,5,8,11,14), 4] <- cse.ab[1:5];
table.all[c(3,6,9,12,15), 4] <- bse.ab[1:5];

table.all[c(1,4,7,10,13), 5] <- coefs.ab.jbc[1:5];
table.all[c(3,6,9,12,15), 5] <- bse.ab.jbc[1:5];

table.all[c(1,4,7,10,13), 6] <- coefs.ab.jbc5[1:5];
table.all[c(3,6,9,12,15), 6] <- bse.ab.jbc5[1:5];


table.all[16, 1] <- lr.fe;
table.all[17, 1] <- cse.lr.fe;
table.all[18, 1] <- bse.lr.fe;

table.all[16, 2] <- lr.jbc;
table.all[18, 2] <- bse.lr.jbc;

table.all[16, 3] <- lr.abc4;
table.all[18, 3] <- bse.lr.abc4;

table.all[16, 4] <- lr.ab;
table.all[18, 4] <- bse.lr.ab;

table.all[16, 5] <- lr.ab.jbc;
table.all[17, 5] <- cse.lr.ab;
table.all[18, 5] <- bse.lr.ab.jbc;

table.all[16, 6] <- lr.ab.jbc5;
table.all[18, 6] <- bse.lr.ab.jbc5;


table.all[1, ] <- 100 * table.all[1, ];
table.all[2, ] <- 100 * table.all[2, ];
table.all[3, ] <- 100 * table.all[3, ];
table.all[16, ] <- 100 * table.all[16, ];
table.all[17, ] <- 100 * table.all[17, ];
table.all[18, ] <- 100 * table.all[18, ];


xtable(table.all, digits=2);


save.image(file="Results/democracy.RData");



