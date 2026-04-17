This is the readme file for the data and code in R and Stata for the empirical application in Chen, Chernozhukov and Fernandez-Val (2018), "Mastering Panel 'Metrics: Causal Impact of Democracy on Growth". 

Authors: S. Chen, V. Chernozhukov and I. Fernandez-Val

Data Source: Daron Acemoglu (MIT), N = 147 countries, T = 23 years (1987 - 2009), balanced panel

################################
Description of the data: the sample selection and variable construction follows Acemoglu, Naidu, Restrepo and Robinson (2005), "Democracy Does Cause Growth," forthcoming JPE
################################

The variables in the data set include:

country_name  = Country name
webcode       = World Bank country code 
year          = Year
id            = Generated numeric country code
dem           = Democracy measure by ANRR
lgdp          = lag of GDP per capita in 2000 USD from World Bank

################################
Description of R Code File
################################

File: Democracy-AER-v2.R

Note: we use plm package to conduct panel data analysis and boot package to conduct bootstrap with parallel computing. These packages need to be installed with install.packages("plm") and install.packages("boot")

(1) Summarize and tabulate descriptive statistics 
(2) Uncorrected FE estimation, clustered standard errors and bootstrap standard errors. 
(3) Jackknife FE and bootstrap standard errors. 
(4) Analytical bias correction of FE and bootstrap standard errors. 
(5) Uncorrected AB estimation, clustered standard errors and bootstrap standard errors. 
(6) Split-sample bias correction of AB estimation and bootstrap standard errors. 

################################
Description for Stata code file: 
################################

Main Code File: main.do

Use: 

(1) Conducts uncorrected Fixed Effect (FE) and Arellano-Bond (AB) estimation and provides clustered standard errors and bootstrap standard errors. Note: we report first-step result in AB to follow the approach as in Acemoglu et al. 

(2) Conducts Jackknife FE estimation and provides bootstrap standard errors.

(3) Conducts Analytical Bias Correction of FE estimation and provides bootstrap standard errors. The trimming parameter is 4. 

(4) Conducts sample-splitting bias correction of AB estimation and provides bootstrap standard errors. We consider 1 split and 5 splits for sample splitting. 

Note: The number of bootstrap repetitions is 100 to speed up computation, but we recommend bootstrapping 500 times for more accurate results. The Stata code is way slower than the R code because R uses parallel computing in the bootstrap.  Because seed in R and Stata are not identical, sample splitting and bootstrap results might be a little bit different. This program generates the auxiliary data files uncorrected.dta, FEjack.dta, ABCboot.dta and gmmbcboot.dta. 

Auxiliary Code Files that should be saved in the working directory:

1. panelbs.do

Use:  mata code to generate bootstrap panel data.

2. mata_abc.do

Use: mata code to calculate analytical bias correction of fixed effect estimate.

