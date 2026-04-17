/*
This is an empirical application for the paper Chen, Chernozhukov and Fernandez-Val (2018),
"Mastering Panel Metrics: Causal Impact of Democracy on Growth"
based on Acemoglu, Naidu, Restrepo and Robinson (2015), "Democracy Does Cause Growth", 
forthcoming JPE. 

Authors: S. Chen, V. Chernozhukov and I. Fernandez-Val

Data Source: Daron Acemoglu (MIT), N = 147 countries, T = 23 years (1987 - 2009), balanced panel

Description of the data: the sample selection and variable construction following ANRR

The variables in the data set include:

country_name  = Country name
webcode       = World Bank country code 
year          = Year
id            = Generated numeric country code
dem           = Democracy measure by ANRR
lgdp          = lag of GDP per capita in 2000 USD from World Bank
*/

ssc install xtabond2
ssc install moremata

clear all
cd /Users/Ivan/Dropbox/Shared/Democracy/Data-and-Programs
use "democracy-balanced-l4.dta", clear

**********************************************************************
**** Uncorrected Estimation
**** Note: POLS, FD, FE-IV, AB, AH are also doable using this setting. 
**** For simplicity, we show results for FE and AB. 
**********************************************************************
quietly {
* Claim panel structure 
tsset id year 
sort id year
* tabulate years and generate year dummy
tab year, gen(yr)

* FE
xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
matrix FE = (_b[dem],_b[L1.lgdp],_b[L2.lgdp],_b[L3.lgdp],_b[L4.lgdp]\_se[dem],_se[L1.lgdp],_se[L2.lgdp],_se[L3.lgdp],_se[L4.lgdp])
nlcom lr: _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]), post
matrix l = (_b[lr]\_se[lr])
* Estimation matrix
matrix FE = FE, l

* AB
xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
matrix AB = (_b[dem],_b[L1.lgdp],_b[L2.lgdp],_b[L3.lgdp],_b[L4.lgdp]\_se[dem],_se[L1.lgdp],_se[L2.lgdp],_se[L3.lgdp],_se[L4.lgdp])
nlcom lr: _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]), post
matrix l = (_b[lr]\_se[lr])
matrix AB = AB, l
matrix rownames AB = Coefs CSE
matrix colnames AB = dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr

set matsize 1000
set seed 1
* Use post command to store boot results
tempname mem
* First postfile, then do for loop
postfile `mem' dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 long_run dem_ab lgdp_l1_ab lgdp_l2_ab lgdp_l3_ab lgdp_l4_ab long_run_ab using "uncorrected.dta", replace
* Bootstrap
forvalues i = 1/100 {
 preserve
 * Call panelns.do file to generate data_bs
 quietly do panelbs.do
 * Reclaim panel structure
 quietly tsset id year
 quietly sort id year
 * Storing estimation results
 quietly: xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
 mat results = r(table)
 local dem = results[1, 5]
 local lgdp_l1 = results[1, 1]
 local lgdp_l2 = results[1, 2]
 local lgdp_l3 = results[1, 3]
 local lgdp_l4 = results[1, 4]
 local long_run = results[1, 5]/(1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])
 
 xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 mat results = r(table)
 local dem_ab = results[1, 5]
 local lgdp_l1_ab = results[1, 1]
 local lgdp_l2_ab = results[1, 2]
 local lgdp_l3_ab = results[1, 3]
 local lgdp_l4_ab = results[1, 4]
 local long_run_ab = results[1, 5]/(1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])
 post `mem' (`dem') (`lgdp_l1') (`lgdp_l2') (`lgdp_l3') (`lgdp_l4') (`long_run') (`dem_ab') (`lgdp_l1_ab') (`lgdp_l2_ab') (`lgdp_l3_ab') (`lgdp_l4_ab') (`long_run_ab')
 restore
}
postclose `mem'
preserve
use uncorrected.dta, clear
* Now calculate bootstrap standard error using interquartile function
mata
function bse(x)
{
 rsd = mm_iqrange(x, 1)/1.34898
 return(rsd)
}
bs_result = st_data(., .)
bse = bse(bs_result)
st_matrix("bse", bse)
end
matrix FE = FE\bse[1, 1..6]
matrix AB = AB\bse[1, 7..12]
matrix FE = FE[1..3,1]\FE[1..3,2]\FE[1..3,3]\FE[1..3,4]\FE[1..3,5]\FE[1..3,6]
matrix AB = AB[1..3,1]\AB[1..3,2]\AB[1..3,3]\AB[1..3,4]\AB[1..3,5]\AB[1..3,6]
matrix colnames FE = FE
matrix colnames AB = AB
matrix rownames FE = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
matrix rownames AB = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
restore
}

**********************************************************************
**** Jackknife FE Estimation
**** Note: The code is designed for balanced panel
**********************************************************************
* Estimation
quietly {
set seed 1
* whole sample estimation
quietly: xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
* Store results
matrix b=(_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
matrix b = 19 * b / 9
* Split over time-series dimension
quietly: xtreg lgdp l(1/4).lgdp dem yr* if year<=2000, fe r cluster(id)
matrix c1 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
quietly: xtreg lgdp l(1/4).lgdp dem yr* if year>=2000, fe r cluster(id)
matrix c2 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
matrix c = 10 * (c1 + c2) /18
matrix b = b - c
matrix colnames b = dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr
matrix rownames b = Coefs_full
}
* Bootstrap
quietly {
tempname mem
postfile `mem' dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr using "FEjack.dta", replace
forvalues i = 1/100 {
 preserve
 * Call panelns.do file to generate data_bs
 quietly do panelbs.do
 * Reclaim panel structure
 quietly tsset id year
 quietly sort id year
 
 * Storing estimation results
 xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
 mat results = r(table)
 local dem0 = results[1, 5]
 local lgdp0_l1 = results[1, 1]
 local lgdp0_l2 = results[1, 2]
 local lgdp0_l3 = results[1, 3]
 local lgdp0_l4 = results[1, 4]
 local lr0 = results[1, 5]/(1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])
 
 xtreg lgdp l(1/4).lgdp dem yr* if year<=2000, fe r cluster(id)
 mat results1 = r(table)
 local dem1 = results1[1, 5]
 local lgdp1_l1 = results1[1, 1]
 local lgdp1_l2 = results1[1, 2]
 local lgdp1_l3 = results1[1, 3]
 local lgdp1_l4 = results1[1, 4]
 local lr1 = results1[1, 5]/(1-results1[1, 1]-results1[1, 2]-results1[1, 3]-results1[1, 4])
 
 xtreg lgdp l(1/4).lgdp dem yr* if year>=2000, fe r cluster(id)
 mat results2 = r(table)
 local dem2 = results2[1, 5]
 local lgdp2_l1 = results2[1, 1]
 local lgdp2_l2 = results2[1, 2]
 local lgdp2_l3 = results2[1, 3]
 local lgdp2_l4 = results2[1, 4]
 local lr2 = results2[1, 5]/(1-results2[1, 1]-results2[1, 2]-results2[1, 3]-results2[1, 4])
 
 local dem = 19/9 * `dem0'- 10/18 * (`dem1' + `dem2')
 local lgdp_l1 = 19/9 * `lgdp0_l1' - 10/18 * (`lgdp1_l1' + `lgdp2_l1')
 local lgdp_l2 = 19/9 * `lgdp0_l2' - 10/18 * (`lgdp1_l2' + `lgdp2_l2')
 local lgdp_l3 = 19/9 * `lgdp0_l3' - 10/18 * (`lgdp1_l3' + `lgdp2_l3')
 local lgdp_l4 = 19/9 * `lgdp0_l4' - 10/18 * (`lgdp1_l4' + `lgdp2_l4')
 local lr = 19/9 * `lr0' - 10/18 * (`lr1' + `lr2')
 
 post `mem' (`dem') (`lgdp_l1') (`lgdp_l2') (`lgdp_l3') (`lgdp_l4') (`lr') 
 restore
}
postclose `mem'
preserve
use FEjack.dta, clear
mata
bs_result = st_data(., .)
bse = bse(bs_result)
st_matrix("bse", bse)
end
matrix cse = J(1, 6, .)
matrix FEjack = b\cse\bse
matrix FEjack = FEjack[1..3,1]\FEjack[1..3,2]\FEjack[1..3,3]\FEjack[1..3,4]\FEjack[1..3,5]\FEjack[1..3,6]
matrix colnames FEjack = DFE-SS
matrix rownames FEjack = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
restore
}

**********************************************************************
**** Analytical Bias Correction of FE estimation
**** Note: We show estimation with trimming par being 4
**********************************************************************
* Estimation
quietly {
use "democracy-balanced-l4.dta", clear
tsset id year 
sort id year
* tabulate years and generate year dummy
tab year, gen(yr)
tab id, gen(id)
xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
matrix abcf = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp])

local lr = abcf[1,1]/(1-abcf[1,2]-abcf[1,3]-abcf[1,4]-abcf[1,5])
local fir = 1/(1-abcf[1,2]-abcf[1,3]-abcf[1,4]-abcf[1,5])
local lr2 = abcf[1,1]/((1-abcf[1,2]-abcf[1,3]-abcf[1,4]-abcf[1,5])^2)
* lrmat mat is for bias correction
matrix lrmat = (`fir', `lr2', `lr2', `lr2', `lr2')

* generate lags
forvalues i = 1/4 {
 gen lgdp_l`i' = L`i'.lgdp
}

reg lgdp l(1/4).lgdp dem yr* id*
predict res, residuals
* Dropping missing values of res also drops first four years' data
drop if res == . 
drop yr1 yr2 yr3 yr4

* Two auxiliary mata functions to subset indexes and calculate analytical bias corrections
mata
function subset(x)
{
 num = rows(x)
 group = (num - mod(num, 19))/19
 temp = J(num, 1, .)
 for (i=1; i<= group; i++) {
    temp[(2+19*(i-1))..(19*i)] = x[(2+19*(i-1))..(19*i)]
 }
 if (mod(num, 19)>1) temp[(19*group+2)..rows(temp)] = x[(19*group+2)..rows(x)]
 else temp = temp
 temp = select(temp, rowmissing(temp):==0)
 return(temp)
}

function abc(X, res, l, T)
{
 jac = invsym(X'*X/rows(X))[2..6, 2..6]
 indexes = 1::rows(res)
 bscore = mm_expand(0, 5)
 for (i=1; i<=l; i++){
    indexes = subset(indexes)
    lindexes = indexes - mm_expand(i, rows(indexes))
    bscore = bscore + (X[indexes, (2..6)])'*res[lindexes]/rows(indexes)
 }
  bias = -jac * bscore
  bias = bias / T
  return(bias)
}
end

* Call auxiliary do files
do mata_abc.do

mat coefs4 = abcf - bias4
mat b4 = lrmat * bias4'
mat lr4 = `lr' - b4

mat abc4 = coefs4, lr4
mat rownames abc4 = Coefs
mat colnames abc4 = dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr
}

* Bootstrap
quietly {
use "democracy-balanced-l4.dta", clear
tsset id year 
sort id year
* tabulate years and generate year dummy
tab year, gen(yr)
tab id, gen(id)

set seed 888
tempname mem
postfile `mem' dem4 lgdp4_l1 lgdp4_l2 lgdp4_l3 lgdp4_l4 lr4 using "ABCboot.dta", replace
forvalues j = 1/100 {
 preserve
 * Call panelns.do file to generate data_bs
 quietly do panelbs.do
 quietly tsset id year
 quietly sort id year
 * Storing estimation results
 
 quietly: xtreg lgdp l(1/4).lgdp dem yr*, fe r cluster(id)
 mat results = r(table)
 local dem0 = results[1, 5]
 local lgdp0_l1 = results[1, 1]
 local lgdp0_l2 = results[1, 2]
 local lgdp0_l3 = results[1, 3]
 local lgdp0_l4 = results[1, 4]
 local lr = results[1, 5]/(1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])
 local lr2 = results[1, 5]/((1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])^2)
 local fir = 1/(1-results[1, 1]-results[1, 2]-results[1, 3]-results[1, 4])
 matrix lrmat = (`fir', `lr2', `lr2', `lr2', `lr2')
 
 forvalues i = 1/4 {
    gen lgdp_l`i' = L`i'.lgdp
 }

 quietly: reg lgdp l(1/4).lgdp dem yr* id*
 predict res, residuals
 * Dropping missing values of res also drops first four years' data
 drop if res == . 
 drop yr1 yr2 yr3 yr4
 
 do mata_abc.do
 
 mat b4 = lrmat * bias4'
 local dem4 = `dem0' - bias4[1,1]
 local lgdp_l1_4 = `lgdp0_l1' - bias4[1,2]
 local lgdp_l2_4 = `lgdp0_l2' - bias4[1,3]
 local lgdp_l3_4 = `lgdp0_l3' - bias4[1,4]
 local lgdp_l4_4 = `lgdp0_l4' - bias4[1,5]
 local long_run_4 = `lr' - b4[1, 1]
 
 post `mem' (`dem4') (`lgdp_l1_4') (`lgdp_l2_4') (`lgdp_l3_4') (`lgdp_l4_4') (`long_run_4')
 restore
 }
 postclose `mem'
 preserve
 use ABCboot.dta, replace
 mata
 bs_result = st_data(., .)
 bse = bse(bs_result)
 st_matrix("bse", bse)
 end
 * Show the resulting matrix
 mat cse = J(1, 6, .)
 mat ABC4 = abc4\cse\bse
 mat ABC4 = ABC4[1..3,1]\ABC4[1..3,2]\ABC4[1..3,3]\ABC4[1..3,4]\ABC4[1..3,5]\ABC4[1..3,6]
 mat colnames ABC4 = DFE-A4
 mat rownames ABC4 = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
 restore
}

**********************************************************************
**** Bias Correction of GMM (AB)
**** Note: We conduct estimation with 1 and 5 splits 
**** Since seed in Stata and R are not identical, the two languages might
**** produce slightly different results
**********************************************************************
* Estimation
quietly: {
 use "democracy-balanced-l4.dta", clear
 tsset id year 
 sort id year
 tab year, gen(yr)
 set seed 888
 * Save data in a tempfile (handy for sample splitting)
 tempfile whole
 save `whole', replace
 * Whole sample estimation
 
 * Arellano-Bond
 quietly: xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 matrix AB0 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
 * Now split panel data along cross-section. whole: whole sample; sample1: first part; sample2: second part 
 use `whole', replace
 collapse (mean) dem, by(id)
 keep id
 sample 50
 tempfile ind
 save `ind', replace
 use `whole', replace
 merge m:1 id using `ind'
 drop if _merge == 1
 drop _merge
 tempfile sample1
 save `sample1', replace
 use `whole', replace
 merge m:1 id using `ind'
 drop if _merge == 3
 drop _merge
 tempfile sample2
 save `sample2', replace
}

* Subsample estimation
quietly {
 use `sample1', replace
 tsset id year
 sort id year
 xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 matrix AB1 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
 use `sample2', replace
 tsset id year
 sort id year
 xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 matrix AB2 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
}

* One split results
quietly: {
mat AB_jbc = 2 * AB0 - 0.5 * (AB1 + AB2)
mat rownames AB_jbc = Coefs
mat colnames AB_jbc = dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr
}

* Now do 5 splits
quietly {
mat avg_ab = J(1, 6, 0)

forvalues i = 1/5 {
 use `whole', replace
 collapse (mean) dem, by(id)
 keep id
 sample 50
 tempfile ind
 save `ind', replace
 use `whole', replace
 merge m:1 id using `ind'
 drop if _merge == 1
 drop _merge
 tempfile sample1
 save `sample1', replace
 use `whole', replace
 merge m:1 id using `ind'
 drop if _merge == 3
 drop _merge
 tempfile sample2
 save `sample2', replace
 
 use `sample1', replace
 tsset id year
 sort id year
 xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 matrix ab1 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
 use `sample2', replace
 tsset id year
 sort id year
 xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
 matrix ab2 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
 matrix avg_ab = avg_ab + (0.5 * (ab1 + ab2))/5
}
use `whole', replace
mat AB_jbc5 = 2 * AB0 - avg_ab

mat rownames AB_jbc5 = Coefs
mat colnames AB_jbc5 = dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr
}

* Bootstrap
quietly: {
use "democracy-balanced-l4.dta", clear
tsset id year 
sort id year
tab year, gen(yr)

* save original data
tempfile whole
save `whole', replace

* change id in the data so that sample splitting in bootstrapped panel data is valid
mata
id = st_data(., "id")
N = 147
T = 23
id = (1::N)#mm_expand(1, T)
st_store(., 4, id)
end
tsset id year
sort id year
tempfile wholebs
save `wholebs', replace

 tempname mem
 postfile `mem' dem lgdp_l1 lgdp_l2 lgdp_l3 lgdp_l4 lr dem5 lgdp_l1_5 lgdp_l2_5 lgdp_l3_5 lgdp_l4_5 lr5 using "gmmbcboot.dta", replace
 * storing bootstrap results
 forvalues i = 1/100 {
  
  use `wholebs', replace
  
  * generate bootstrap panel data set
  do panelbs.do
  
  * whole sample estimation
  xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
  matrix AB0 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
  
  * randomly split sample along cross-section
  collapse (mean) dem, by(id)
  keep id
  sample 50
  tempfile indbs
  save `indbs', replace
  use `wholebs', replace
  merge m:1 id using `indbs'
  drop if _merge == 1
  drop _merge
  tempfile sample1bs
  save `sample1bs', replace
  use `wholebs', replace
  merge m:1 id using `indbs'
  drop if _merge == 3
  drop _merge
  tempfile sample2bs
  save `sample2bs', replace
  
  * subsample estimation 
  use `sample1bs', replace
  tsset id year
  sort id year
  xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
  matrix AB1 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
  use `sample2bs', replace
  tsset id year
  sort id year
  xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
  matrix AB2 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
  
  * One split result
  mat AB_bt1 = 2 * AB0 - 0.5 * (AB1 + AB2)
  
  * Five split result
  mat avg_ab = J(1, 6, 0)
  
  forvalues j = 1/5 {
    use `wholebs', replace
    * randomly split sample along cross-section
    collapse (mean) dem, by(id)
    keep id
    sample 50
    tempfile indbs
    save `indbs', replace
    use `wholebs', replace
    merge m:1 id using `indbs'
    drop if _merge == 1
    drop _merge
    tempfile sample1bs
    save `sample1bs', replace
    use `wholebs', replace
    merge m:1 id using `indbs'
    drop if _merge == 3
    drop _merge
    tempfile sample2bs
    save `sample2bs', replace
	
	* subsample est
	use `sample1bs', replace
    tsset id year
    sort id year
    quietly: xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
    matrix ab1 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
 
    use `sample2bs', replace
    tsset id year
    sort id year
    quietly: xtabond2 lgdp l(1/4).lgdp dem yr*, gmmstyle(lgdp, laglimits(2 .)) gmmstyle(dem, laglimits(1 .)) ivstyle(yr*, p) noleveleq robust nodiffsargan
    matrix ab2 = (_b[dem],_b[L1.lgdp], _b[L2.lgdp], _b[L3.lgdp], _b[L4.lgdp], _b[dem]/(1-_b[L1.lgdp]-_b[L2.lgdp]-_b[L3.lgdp]-_b[L4.lgdp]))
	
    matrix avg_ab = avg_ab + (0.5 * (ab1 + ab2))/5
  }
  mat AB_bt5 = 2 * AB0 - avg_ab
  
  foreach n in 1 5 {
    local dem`n' = AB_bt`n'[1, 1]
	local l1`n' = AB_bt`n'[1, 2]
    local l2`n' = AB_bt`n'[1, 3]
    local l3`n' = AB_bt`n'[1, 4]
    local l4`n' = AB_bt`n'[1, 5]
    local lr`n' = AB_bt`n'[1, 6]
  }
  
  post `mem' (`dem1') (`l11') (`l21') (`l31') (`l41') (`lr1') (`dem5') (`l15') (`l25') (`l35') (`l45') (`lr5')
 }
 postclose `mem'
 preserve
 use gmmbcboot.dta, clear
 mata
 bs_result = st_data(., .)
 bse = bse(bs_result)
 st_matrix("bse", bse)
 end
 matrix cse = J(1, 6, .)
 * Show the results
 mat AB_jbc_1 = AB_jbc\cse\bse[1,1..6]
 mat AB_jbc_5 = AB_jbc5\cse\bse[1,7..12]
 mat AB_jbc_1 = AB_jbc_1[1..3,1]\AB_jbc_1[1..3,2]\AB_jbc_1[1..3,3]\AB_jbc_1[1..3,4]\AB_jbc_1[1..3,5]\AB_jbc_1[1..3,6]
 mat AB_jbc_5 = AB_jbc_5[1..3,1]\AB_jbc_5[1..3,2]\AB_jbc_5[1..3,3]\AB_jbc_5[1..3,4]\AB_jbc_5[1..3,5]\AB_jbc_5[1..3,6]
 mat colnames AB_jbc_1 = DAB-SS1
 mat colnames AB_jbc_5 = DAB-SS5
 mat rownames AB_jbc_1 = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
 mat rownames AB_jbc_5 = dem cse bse lgdp_l1 cse bse lgdp_l2 cse bse lgdp_l3 cse bse lgdp_l4 cse bse lr cse bse
 restore
 use `whole', clear
}

* Report results in a table
mat TABLE = FE, FEjack, ABC4, AB, AB_jbc_1, AB_jbc_5
mat list TABLE

