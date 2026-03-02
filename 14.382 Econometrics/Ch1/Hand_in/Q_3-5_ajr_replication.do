
clear
set more off

* change directory here

insheet using "/bbkinghome/pskorin/Downloads/acemoglu_col.txt", clear
gen double logmort = log(mort)

local z = invnormal(0.975)

****************************************************
* 1) OLS
****************************************************
reg gdp exprop latitude, vce(robust)   // (or your exact OLS spec)
scalar b_ols  = _b[exprop]
scalar se_ols = _se[exprop]
scalar lo_ols = b_ols - `z'*se_ols
scalar hi_ols = b_ols + `z'*se_ols

****************************************************
* 2) IV (2SLS)
****************************************************
ivregress gmm gdp (exprop = logmort) latitude, r
scalar b_iv  = _b[exprop]
scalar se_iv = _se[exprop]
scalar lo_iv = b_iv - `z'*se_iv
scalar hi_iv = b_iv + `z'*se_iv

****************************************************
* 3) AR 95% confidence set
*    (assumes you already created a 251-point grid b[i,1])
****************************************************

set obs 251

local a = 0.107
local c = 2.262
local n = 251

local step = (`c' - `a')/(`n' - 1)

matrix b = J(`n', 1, .)

forvalues i = 1/`n' {
    matrix b[`i',1] = `a' + (`i'-1)*`step'
}

capture drop dep u m
capture drop beta stat pval in_set
gen beta  = .
gen stat  = .
gen pval  = .
gen in_set = .

* residualize gdp on controls
quietly reg gdp latitude
quietly predict double gdp_tilde, resid

* residualize exprop on controls
quietly reg exprop latitude
quietly predict double exprop_tilde, resid

* residualize instrument on same controls
quietly reg logmort latitude
quietly predict double z_tilde, resid

forvalues i = 1(1)251 {
    scalar b1 = b[`i',1]
    quietly replace beta = b1 in `i'

    quietly gen double u = gdp_tilde - b1*exprop_tilde

    * moment
    quietly gen double m  = z_tilde*u
    quietly gen double m2 = m^2

    quietly summarize m, meanonly
    scalar mbar = r(mean)

    quietly summarize m2, meanonly
    scalar Em2 = r(mean)

    count if !missing(z_tilde)
    scalar N = r(N)
    
    scalar Vm = (Em2 - mbar^2)*N/(N-1)

    scalar chi2 = N*(mbar^2)/Vm
    scalar pin    = chi2tail(1, chi2)

    quietly replace stat   = chi2 in `i'
    quietly replace pval   = pin    in `i'
    quietly replace in_set = (pin >= .05) in `i'

    capture drop u m m2
}

summ beta if in_set, meanonly

scalar lo_ar = r(min)
scalar hi_ar = r(max)

* Optional "summary point + implied SE" (not a true estimator/SE)
scalar b_ar  = (hi_ar + lo_ar)/2
scalar se_ar = (hi_ar - lo_ar)/(2*`z')

****************************************************
* 4) Build results table dataset
****************************************************
clear
set obs 3
gen str6 method = ""
gen double b  = .
gen double se = .
gen double lo = .
gen double hi = .

replace method = "OLS" in 1
replace b  = b_ols  in 1
replace se = se_ols in 1
replace lo = lo_ols in 1
replace hi = hi_ols in 1

replace method = "IV" in 2
replace b  = b_iv  in 2
replace se = se_iv in 2
replace lo = lo_iv in 2
replace hi = hi_iv in 2

replace method = "AR" in 3
replace b  = b_ar  in 3      // summary midpoint (optional)
replace se = se_ar in 3      // implied from CI width (optional)
replace lo = lo_ar in 3
replace hi = hi_ar in 3

format b se lo hi %9.2f
list method b se lo hi, noobs abbrev(16)

listtex method b se lo hi using results_table.tex, ///
    replace ///
    rstyle(tabular) ///
    head("\begin{tabular}{lcccc} \toprule \multicolumn{5}{c}{Effect of Institutions on Growth in AJR data} \\ \midrule Method & Estimate & Std. Error & Lower 95\% & Upper 95\% \\ \midrule") ///
    foot("\bottomrule \end{tabular}")
