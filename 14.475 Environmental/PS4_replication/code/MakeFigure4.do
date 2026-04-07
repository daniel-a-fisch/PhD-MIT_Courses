

* Make Burke and Emerick Figure 4

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine

local crop corn
local cut -0.0044
use data/us_panel, clear
merge n:1 fips using data/`crop'1980sample
drop _merge
keep if year>=1953 & year<=2005
local t 29
local j 42
keep dday`t'C dday0C prec `crop'yield `crop'_area fips longitude year in80sample
reshape wide dday`t'C dday0C prec `crop'yield `crop'_area, i(fips longitude in80sample) j(year) 
tostring fips, gen(fipschar)
replace fipschar="0"+fipschar if length(fipschar)==4
gen stfips=substr(fipschar,1,2)
destring stfips, replace
drop fipschar
forvalues k=1955(1)2003 { 
local l1 = `k'-1
local l2 = `k'-2
local ll1 = `k' + 1 
local ll2 = `k' + 2
gen dday0Csmooth`k' = (dday0C`l1' + dday0C`l2' + dday0C`k' + dday0C`ll1' + dday0C`ll2') / 5
gen dday`t'Csmooth`k' = (dday`t'C`l1' + dday`t'C`l2' + dday`t'C`k' + dday`t'C`ll1' + dday`t'C`ll2') / 5
gen precsmooth`k' = (prec`l1' + prec`l2' + prec`k' + prec`ll1' + prec`ll2') / 5
gen `crop'_areasmooth`k' = (`crop'_area`l1' + `crop'_area`l2' + `crop'_area`k' + `crop'_area`ll1' + `crop'_area`ll2') / 5
gen `crop'yieldsmooth`k' = (`crop'yield`l1' + `crop'yield`l2' + `crop'yield`k' + `crop'yield`ll1' + `crop'yield`ll2') / 5
}
foreach start in 1955 1960 1965 1970 1975 1980 1985 1990 1995 { 
foreach diffl in 5 10 15 20 25 30 { 
local end = `start' + `diffl'
if `end' <= 2000 {
	qui gen prec_lo_`start'_`end'=0
	qui gen prec_hi_`start'_`end'=0
	qui replace prec_lo_`start'_`end' = (precsmooth`end'- precsmooth`start') if precsmooth`start'<`j' & precsmooth`end'<=`j'
	qui replace prec_hi_`start'_`end' = (precsmooth`end' - precsmooth`start') if precsmooth`start'>`j' & precsmooth`end'>`j'
	qui replace prec_lo_`start'_`end' = (`j' - precsmooth`start') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	qui replace prec_hi_`start'_`end' = (precsmooth`end' - `j') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	qui replace prec_lo_`start'_`end' = (precsmooth`end' - `j') if precsmooth`start' >`j' & precsmooth`end'<=`j'
	qui replace prec_hi_`start'_`end' = (`j' - precsmooth`start') if precsmooth`start' >`j' & precsmooth`end'<=`j'
	qui gen lower_`start'_`end' = (dday0Csmooth`end' - dday`t'Csmooth`end') - (dday0Csmooth`start' - dday`t'Csmooth`start')
	qui gen higher_`start'_`end' = dday`t'Csmooth`end' - dday`t'Csmooth`start'
	qui gen log`crop'yield_diff`start'`end' = ln(`crop'yieldsmooth`end') - ln(`crop'yieldsmooth`start')
	qui gen beta_`start'_`end'=.
}
}
}
keep if in80sample==1
tempfile obsforit 
save "`obsforit'", replace /*all data to be drawn from for each bootstrap sample*/
keep if _n==1 
keep beta_*
tempfile betas 
save "`betas'", replace /*data to be updated with beta GDD high for each iteration*/

* NOW RUN BOOTSTRAP.  This takes a little while; go get a beverage/ check email. 
set seed 8571579
forvalues i=1(1)1000 { /*1000 iterations*/
use "`obsforit'", clear 
bsample 31, cl(stfips)
foreach start in 1955 1960 1965 1970 1975 1980 1985 1990 1995 { 
foreach diffl in 5 10 15 20 25 30 { 
local end = `start' + `diffl'
if `end' <= 2000 {
qui areg log`crop'yield_diff`start'`end' lower_`start'_`end' higher_`start'_`end' prec_lo_`start'_`end' prec_hi_`start'_`end' if longitude>-100 & in80sample==1 [aweight=`crop'_areasmooth`start'], a(stfips)
qui replace beta_`start'_`end' = _b[higher_`start'_`end'] in 1
}
}
}
di `i'
qui keep if _n==1
qui keep beta_*
qui append using "`betas'"
qui save "`betas'", replace
}
use "`betas'", clear
drop if _n==1001
save output/Fig4_data.dta, replace
*calculate standard errors
use output/Fig4_data.dta, clear
foreach start in 1955 1960 1965 1970 1975 1980 1985 1990 1995 { 
foreach diffl in 5 10 15 20 25 30 { 
local end = `start' + `diffl'
if `end' <= 2000 {
gen diff_`start'_`end' = beta_`start'_`end' - beta_1980_2000
}
}
}
	keep diff*
	gen n = _n
	reshape long diff_, i(n) j(tp) string
	collapse (mean) diffmean=diff_ (sd) diffsd=diff_, by(tp)
	gen starting = substr(tp,1,4) 
	gen ending = substr(tp,6,4)
	destring starting ending, replace
	gen dur = ending - starting
	gen lowci_d = diffmean - 1.96*diffsd
	gen highci_d = diffmean + 1.96*diffsd

twoway (rcap lowci_d highci_d starting if dur==5, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==5, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("5 Year Differences") name(f1, replace)

twoway (rcap lowci_d highci_d starting if dur==10, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==10, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("10 Year Differences") name(f2, replace)

twoway (rcap lowci_d highci_d starting if dur==15, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==15, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("15 Year Differences") name(f3, replace)

twoway (rcap lowci_d highci_d starting if dur==20, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==20, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("20 Year Differences") name(f4, replace)

twoway (rcap lowci_d highci_d starting if dur==25, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==25, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("25 Year Differences") name(f5, replace)

twoway (rcap lowci_d highci_d starting if dur==30, lcolor(black) scheme(s1color)) (scatter diffmean starting if dur==30, mcolor(black)) ///
, ytitle(Difference with 1980-2000 coefficient) xtitle(Starting Year) yline(0, lcolor(black)) legend(off) title("30 Year Differences") name(f6, replace)

graph combine f1 f2 f3 f4 f5 f6, altshrink scheme(s1color) ycom iscale(1.4)
graph export output/Figure4.pdf, as(pdf) replace	

