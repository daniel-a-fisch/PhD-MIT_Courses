
*  	Generate data for Figure 5 in Burke and Emerick
*	Outputs a file that is read into R for creation of figure

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


* First create pared down datasets that are quick to load
* first construct dataset(s)
local crop corn
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
foreach start in 1970 1975 1980 1985 { 
	foreach diffl in 15 20 25 30 { 
	local end = `start' + `diffl'
	if `end' <= 2000 {
	gen prec_lo_`start'_`end'=0
	gen prec_hi_`start'_`end'=0
	replace prec_lo_`start'_`end' = (precsmooth`end'- precsmooth`start') if precsmooth`start'<`j' & precsmooth`end'<=`j'
	replace prec_hi_`start'_`end' = (precsmooth`end' - precsmooth`start') if precsmooth`start'>`j' & precsmooth`end'>`j'
	replace prec_lo_`start'_`end' = (`j' - precsmooth`start') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	replace prec_hi_`start'_`end' = (precsmooth`end' - `j') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	replace prec_lo_`start'_`end' = (precsmooth`end' - `j') if precsmooth`start' >`j' & precsmooth`end'<=`j'
	replace prec_hi_`start'_`end' = (`j' - precsmooth`start') if precsmooth`start' >`j' & precsmooth`end'<=`j'
	gen lower_`start'_`end' = (dday0Csmooth`end' - dday`t'Csmooth`end') - (dday0Csmooth`start' - dday`t'Csmooth`start')
	gen higher_`start'_`end' = dday`t'Csmooth`end' - dday`t'Csmooth`start'
	gen logyield_`start'_`end' = ln(`crop'yieldsmooth`end') - ln(`crop'yieldsmooth`start')
	}
	}
	}
keep if longitude>-100
keep fips stfips logyield* prec_lo* prec_hi* lower* higher* corn_areasmooth1970 corn_areasmooth1975 corn_areasmooth1980 corn_areasmooth1985
save output/piecewise_boot1, replace
keep fips
tempfile tokeep
save "`tokeep'", replace  //only going to keep the counties that we have in the long diffs corn regressions
use data/us_panel, clear
local t 29
local j 42
tostring fips, gen(fipschar)
replace fipschar="0"+fipschar if length(fipschar)==4
gen stfips=substr(fipschar,1,2)
destring stfips, replace
drop fipschar
keep if longitude>-100 & year>=1968 & year<=2002
gen logcornyield = log(cornyield)
gen lower = dday0C - dday`t'C
gen higher = dday`t'C
gen prec_lo = (prec - `j')*(prec<=`j')
gen prec_hi = (prec - `j')*(prec>`j')
merge n:1 fips using "`tokeep'"
keep if _merge==3  //just keeping the counties that we have in the long diffs
keep logcornyield lower higher prec_lo prec_hi fips year stfips corn_area
save output/panel_boot1, replace

* Now run bootstrap
capture postutil clear
postfile boot str10(reg_model) replicate yr_start yr_end temp_lo temp_hi prec_lo prec_hi using output/Fig5_data, replace
set seed 8675309
forvalues i = 1/1000 {  //this takes a while
di "`i'"
use output/piecewise_boot1, clear
bsample, cluster(stfips)
foreach start in 1970 1975 1980 1985 { 
	foreach diffl in 15 20 25 30 { 
	local end = `start' + `diffl'
	if `end' <= 2000 {
	qui areg logyield_`start'_`end' lower_`start'_`end' higher_`start'_`end' prec_lo_`start'_`end' prec_hi_`start'_`end' [aweight=corn_areasmooth`start'], a(stfips)
	post boot ("diffs") (`i') (`start') (`end') (_b[lower_`start'_`end']) (_b[higher_`start'_`end']) (_b[prec_lo_`start'_`end']) (_b[prec_hi_`start'_`end'])
	}
	}
	}
	qui keep stfips
	qui duplicates drop
	tempfile tokeep
	qui save "`tokeep'", replace
	use output/panel_boot1, clear
	qui merge n:1 stfips using "`tokeep'"
	qui keep if _merge==3
foreach start in 1970 1975 1980 1985 { 
	foreach diffl in 15 20 25 30 { 
	local end = `start' + `diffl'
	if `end' <= 2000 {
	local ll=`start' - 2
	local lh=`end' + 2
	capture drop area
	egen area = mean(corn_area) if year>=`ll' & year<=`lh', by(fips)
	qui xtreg logcornyield lower higher prec_lo prec_hi i.year if year>=`ll' & year<=`lh' [aweight=area], fe i(fips)
	post boot ("panel") (`i') (`start') (`end') (_b[lower]) (_b[higher]) (_b[prec_lo]) (_b[prec_hi])
	}
	}
	}
}
postclose boot

use output/Fig5_data, clear
*outsheet using $data/bootstrap_cornyield_1980_00.csv, comma replace
egen period = concat(yr_start yr_end reg_model), punct("_")
keep if reg_model=="diffs"
keep replicate temp_hi* period
reshape wide temp, j(period) i(replicate) s
tempfile temp
save "`temp'", replace
use output/Fig5_data, clear
egen period = concat(yr_start yr_end reg_model), punct("_")
keep if reg_model=="panel"
keep replicate temp_hi* period
reshape wide temp, j(period) i(replicate) s
merge 1:1 replicate using "`temp'"
local perlist 1970_1985 1970_1990 1970_1995 1970_2000 1975_1990 1975_1995 1975_2000 1980_1995 1980_2000 1985_2000
foreach p of local perlist {
	gen adapt`p' = 1- temp_hi`p'_diffs/temp_hi`p'_panel
	xtile tile`p' = adapt`p', n(200)
	}
outsheet using output/Fig5_data.csv, comma replace
