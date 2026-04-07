

* Replication data for Table 5 in Burke and Emerick

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine



* First run high school graduation rate and Bush vote share interactions

insheet using data/election_results_2000.csv, clear
drop if bush_pct=="NA"
destring bush_pct, replace
keep fips bush_pct 
tempfile votes 
save "`votes'", replace

use data/yield_piecewise, clear
merge 1:1 fips using "`votes'"	
drop _merge
merge 1:1 fips using data/highschoolgrad1980
	local t 29 
	local j 42
	gen prec_lo=0
	gen prec_hi=0
	replace prec_lo = (prec_smooth2000 - prec_smooth1980) if prec_smooth1980<`j' & prec_smooth2000<=`j'
	replace prec_hi = (prec_smooth2000 - prec_smooth1980) if prec_smooth1980>`j' & prec_smooth2000>`j'
	replace prec_lo = (`j' - prec_smooth1980) if prec_smooth1980<=`j' & prec_smooth2000>`j'
	replace prec_hi = (prec_smooth2000 - `j') if prec_smooth1980<=`j' & prec_smooth2000>`j'
	replace prec_lo = (prec_smooth2000 - `j') if prec_smooth1980 >`j' & prec_smooth2000<=`j'
	replace prec_hi = (`j' - prec_smooth1980) if prec_smooth1980 >`j' & prec_smooth2000<=`j'
gen lower = dday0_`t'C_diff1980_2000 
gen higher = dday`t'C_diff1980_2000
areg logcornyield_diff1980_2000 lower higher prec_lo prec_hi if longitude>-100 [aweight=corn_area_smooth1980], robust a(stfips) cl(stfips)
sum bush_pct if e(sample) 
replace bush_pct = bush_pct - r(mean) 
sum gradpct if e(sample) 
replace gradpct = gradpct - r(mean) 

// generate interactions
foreach i in lower higher prec_lo prec_hi { 
	gen bush_`i' = bush_pct*`i'
	gen gradpct_`i' = gradpct*`i'
	}

areg logcornyield_diff1980_2000 lower higher prec_lo prec_hi bush* if longitude>-100 [aweight=corn_area_smooth1980], robust a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store bushvs

areg logcornyield_diff1980_2000 lower higher prec_lo prec_hi gradpct* if longitude>-100 [aweight=corn_area_smooth1980], robust a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store highschool


// now run regressions looking at interaction with past climate exposure
local crop corn
use data/us_panel, clear
merge n:1 fips using data/`crop'1980sample
drop _merge
keep if year>=1950 & year<=2005
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
local start 1980 /*starting year*/
local end 2000 /*ending year*/
local hist 1960 /*start of history period*/
local mid 1970 /*middle of history period*/
	gen prec_lo=0
	gen prec_hi=0
	replace prec_lo = (precsmooth`end'- precsmooth`start') if precsmooth`start'<`j' & precsmooth`end'<=`j'
	replace prec_hi = (precsmooth`end' - precsmooth`start') if precsmooth`start'>`j' & precsmooth`end'>`j'
	replace prec_lo = (`j' - precsmooth`start') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	replace prec_hi = (precsmooth`end' - `j') if precsmooth`start'<=`j' & precsmooth`end'>`j'
	replace prec_lo = (precsmooth`end' - `j') if precsmooth`start' >`j' & precsmooth`end'<=`j'
	replace prec_hi = (`j' - precsmooth`start') if precsmooth`start' >`j' & precsmooth`end'<=`j'
gen lower = (dday0Csmooth`end' - dday`t'Csmooth`end') - (dday0Csmooth`start' - dday`t'Csmooth`start')
gen higher = dday`t'Csmooth`end' - dday`t'Csmooth`start'
gen higher_past = dday`t'Csmooth`start' - dday`t'Csmooth`hist'
egen state_hp = mean(higher_past), by(stfips) 
gen higher_p2 = dday`t'Csmooth`start' - dday`t'Csmooth`mid'
gen higher_p1 = dday`t'Csmooth`mid' - dday`t'Csmooth`hist'
egen varhigh = rowsd(dday`t'C195? dday`t'C196? dday`t'C197? dday`t'C1980)
foreach j in higher_past higher_p2 higher_p1 state_hp varhigh { 
sum `j' if in80sample==1, detail
replace `j' = `j' - r(mean)
}
gen log`crop'yield_diff`start'`end' = ln(`crop'yieldsmooth`end') - ln(`crop'yieldsmooth`start')
gen high_inter = higher_past*higher
gen high_state = state_hp*higher
gen high_p1 = higher_p1*higher
gen high_p2 = higher_p2*higher
gen var_higher = varhigh*higher

areg log`crop'yield_diff`start'`end' lower higher high_inter higher_past prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=`crop'_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store one_inter

areg log`crop'yield_diff`start'`end' lower higher high_p2 higher_p2 prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=`crop'_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store two_inter

areg log`crop'yield_diff`start'`end' lower higher high_state prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=`crop'_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store state_inter

local start 1980 /*starting year*/
local end 2000 /*ending year*/
local hist 1960 /*start of history period*/
local mid 1970 /*middle of history period*/
loc crop corn
areg log`crop'yield_diff`start'`end' lower higher var_higher varhigh prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=`crop'_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "29C"
estadd loc pt "42cm"
estimates store sd_inter


lab var lower "GDD below"
lab var higher "GDD above"
lab var prec_lo "Precip below"
lab var prec_hi "Precip above"
lab var higher_past "GDD above, 1960-1980"
lab var higher_p1 "GDD above, 1960-1970"
lab var higher_p2 "GDD above, 1970-1980"
lab var high_inter "GDD above*GDD above, 1960-1980"
lab var high_p1 "GDD above*GDD above, 1960-1970"
lab var high_p2 "GDD above*GDD above, 1970-1980"
lab var high_state "GDD above*State GDD above, 1960-1980"
lab var var_higher "GDD above*SD GDD above, 1950-1980"
gen bush_higher=.
gen gradpct_higher=.
lab var bush_higher "GDD above*Republican vote share, 2000"
lab var gradpct_higher "GDD above*High school grad. rate, 1980"

esttab sd_inter one_inter two_inter state_inter bushvs highschool  using output/Table5.tex, ///
	replace b(%10.5f) se l sfmt(%10.2f %10.0f %10.3f %8s) ///
	scalars("ymean Mean of Dep Variable" "N Observations" "r2 R squared" "fe Fixed Effects") ///
	star(* 0.10 ** 0.05 *** 0.01) compress keep(higher high_inter high_p2 high_state var_higher bush_higher gradpct_higher) nonotes width(\hsize) ///
	nomtitles
