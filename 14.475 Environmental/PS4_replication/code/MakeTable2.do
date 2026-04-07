
* Replication data for Table 2 in Burke and Emerick


clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine



// First generate datasets for the four 20-year differences that we're going to use
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
tempfile obsforit 
save "`obsforit'", replace
*put starting points in the list for the local start
foreach start in 1955 1960 1975 1980 { 
loc diffl 20  //just looking at 20-year differences 
local end = `start' + `diffl'
use "`obsforit'", clear
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
gen log`crop'yield_diff`start'`end' = ln(`crop'yieldsmooth`end') - ln(`crop'yieldsmooth`start')
keep log`crop'yield_diff`start'`end' lower higher prec_lo prec_hi longitude in80sample fips `crop'_areasmooth`start' stfips
rename log`crop'yield_diff`start'`end' logyielddiff 
rename `crop'_areasmooth`start' cropweight
gen startyear = `start'
gen endyear = `end'
tempfile ds`start'`end'
save "`ds`start'`end''", replace
}



// now append datasets together and run regressions
use "`ds19551975'", clear 
append using "`ds19601980'"
append using "`ds19751995'"
append using "`ds19802000'"
foreach q in 1955 1960 { 
gen yearstupid`q' = startyear==`q'
gen shitty`q' = yearstupid`q'*cropweight
egen initialweight`q' = max(shitty`q'), by(fips)
}
areg logyielddiff lower higher prec_lo prec_hi i.startyear if longitude>-100 & in80sample==1 & inlist(startyear,1955,1975) [aweight=initialweight1955], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State Yr"
estadd loc tm "`t'"
estadd loc pr "`j'"
estimates store `crop'sfe55
areg logyielddiff lower higher prec_lo prec_hi i.startyear if longitude>-100 & in80sample==1 & inlist(startyear,1960,1980) [aweight=initialweight1960], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State Yr"
estadd loc tm "`t'"
estadd loc pr "`j'"
estimates store `crop'sfe60
xi: xtreg logyielddiff lower higher prec_lo prec_hi i.startyear if longitude>-100 & in80sample==1  & inlist(startyear,1955,1975) [aweight=initialweight1955], fe i(fips) vce(cl stfips)
estadd ysumm 
estadd loc fe "Cty Yr"
estadd loc tm "`t'"
estadd loc pr "`j'"
estimates store `crop'cfe55
xi: xtreg logyielddiff lower higher prec_lo prec_hi i.startyear if longitude>-100 & in80sample==1  & inlist(startyear,1960,1980) [aweight=initialweight1960], fe i(fips) vce(cl stfips)
estadd ysumm 
estadd loc fe "Cty Yr"
estadd loc tm "`t'"
estadd loc pr "`j'"
estimates store `crop'cfe60

lab var lower "GDD below threshold"
lab var higher "GDD above threshold"
lab var prec_lo "Precip below threshold"
lab var prec_hi "Precip above threshold"

esttab cornsfe55 corncfe55 cornsfe60 corncfe60 using output/Table2.tex, drop(_cons 19* *_Ist*) ///
	replace b(%10.4f %10.4f %10.4f %10.4f %10.4f) se l sfmt(%10.0f %10.3f %8s %5s %5s) ///
	scalars("N Observations" "r2 R squared" "fe Fixed Effects" "tm T threshold" "pr P threshold") ///
	star(* 0.10 ** 0.05 *** 0.01) compress order(lower higher prec*) nonotes width(\hsize) ///
	mtitles("1955-1995" "1955-1995" "1960-2000" "1960-2000") 
