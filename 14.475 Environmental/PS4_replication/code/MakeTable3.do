

* Replication data for Table 3 in Burke and Emerick


clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


* set temperature and precip thresholds
local j 50
local t 28

use data/us_panel, clear
tostring fips, gen(fipschar)
replace fipschar="0"+fipschar if length(fipschar)==4
gen stfips=substr(fipschar,1,2)
destring stfips, replace
drop fipschar
gen logcornyield = log(cornyield)
merge 1:1 fips year using data/othercrop_revenue.dta 
tab _merge
drop if _merge==2
drop _merge
rename winwheat_area area_winwheat

merge n:1 stfips year using data/othercrop_prices.dta
tab _merge
drop if _merge==2
drop _merge

gen lower = dday0C - dday`t'C
gen higher = dday`t'C
gen prec_lo = (prec - `j')*(prec<=`j')
gen prec_hi = (prec - `j')*(prec>`j')
foreach i in corn soy cotton hay wheat { 
gen `i'_revenue = `i'_area*`i'yield*price_`i'
gen ln_`i'_revenuepa = ln(`i'_revenue / `i'_area)
gen l_`i'yield = ln(`i'yield)
egen `i'_area_78_02 = mean(`i'_area) if year>=1978 & year<=2002, by(fips)
}
gen rice_revenue = rice_area*riceyield*(price_rice/100)
gen ln_rice_revenue = ln(rice_revenue)
egen all_revenue = rowtotal(*_revenue)
egen all_area = rowtotal(*_area)
egen all_area_78_02 = mean(all_area) if year>=1978 & year<=2002, by(fips)
gen l_all_revenue = ln(all_revenue)
gen l_all_revpa = ln(all_revenue / all_area)
gen all_revpa = all_revenue / all_area
gen corn_revpa = corn_revenue / corn_area
rename area_winwheat winwheat_area
gen lwinwheat = ln(winwheat_area)
foreach v in corn hay soy cotton wheat { 
gen `v'share = `v'_area / (all_area)
}

qui xtreg logcornyield lower higher prec_lo prec_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
gen cornsamp = e(sample)
xtreg ln_corn_revenuepa lower higher prec_lo prec_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
estadd ysumm 
estadd loc fe "Cty, Yr"
est store panel_rev_corn

xtreg l_all_revpa lower higher prec_lo prec_hi i.year if longitude>-100 & year>=1978 & year<=2002 & cornsamp==1 [aweight=all_area_78_02], fe i(fips) vce(cl stfips)
estadd ysumm 
estadd loc fe "Cty, Yr"
est store panel_rev



*long differences
merge n:1 fips using data/corn1980sample
drop _merge
keep if year>=1953 & year<=2005
keep dday`t'C dday0C prec *_area all_revpa corn_revpa *share fips stfips longitude year in80sample
reshape wide dday`t'C dday0C prec *_area all_revpa corn_revpa *share, i(fips stfips longitude in80sample) j(year) 

local start 1980 /*starting year*/
local end 2000 /*ending year*/
*forvalues k=1960(1)2000 { 
foreach k in 1980 2000 {
local l1 = `k'-1
local l2 = `k'-2
local ll1 = `k' + 1 
local ll2 = `k' + 2
gen dday0Csmooth`k' = (dday0C`l1' + dday0C`l2' + dday0C`k' + dday0C`ll1' + dday0C`ll2') / 5
gen dday`t'Csmooth`k' = (dday`t'C`l1' + dday`t'C`l2' + dday`t'C`k' + dday`t'C`ll1' + dday`t'C`ll2') / 5
gen precsmooth`k' = (prec`l1' + prec`l2' + prec`k' + prec`ll1' + prec`ll2') / 5
gen all_areasmooth`k' = (all_area`l1' + all_area`l2' + all_area`k' + all_area`ll1' + all_area`ll2') / 5
gen all_revpasmooth`k' = (ln(all_revpa`l1') + ln(all_revpa`l2') + ln(all_revpa`k') + ln(all_revpa`ll1') + ln(all_revpa`ll2')) / 5
gen corn_areasmooth`k' = (corn_area`l1' + corn_area`l2' + corn_area`k' + corn_area`ll1' + corn_area`ll2') / 5
gen corn_revpasmooth`k' = (ln(corn_revpa`l1') + ln(corn_revpa`l2') + ln(corn_revpa`k') + ln(corn_revpa`ll1') + ln(corn_revpa`ll2')) / 5
foreach h in corn hay soy cotton wheat { 
gen `h'share_smooth`k' = (`h'share`l1' + `h'share`l2' + `h'share`k' + `h'share`ll1' + `h'share`ll2') / 5
}
}
tempfile obsforit 
save "`obsforit'", replace
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
gen logrev_diff`start'`end' = all_revpasmooth`end' - all_revpasmooth`start'
gen logcorn_diff`start'`end' = corn_revpasmooth`end' - corn_revpasmooth`start'
gen logcarea_diff`start'`end' = ln(corn_areasmooth`end') - ln(corn_areasmooth`start')
foreach h in corn hay soy cotton wheat { 
gen `h'share_diff`start'`end' = `h'share_smooth`end' - `h'share_smooth`start'
}

areg logcorn_diff`start'`end' lower higher prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=corn_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estimates store ld_rev_corn

areg logrev_diff`start'`end' lower higher prec_lo prec_hi if longitude>-100 & in80sample==1 [aweight=all_areasmooth`start'], a(stfips) robust cl(stfips)
estadd ysumm 
estadd loc fe "State"
estimates store ld_rev


lab var lower "GDD below threshold"
lab var higher "GDD above threshold"
lab var prec_lo "Precip below threshold"
lab var prec_hi "Precip above threshold"
esttab panel_rev_corn ld_rev_corn panel_rev ld_rev using output/Table3.tex,  ///
	replace b(%10.4f %10.4f %10.4f %10.4f %10.4f) se l sfmt(%10.2f %10.0f %10.3f %3s %5s %5s) ///
	scalars("ymean Mean of Dep Variable" "N Observations" "r2 R squared" "fe Fixed Effects") ///
	star(* 0.10 ** 0.05 *** 0.01) compress order(lower higher prec_lo prec_hi) nonotes width(\hsize) ///
	mtitles("Panel" "Diffs" "Panel" "Diffs") drop(19* 20*) ///
	mgroups("\multicolumn{2}{c}{Corn} & \multicolumn{2}{c}{Main Spring Crops} \\ \cline{2-5}", pattern(1 0 1 0))
