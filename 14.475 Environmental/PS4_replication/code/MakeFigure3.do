
* Make Burke and Emerick Figure 3

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


postfile fig cons secons temp using output/Fig3_data_LD, replace
use data/yield_piecewise, clear	
local t 29
local j 42
local t1=`t'+1
di `t1'
gen lower = dday0_`t'C_diff1980_2000
gen higher = dday`t'C_diff1980_2000
areg logcornyield_diff1980_2000 lower higher prec`j'_lo prec`j'_hi if longitude>-100 & cornsample [aweight=corn_area_smooth1980], a(stfips) robust cl(stfips)
gen incornreg = e(sample)
*save means in locals
foreach i in lower higher prec`j'_lo prec`j'_hi { 
sum `i' if incornreg==1
scalar `i'_m = r(mean) 
}
*loop through and calculate predicted ln yield for values of temp from 0 to 40
gen dm_prec`j'_lo = prec`j'_lo - prec`j'_lo_m
gen dm_prec`j'_hi = prec`j'_hi - prec`j'_hi_m
forvalues k=0(1)`t' { 
qui gen lower_`k' = lower - `k' 
qui areg logcornyield_diff1980_2000 lower_`k' higher dm_prec`j'_lo dm_prec`j'_hi if longitude>-100 & cornsample [aweight=corn_area_smooth1980], a(stfips) robust cl(stfips)
post fig (_b[_cons]) (_se[_cons]) (`k')
}
forvalues k=`t1'(1)40 { 
qui gen higher_`k' = higher - `k' + `t'
qui areg logcornyield_diff1980_2000 lower_`t' higher_`k' dm_prec`j'_lo dm_prec`j'_hi if longitude>-100 & cornsample [aweight=corn_area_smooth1980], a(stfips) robust cl(stfips)
post fig (_b[_cons]) (_se[_cons]) (`k')
}
postclose fig

*now do it for the panel and merge results together
postfile fig cons_p secons_p temp using output/Fig3_data_Panel, replace
use data/us_panel, clear
local t 29
local j 42
local t1=`t'+1
tostring fips, gen(fipschar)
replace fipschar="0"+fipschar if length(fipschar)==4
gen stfips=substr(fipschar,1,2)
destring stfips, replace
drop fipschar
gen logcornyield = log(cornyield)
egen corn_area_78_02 = mean(corn_area) if year>=1978 & year<=2002, by(fips)
gen lower = dday0C - dday`t'C
gen higher = dday`t'C
gen prec`j'_lo = (prec - `j')*(prec<=`j')
gen prec`j'_hi = (prec - `j')*(prec>`j')
xtreg logcornyield lower higher prec`j'_lo prec`j'_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
gen incornreg = e(sample)
*save means in locals
foreach i in lower higher prec`j'_lo prec`j'_hi { 
sum `i' if incornreg==1
scalar `i'_m = r(mean) 
}
*loop through and calculate predicted ln yield for values of temp from 0 to 40
gen dm_prec`j'_lo = prec`j'_lo - prec`j'_lo_m
gen dm_prec`j'_hi = prec`j'_hi - prec`j'_hi_m
forvalues k=0(1)`t' { 
qui gen lower_`k' = lower - `k' 
qui xtreg logcornyield lower_`k' higher dm_prec`j'_lo dm_prec`j'_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
post fig (_b[_cons]) (_se[_cons]) (`k')
}
forvalues k=`t1'(1)40 { 
qui gen higher_`k' = higher - `k' + `t'
qui xtreg logcornyield lower_`t' higher_`k' dm_prec`j'_lo dm_prec`j'_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
post fig (_b[_cons]) (_se[_cons]) (`k')
}
postclose fig
use output/Fig3_data_Panel, clear
merge 1:1 temp using output/Fig3_data_LD
drop _merge
gen predlow = cons - 2.04*secons 
gen predhigh = cons + 2.04*secons
gen predlow_p = cons_p - 1.96*secons_p 
gen predhigh_p = cons_p + 1.96*secons_p
gen ncons = cons - cons[1] /*normalization to get y intercept the same*/
gen ncons_p = cons_p - cons_p[1] /*normalization to get y intercept the same*/
gen npredlow = ncons - 2.04*secons
gen npredhigh = ncons + 2.04*secons
gen ln = 0
twoway (rarea npredhigh npredlow temp, sort col(gs14)) (line ncons temp, lc(black) lw(medthick)) (line ncons_p temp, lp(dash) lw(medthick) lcolor(gray))  ///
	(line ln temp, lp(dot) lw(medthin)), legend(order(2 "LD" 3 "Panel")) xtitle("Temperature (C)") ytitle("Normalized predicted log yields") xsize(7) ysize(6) scheme(s1color)
graph export output/Figure3.pdf, as(pdf) replace

