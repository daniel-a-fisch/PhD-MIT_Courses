
* Make Table 1 in Burke and Emerick

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


local tms 29 28
foreach t of local tms {  //looping over thresholds
use data/yield_piecewise, clear
	if `t'==29 {
		local j 42
		}
	if `t'==28 {
		local j 50
		}
	//define piecewise precip variables
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
reg logcornyield_diff1980_2000 lower higher prec_lo prec_hi if longitude>-100 [aweight=corn_area_smooth1980], cl(stfips)
est sto reg1_`t'
estadd loc fe "None"
estadd loc tm "`t'C"
estadd loc pr "`j'cm"
areg logcornyield_diff1980_2000 lower higher prec_lo prec_hi if longitude>-100 [aweight=corn_area_smooth1980], a(stfips) cl(stfips)
sum higher if e(sample), detail
sum prec_lo if e(sample), detail
sum prec_hi if e(sample), detail
est sto reg2_`t'
estadd loc fe "State"
estadd loc tm "`t'C"
estadd loc pr "`j'cm"

* repeat for panel
use data/us_panel, clear
tostring fips, gen(fipschar)
replace fipschar="0"+fipschar if length(fipschar)==4
gen stfips=substr(fipschar,1,2)
destring stfips, replace
drop fipschar
gen logcornyield = log(cornyield)
egen corn_area_78_02 = mean(corn_area) if year>=1978 & year<=2002, by(fips)
gen lower = dday0C - dday`t'C
gen higher = dday`t'C
gen prec_lo = (prec - `j')*(prec<=`j')
gen prec_hi = (prec - `j')*(prec>`j')
areg logcornyield lower higher prec_lo prec_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], a(fips) cl(stfips)
est sto reg3_`t'
estadd loc fe "Cty, Yr"
estadd loc tm "`t'C"
estadd loc pr "`j'cm"
set matsize 800
egen statebyyear = group(stfips year)

areg logcornyield lower higher prec_lo prec_hi i.statebyyear if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], a(fips) cl(stfips) 
est sto reg4_`t'
estadd loc fe "Cty, State-Yr"
estadd loc tm "`t'C"
estadd loc pr "`j'cm"
}

*write out latex table
lab var lower "GDD below threshold"
lab var higher "GDD above threshold"
lab var prec_lo "Precip below threshold"
lab var prec_hi "Precip above threshold"
set matsize 10000
esttab reg* using output/Table1.tex, drop( *.year *.statebyyear )  ///
	replace b(%10.4f %10.4f %10.4f %10.4f %10.4f) se l sfmt(%10.0f %10.3f %8s %5s %5s) ///
	scalars("N Observations" "r2 R squared" "fe Fixed Effects" "tm T threshold" "pr P threshold") ///
	star(* 0.10 ** 0.05 *** 0.01) compress order(lower higher prec*) nonotes width(\hsize) ///
	mtitles("Diffs" "Diffs" "Panel" "Panel" "Diffs" "Diffs" "Panel" "Panel") 

	

