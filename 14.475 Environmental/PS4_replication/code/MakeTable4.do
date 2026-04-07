
* Replication data for Table 4 in Burke and Emerick

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


use data/farmCensusData, clear 
keep farm_acres farm_num fips year 
reshape wide farm_acres farm_num, i(fips) j(year)
merge 1:1 fips using data/yield_piecewise
drop _merge
merge 1:1 fips using data/corn1980sample.dta
drop _merge
foreach i in 1978 1982 1987 1992 1997 2002 { 
foreach j in farm_num farm_acres  {
gen log`j'`i' = ln(`j'`i')
}
}

local j 42
local t 29
gen lower = dday0_`t'C_diff1980_2000 
gen higher = dday`t'C_diff1980_2000
gen prec_lo = prec`j'_lo
gen prec_hi = prec`j'_hi
*calculate changes in variables using census data, 1978-2002
gen logfarms_diff = ((ln(farm_num2002) + ln(farm_num1997)) / 2) - ((ln(farm_num1978) + ln(farm_num1982)) / 2)
gen logacres_diff = ((ln(farm_acres2002) + ln(farm_acres1997)) / 2) - ((ln(farm_acres1978) + ln(farm_acres1982)) / 2)
gen land_base = (farm_area_total_78 + farm_area_total_82)/2  //weights for land value regressions
gen aveland = (farm_area_total_78 + farm_area_total_82) / 2
*corn area variables
gen chgcornarea = log(corn_area_smooth2000) - log(corn_area_smooth1980)
gen cornpct2000= (corn_areaharvest_97/farm_area_crop_97 + corn_areaharvest2002/farm_area_crop2002)/2
gen cornpct1980 = (corn_areaharvest_78/farm_area_crop_78 + corn_areaharvest_82/farm_area_crop_82)/2
gen chgcornpct = cornpct2000 - cornpct1980
* population
gen chgpop = ln(pop2000) - ln(pop1980)


* define sample.  we are dropping 5 observations with really high temperature change. see footnote 27 for a discussion
gen samp1 = (longitude>-100 & in80sample==1)
gen samp2 = (samp1==1 & higher<50 & lower<290)
gen rural = density2000<400


* Run regressions and write to table

areg chgcornarea lower higher prec_lo prec_hi if samp2==1 [aweight=land_base], a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "`t'"
estadd loc pt "`j'"
estimates store cornarea1

areg chgcornpct lower higher prec_lo prec_hi if samp2==1 [aweight=land_base], a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "`t'"
estadd loc pt "`j'"
estimates store cornshare1

areg logacres_diff lower higher prec_lo prec_hi if samp2==1 [aweight=land_base], a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "`t'"
estadd loc pt "`j'"
estimates store areafarms1

areg logfarms_diff lower higher prec_lo prec_hi if samp2==1 [aweight=land_base], a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "`t'"
estadd loc pt "`j'"
estimates store numfarms1

areg chgpop lower higher prec_lo prec_hi if samp2==1 [aweight=land_base], a(stfips) cl(stfips)
estadd ysumm 
estadd loc fe "State"
estadd loc tt "`t'"
estadd loc pt "`j'"
estimates store pop1


* write out table
lab var lower "GDD below threshold"
lab var higher "GDD above threshold"
lab var prec_lo "Precip below threshold"
lab var prec_hi "Precip above threshold"
esttab cornarea1 cornshare1 areafarms1 numfarms1 pop1 using output/Table4.tex, ///
	replace b(%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f) se l sfmt(%10.0f %10.3f %10.3f %8s %5s %5s) ///
	scalars("N Observations" "ymean Mean of Dep Variable" "r2 R squared" "fe Fixed Effects" "tt T threshold" "pt P threshold") ///
	star(* 0.10 ** 0.05 *** 0.01) compress order(lower higher prec*) align(ccccc) ///
	mtitles("Corn area" "Corn share" "Farm area" "Num. farms" "Population") width(\hsize) nonotes


