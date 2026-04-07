

clear all
set mem 1g
set matsize 10000
set more off, permanently

cd /Documents/Dropbox/adaptation/replication/  //Navigate to replication folder on your own machine


****  To make some of the plots, going to evaluate coefficients at 1C warming and 10% change in precip
use fips prec_smooth* corn_area_smooth* longitude using data/yield_piecewise, clear
merge 1:1 fips using data/a1b_change_gdd
keep if _merge==3
drop _merge
merge 1:1 fips using data/a1b_change_prec
keep if _merge==3
drop _merge
gen area = (corn_area_smooth1980 + corn_area_smooth2000)/2
gen prec = (prec_smooth1980 + prec_smooth1980)/2  

* generate changes in temperature we need to evaluate coefficients
local models unif1c cccmat63 cnrm csiro gfdl0 gfdl1 gissaom gisseh gisser iap inmcm3 ipsl ///
	mirochires mirocmedres echam mri ccsm pcm hadcm3
loc i 29
foreach m of local models {
	qui summ gdd0_`i'_`m' if longitude>-100 [aweight=area]
	global gddchglo_`i'_`m' = r(mean)
	qui summ gdd`i'_`m' if longitude>-100 [aweight=area]
	global gddchghi_`i'_`m' = r(mean)	
	}
	

*changes in precip. we need the difference in inches; model estimates are % changes
loc j 42
foreach m of local models {
	qui summ prec_`m' if longitude>-100 [aweight=area]
	local prec_chg = r(mean)
	cap drop prec1 *lod *hid *lod1 *hid1 lodif hidif
	gen prec`j'_lod = (prec - `j')*(prec<=`j')
	gen prec`j'_hid = (prec - `j')*(prec>`j')
	gen prec1 = prec*(1+`prec_chg')
	gen prec`j'_lod1 = (prec1 - `j')*(prec1<=`j')
	gen prec`j'_hid1 = (prec1 - `j')*(prec1>`j')
	gen lodif = prec`j'_lod1 - prec`j'_lod
	summ lodif if longitude>-100 [aweight=area]
	global plo_`j'_`m' = r(mean)
	gen hidif = prec`j'_hid1 - prec`j'_hid
	summ hidif if longitude>-100 [aweight=area]
	global phi_`j'_`m' = r(mean)
	}
	

* calculate climate change impacts for different combinations of temp and precip thresholds, maize
capture postutil clear
postfile impact str25(reg_model) str25(clim_model) tmp_thres prec_thres temp_b temp_se prec_b prec_se comb_b comb_se using output/Fig6_data, replace
local models unif1c cccmat63 cnrm csiro gfdl0 gfdl1 gissaom gisseh gisser iap inmcm3 ipsl ///
	mirochires mirocmedres echam mri ccsm pcm hadcm3

local j 42  //precip threshold
local t 29  //temperature threshold

* run the long diffs regression
use data/yield_piecewise, clear
gen lower = dday0_`t'C_diff1980_2000
gen higher = dday`t'C_diff1980_2000
areg logcornyield_diff1980_2000 lower higher prec`j'_lo prec`j'_hi if longitude>-100 & cornsample [aweight=corn_area_smooth1980], a(stfips) cl(stfips)
foreach m of local models {
	lincom _b[lower]*${gddchglo_`t'_`m'} + _b[higher]*${gddchghi_`t'_`m'}
	local a = r(estimate)
	local b = r(se)
	lincom _b[prec`j'_lo]*${plo_`j'_`m'} + _b[prec`j'_hi]*${phi_`j'_`m'}
	local c = r(estimate)
	local d = r(se)
	lincom _b[lower]*${gddchglo_`t'_`m'} + _b[higher]*${gddchghi_`t'_`m'} + _b[prec`j'_lo]*${plo_`j'_`m'} + _b[prec`j'_hi]*${phi_`j'_`m'}
	post impact ("diffs") ("`m'") (`t') (`j') (`a') (`b') (`c') (`d') (r(estimate)) (r(se))
	}

*panel regression
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
gen prec`j'_lo = (prec - `j')*(prec<=`j')
gen prec`j'_hi = (prec - `j')*(prec>`j')
qui xtreg logcornyield lower higher prec`j'_lo prec`j'_hi i.year if longitude>-100 & year>=1978 & year<=2002 [aweight=corn_area_78_02], fe i(fips) vce(cl stfips)
foreach m of local models {
	lincom _b[lower]*${gddchglo_`t'_`m'} + _b[higher]*${gddchghi_`t'_`m'}
	local a = r(estimate)
	local b = r(se)
	lincom _b[prec`j'_lo]*${plo_`j'_`m'} + _b[prec`j'_hi]*${phi_`j'_`m'}
	local c = r(estimate)
	local d = r(se)
	lincom _b[lower]*${gddchglo_`t'_`m'} + _b[higher]*${gddchghi_`t'_`m'} + _b[prec`j'_lo]*${plo_`j'_`m'} + _b[prec`j'_hi]*${phi_`j'_`m'}
	post impact ("panel") ("`m'") (`t') (`j') (`a') (`b') (`c') (`d') (r(estimate)) (r(se))
	}

postclose impact




* Plot impacts for diffs model- temp, precip, and combined
use output/Fig6_data, clear
drop if clim_model=="unif1c"
keep if tmp_thres==29 & prec_thres==42
keep if reg_model=="diffs"
gen tlo = temp_b - 1.96*temp_se
gen thi = temp_b + 1.96*temp_se
gen plo = prec_b - 1.96*prec_se
gen phi = prec_b + 1.96*prec_se
gen lo = comb_b - 1.96*comb_se
gen hi = comb_b + 1.96*comb_se
gsort -comb_b
gen n1 = _n-0.15
gen n2 = _n
gen n3 = _n+0.15
gen yl = lo-0.1
replace yl = -0.9 if yl<-0.9
twoway (rcap plo phi n2, lcolor(blue)) (scatter prec_b n2, mcolor(blue) m(c)) ///
	(rcap tlo thi n1, lcolor(black)) (scatter temp_b n1, mcolor(black) m(c)) ///
	(rcap lo hi n3, lcolor(red)) (scatter comb_b n3, mcolor(red) m(c)) ///
	(scatter yl n2, m(i) mlabel(clim_model) mlabangle(90) mlabposition(8)), ///
	ytitle(Change in log corn yield) yline(0, lcolor(black) lstyle(dash)) xlabel(none) ///
	legend(row(1) order(1 "Precip" 3 "Temperature" 5 "Combined")) xtitle("") xsize(8) ysize(5) yline(-0.2(-0.2)-1, lst(grid)) name(g1, replace)

* plot impacts, diffs versus panel
use output/Fig6_data, clear
drop if clim_model=="unif1c"  //uniform warming scenario
sort clim_model reg_model
gen tosort = comb_b //making a variable so that we can sort just on the long diffs projections
replace tosort = tosort[_n-1] if reg_model=="panel"
gsort -tosort +reg_model
egen n = group(tosort)
replace n = 19-n
gen n1 = n+0.15 if reg_model=="diffs"
replace n1 = n-0.15 if reg_model=="panel"
gen lo = comb_b - 1.96*comb_se
gen hi = comb_b + 1.96*comb_se
gen yl = lo-0.1
replace yl = -0.9 if yl<-0.9
summ comb_b if reg_model=="diffs", det
gen medn = `r(p50)'
twoway (line medn n, lc(gray) lp(dash)) /// 
	(rcap lo hi n1 if reg_model=="diffs", lc(red)) (scatter comb_b n1 if reg_model=="diffs", m(c) mcolor(red)) ///
	(rcap lo hi n1 if reg_model=="panel", lc(black)) (scatter comb_b n1 if reg_model=="panel", m(c) mcolor(black)) ///
	(scatter yl n if reg_model=="diffs", m(i) mlabel(clim_model) mlabangle(90) mlabposition(8)), ///
	ytitle(Change in log corn yield) yline(0, lcolor(black) lstyle(dash)) xlabel(none) ///
	legend(row(1) order(2 "LD" 4 "Panel")) xtitle("") xsize(8) ysize(5) yline(-0.2(-0.2)-1, lst(grid)) name(g2, replace)

graph combine g1 g2, r(2) xsize(10) ysize(10)
graph export output/Figure6.pdf, as(pdf) replace	
