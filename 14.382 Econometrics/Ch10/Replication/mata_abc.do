/* This part is the mata function for conducting analytical bias correction calculation*/
quietly {
mata
dem = st_data(., "dem")
cons = J(rows(dem), 1, 1)
lag = st_data(., ("lgdp_l1", "lgdp_l2", "lgdp_l3", "lgdp_l4"))
yr = st_data(., 8..25)
id = st_data(., 27..172)
N = cols(id) + 1
res = st_data(., "res")
X = (cons, dem, lag, yr, id)
mata drop dem cons lag yr id
T = rows(res)/N
bias4 = abc(X, res, 4, T)'
st_matrix("bias4", bias4)

end
}
