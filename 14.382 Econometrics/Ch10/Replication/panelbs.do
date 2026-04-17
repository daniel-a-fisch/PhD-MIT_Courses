/* This is the mata function for generating bootstrap panel data.*/

mata
lgdp = st_data(., "lgdp")
dem = st_data(., "dem")
data = (dem, lgdp)
N = 147
T = 23
ids = mm_sample(N, N)#mm_expand(1, T)
sub = (ids - mm_expand(1, N*T))*T + mm_expand(1::T, N)
data_bs = data[sub, ]
id = (1::N)#mm_expand(1, T)
year = mm_expand(1987::2009, N)
data_bs = (year, id, data_bs)
st_store(., 3..6, data_bs)
end
