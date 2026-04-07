
# Make Figure 1 in Burke and Emerick

library(maps)
library(sp)
library(fields)
library(foreign)

"%&%"<-function(x,y)paste(x,y,sep="")

setwd("/Documents/Dropbox/adaptation/replication/")

data <- read.dta("data/yield_piecewise.dta")
data=data[is.na(data$dday0C_smooth2000)==F & data$longitude>(-100) & is.na(data$longitude[i])==F,]

tchg = data$tavg_smooth2000 - data$tavg_smooth1980
tchg_sd = tchg/data$tavg_sd
pchg = (data$prec_smooth2000 - data$prec_smooth1980)/data$prec_smooth1980


pdf(file="output/Figure1.pdf",height=6,width=12)
#temperature first
layout(matrix(c(1,1,1,2,3,3,3,4,5,5,5,6),nrow=4,byrow=F))
par(mar=c(0,0,1,1))
map(database="state",xlim=c(-100,-67))
title(main="Temperature",cex.main=2)
vals=round(tchg*100)
colz=tim.colors(max(vals,na.rm=T)-min(vals,na.rm=T))
vals = vals+abs(min(vals,na.rm=T)) #recenter so they are positive
for (i in 1:dim(data)[1]) {
#  if (is.na(data$longitude[i])==F & data$longitude[i]>(-100) ) {
    points(data$longitude[i],data$latitude[i],col=colz[vals[i]],pch=19,cex=1.3) #}
}
map(database="state",add=T)
vals=round(tchg,3)
mx=max(tchg,na.rm=T)
mn=min(tchg,na.rm=T)
lng=(mx-mn)/30
rg=seq(mn,mx,lng)
colz=tim.colors(length(rg))
par(mar=c(4,5,0,5))
hist(tchg,breaks=rg,col=colz,xlab="temp change (C)",main="",yaxt="n",ylab="",cex.axis=1.5,cex.lab=1.5)

#precip
toplot = pchg*100
toplot[toplot<(-50)]=-50
toplot[toplot>45]=45 
map(database="state",xlim=c(-100,-67))
title(main="Precipitation",cex.main=2)
vals=round(toplot)
colz=rev(tim.colors(max(vals,na.rm=T)-min(vals,na.rm=T)))
vals = vals+abs(min(vals,na.rm=T)) #recenter so they are positive
for (i in 1:dim(data)[1]) {
  if (is.na(data$longitude[i])==F & data$longitude[i]>(-100) ) {
    points(data$longitude[i],data$latitude[i],col=colz[vals[i]],pch=19,cex=1.3)}
}
map(database="state",add=T)
vals=round(toplot,3)
mx=max(toplot,na.rm=T)
mn=min(toplot,na.rm=T)
lng=(mx-mn)/30
rg=seq(mn,mx,lng)
colz=rev(tim.colors(length(rg)))
par(mar=c(4,5,0,5))
hist(toplot,breaks=rg,col=colz,xlab="precip change (%)",main="",yaxt="n",ylab="",cex.axis=1.5,cex.lab=1.5)

#cornyield
ychg = log(data$cornyield_smooth2000) - log(data$cornyield_smooth1980)
map(database="state",xlim=c(-100,-67))
title(main="Corn yield",cex.main=2)
ychg[ychg<(-0.5)]=-0.5  #censor bottom at -0.5 to make plot prettier
vals=round(ychg*100)
colz=tim.colors(max(vals,na.rm=T)-min(vals,na.rm=T))
vals = vals+abs(min(vals,na.rm=T)) #recenter so they are positive
vals=max(vals,na.rm=T) - vals
for (i in 1:dim(data)[1]) {
  if (is.na(data$longitude[i])==F & data$longitude[i]>(-100) ) {
    points(data$longitude[i],data$latitude[i],col=colz[vals[i]],pch=19,cex=1.3)}
}
map(database="state",add=T)
vals=round(ychg,3)
mx=max(ychg,na.rm=T)
mn=min(ychg,na.rm=T)
lng=(mx-mn)/30
rg=seq(mn,mx,lng)
colz=rev(tim.colors(length(rg)))
par(mar=c(4,5,0,5))
hist(ychg,breaks=rg,col=colz,xlab="change in yield (log)",main="",yaxt="n",ylab="",cex.axis=1.5,cex.lab=1.5)
dev.off()