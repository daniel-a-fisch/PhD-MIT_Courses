
# Make Figure 5 in Burke and Emerick.  The user will first need to run Stata file MakeFigure5.do

"%&%"<-function(x,y)paste(x,y,sep="")

setwd("/Documents/Dropbox/adaptation/replication/")

boot=read.csv("output/Fig5_data.csv")
a=substr(names(boot),1,5)
yrs=substr(names(boot)[a=="adapt"],6,14)
keep=c(2,3,4,6,7,9)
yrs=yrs[keep]
pdf(file="output/Figure5.pdf",height=6,width=6)
plot(1,type="l",ylim=c(-0.4,length(yrs)+0.5),xlim=c(-100,100),yaxt="n",ylab="",xlab="% of impact offset")
abline(v=0)
for (i in 1:length(yrs)) {
  toplot = boot[,names(boot)=="adapt"%&%yrs[i]]
  p = round(sum(toplot<0)/1000,2)  #one-sided p-value
  toplot = sort(toplot)[50:950]*100
  boxplot(toplot, horizontal=T, at=i, add=T, axes=F, col="grey")
  z=substr(yrs[i],1,4)%&%"-"%&%substr(yrs[i],6,9)
  text(median(toplot),i+0.4,z,cex=0.7)
  text(-93,i,"("%&%p%&%")",cex=0.75)
}
zz=boot[,a=="adapt"]
toplot=unlist(zz[,keep])  #20, 25, 30year estimates
ll=length(toplot)
p = round(sum(toplot<0)/ll,2)
toplot = sort(toplot)[round(ll*0.05):round(ll*0.95)]*100
boxplot(toplot, horizontal=T, at=0, add=T, axes=F, col="grey")
text(median(toplot),0.4,"Combined",cex=0.7)
text(-93,0,"("%&%p%&%")",cex=0.75)
dev.off()