library(lavaan)
library(psych)
library(semPlot)
library(car)
library(sirt)
library(dplyr)

# set working directory
setwd("D:/BaiduSyncdisk/weak link/HCP_abilities/SEM")
dat1<-read.table("HCPS1200_behavioral_gfactor_modeling.csv", sep = ",",header = T)
head(dat1)

#z-score normalize the data columns
dat1z <- as.data.frame(scale(dat1[ ,4:20])) 
colnames(dat1z) = c("PMAT24_A_CRz", "VSPLOT_TCz", "ListSort_Unadjz", "PicSeq_Unadjz", "IWRD_TOTz", "PicVocab_Unadjz", 
                    "ReadEng_Unadjz", "CardSort_Unadjz", "Flanker_Unadjz", "ProcSpeed_Unadjz", "Language_Task_Story_Accz", 
                    "Language_Task_Math_Accz", "WM_Task_2bk_Place_Accz", "WM_Task_2bk_Tool_Accz", "WM_Task_2bk_Body_Accz", 
                    "Relational_Task_Match_Accz", "Relational_Task_Rel_Accz")
head(dat1z)
dat1b <- cbind(dat1, dat1z)
head(dat1b)

dat2<-read.table("subjid.csv", sep = ",",header = T)
head(dat2)
dat <- merge(dat1b, dat2, by="Subject")
head(dat)
##==================================

Model2MG <- 'g =~ 
#gf
PMAT24_A_CRz + VSPLOT_TCz +
#gc
PicVocab_Unadjz  + ReadEng_Unadjz + 
#mem
PicSeq_Unadjz + IWRD_TOTz + # + ListSort_Unadj +
#speed
CardSort_Unadjz + Flanker_Unadjz + ProcSpeed_Unadjz 

cry =~ b*PicVocab_Unadjz  + b*ReadEng_Unadjz 
mem =~ c*PicSeq_Unadjz + c*IWRD_TOTz  #+ ListSort_Unadjz
spd =~ CardSort_Unadjz + Flanker_Unadjz + ProcSpeed_Unadjz

PMAT24_A_CRz~0*1
PicVocab_Unadjz~0*1
PicSeq_Unadjz~0*1
CardSort_Unadjz~0*1

g~1
cry~1
mem~1
spd~1

g~~g
cry~~cry
mem~~mem
spd~~spd
'
fit2MG <- sem(model = Model2MG, data = dat, missing='ML',orthogonal=TRUE, std.lv=F)
summary(fit2MG, fit.measures=TRUE, standardized=TRUE)

score=predict(fit2MG)
data<-cbind(dat,score)
write.csv(data,file="general_ability.csv")

# visualize to SEM model structure with estimated parameters
semPaths(fit2MG, intercept = FALSE, whatLabel = "est", layout='tree',
         edge.label.cex=1, label.cex=1.5, edge.width=1,
         sizeMan=7,residuals = FALSE)


