library(dplyr)
demo = read.csv('Demographics/PTDEMOG.csv')

#FDG PET

##1.obtain all the patients through ADNI2 and have 5 or more PET images available
fdg = read.csv('PET info/FDG/BAIPETNMRCFDG_11_22_19.csv')

#select subjects whose information on the 4th visit is available,
#and have at least one out of the first 3 visits available
#fdg.id = fdg%>%count(RID)
#all_ids_sum = fdg.id%>%count(n)
#fdg.id = fdg.id%>%filter(n>=3 & n<=6)%>%filter(!RID  %in% c(1205, 1157))

#save the month as time for further selection
fdg.info = fdg%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))

#select those whose 5th visit cognitive score is available
adas.5th = adas.fdg%>%filter(time == 24)%>%filter(TOTAL11 != -1)

#further select who have at least 3 visits(image and score)
#(out of the first 4) available
adas.5 = adas.fdg%>%filter(RID %in% adas.5th$RID)%>%
  filter(time < 24)%>%filter(TOTAL11 != -1)%>%
  count(RID)%>%filter(n>=3)

fdg.5 = fdg.info%>%filter(RID %in% adas.5$RID)%>%
  filter(time < 24)%>%count(RID)%>%filter(n>=3)
#check sample size for each time point
time.point = sort(unique(adas.fdg$time))
for(i in 1:(length(time.point)-1)){
  tmp = fdg.info%>%filter(time == time.point[i])
  tmp.1 = adas.fdg%>%filter(RID %in% tmp$RID)%>%filter(time == time.point[i+1])
  print(c(time.point[i], dim(tmp)[1], dim(tmp.1)[1]))
}
length(unique(fdg.info$RID))
#the time point of the last image for each subject
#fdg.last = fdg.info%>%group_by(RID)%>%arrange(time)%>%filter(row_number() == n())%>%ungroup(RID)

#fdg.diff = fdg.info%>%group_by(RID)%>%arrange(time)%>%
#   mutate(diff = time - lag(time, default = first(time)))%>%
#   arrange(RID)
# fdg.info.tmp = fdg.diff%>%filter(row_number() == n())
fdg.info = fdg.info%>%filter(RID %in% fdg.5$RID)%>%filter(time <= 18)
fdg.info = fdg.info[-c(759, 760),]


#get the outcome score of each subject
fdg.score = all.info%>%filter(RID %in% fdg.5$RID)%>%filter(time == 24)

#get the sequence of score of each subject
fdg.score.seq = all.info%>%inner_join(fdg.info, by = c('RID', 'VISCODE2', 'time'))

#get sequence of baseline information of each subject (age, gender)
fdg.baseline = registry%>%filter(RID %in% fdg.info$RID & VISCODE2 == 'bl')%>%
  select(RID, VISCODE2, EXAMDATE)%>%inner_join(demo.1, by = 'RID')%>%
  mutate(age = year(EXAMDATE) - PTDOBYY)

##2.get the subject id
roster = read.csv('Enrollment/ROSTER.csv')
##check one-to-one relationship of RID and PTID
tmp = roster%>%select(RID, SITEID, PTID)%>%group_by(RID)%>%summarise(n = n_distinct(PTID))%>%filter(n>1)
fdg_id = roster%>%select(RID, PTID)%>%filter(RID %in% fdg.info$RID)%>%distinct(RID, PTID)
write.csv(paste(fdg_id$PTID,collapse = ','), 'FDG_ID.csv')

#AV45 PET
av45 = read.csv('PET info/AV45/BAIPETNMRCAV45_11_22_19.csv')

#select subjects whose number of records is no more than 4
av.ids = av45%>%count(RID)
count(av.ids, n)
av.ids = av.ids%>%filter(n>=3 & n<=4)

#further select subjects whose lag between the latest two records is 24(months)
av.info = av45
av.info$VISCODE2[av.info$VISCODE2 == ''] = 'm72'
av.info$VISCODE2[av.info$VISCODE2 == 'init'] = 'm72'

av.info = av.info%>%filter(RID %in% av.ids$RID)%>%
  mutate(time = ifelse(VISCODE2 == 'bl', 'm0', VISCODE2))%>%
  mutate(time = as.numeric(sub('.', '', time)))
av.diff = av.info%>%group_by(RID)%>%arrange(time)%>%
  mutate(diff = time - lag(time, default = first(time)))%>%
  arrange(RID)
count(av.diff%>%filter(row_number() == n())%>%ungroup(RID), diff)
av.24 = av.diff%>%filter(row_number() == n())%>%filter(diff[row_number() == n()] == 12)%>%
  ungroup(RID)
av.info = av45%>%filter(RID %in% av.24$RID)
