#Run preprocess.R, variable selection.R and subject.R first (in this order)
##get file names of images (the codes below should be adjusted based on images, e.g ROIs)
roi.path = function(roi){
  #get the file_names (PIDs)
  cur_path = getwd()
  #should change the folder name according to your need
  folder_name = '/Data/ROIDIF12/'
  image_path = paste0(cur_path, folder_name, roi)
  file_names = dir(image_path)
  image_names = list()
  #for each subject
  for(i in file_names){
    cpath = paste0(image_path, '/', i)
    sub_files = dir(cpath)
    sub_names = c()
    #save all the images
    for(j in 1:length(sub_files)){
      spath = paste0(cpath, '/', sub_files[j])
      sub_names = c(sub_names, spath)
    }
    image_names[[i]] = sub_names
  }
  return(image_names)
}
bp_names = roi.path('BP')
lt_names = roi.path('LT')

#time points for images
image.seq = fdg.score.seq%>%inner_join(fdg_id, by = 'RID')

#list all the subjects by visit
times = unique(fdg.info$VISCODE2)
time_id = list()
for(i in times[-7]){
  time_id[[i]] = image.seq%>%arrange(RID, time)%>%group_by(RID)%>%
    mutate(index = row_number())%>%ungroup(RID)%>%filter(VISCODE2 == i)%>%
    select(RID, PTID, TOTAL11, TOTAL13, index, DXCURREN, time)
}
fdg.info%>%count(VISCODE2)
image.seq%>%count(VISCODE2)

#list all the visit by subject
id.vis = list()
for(i in as.character(fdg_id$PTID)){
  id.vis[[i]] = image.seq%>%arrange(RID, time)%>%filter(PTID == i)%>%select(VISCODE2, RID, PTID,
                                                                            TOTAL11, TOTAL13, DXCURREN, time)
}


#all the image file paths and scores

#save the image by visit
save.by.visit = function(image_names, roi){
  
  for(i in 1:length(times)){
    sub.name = time_id[[i]]
    all.images = c()
    for(j in sub.name$PTID){
      ind = sub.name%>%filter(PTID == j)
      all.images = c(all.images, image_names[[j]][ind$index])
    }
    files.data = data.frame(images = all.images, sub.name)
    #should change saving path
    write.csv(files.data, paste('Code/Info/Imputation/',roi, '/Visit/', times[i], '.csv', sep = ''), row.names = FALSE)
  }
  
}

#save the names by rid
save.by.rid = function(image_names, roi){
  
  for(i in names(id.vis)){
    images = image_names[[i]]
    visits = id.vis[[i]]
    #should change saving path
    write.csv(data.frame(images = images, visits), paste('Code/Info/Imputation/', roi, '/RID/', i, '.csv', sep = ''))
  }
  
}

save.by.visit(bp_names, 'BP')
save.by.rid(bp_names, 'BP')
save.by.visit(lt_names, 'LT')
save.by.rid(lt_names, 'LT')

#save baseline and outcome
fdg.base.out = fdg.baseline%>%inner_join(fdg.score, by = c('RID'))%>%
  inner_join(fdg_id, by = 'RID')%>%
  select(RID, PTGENDER, age, TOTAL11, TOTAL13, VISCODE2.y, PTID, DXCURREN)
write.csv(fdg.base.out, paste('Code/base_out.csv'))

