library(plot.matrix)
library(oro.nifti)
library(dplyr)
library(stringr)
setwd('Y:/PhD/Fall 2020/RA')
cur_path = getwd()
#simulate PET images; similar to simulation.R, but with different
#number of visits for subjects, and different time difference

#subjects should have at least 3 PET images as well as an outcome that is 
#measured 12 months after the last taken PET image

#max length of visits is 9. The mechanism is that for each subject we randomly
#assign the number of visits n_i, then randomly choose n_i visits from 1:9,
#then create PET images for that subject

#there are also three types of subjects: normal, AD, normal to AD and AD to normal
#For those with normal to AD, the first image should be normal, and the last one should be 
#AD. The beginning of AD is randomly chosen among the visit in the midst. The same for
#AD to normal, but we turn the PET images backwards.

#intervals between every two consecutive visits
intervals = c(0, rep(6, 4), rep(12, 4))
times = cumsum(intervals)

#area centered by c1 and c2 are related to label; area around c3 are noise
c1 = c(10, 10, 10)
c2 = c(20, 20, 20)
c3 = c(6, 6, 20)

#max number of visits
max.vis = length(times)
#grids information
g = expand.grid(1:30, 1:30, 1:30)
g$d1 = sqrt ((g$Var1-c1[1])^2 + (g$Var2-c1[2])^2 + (g$Var3 - c1[3])^2)
g$d2 = sqrt ((g$Var1-c2[1])^2 + (g$Var2-c2[2])^2 + (g$Var3 - c2[3])^2)
g$d3 = sqrt ((g$Var1-c3[1])^2 + (g$Var2-c3[2])^2 + (g$Var3 - c3[3])^2)


#create normal control
create.nc = function(start, end, folder, set = 'train', noise, p = 0.3){
  
  #data frame for PET image as well as outcome
  new.df = data.frame()
  for(id in start:end){
    
    #select visits for PET images
    num.vis = sample(3:9, size = 1, prob = c(7:1))
    vis.index = sort(sample(1:9, num.vis))
    #if generate noisy area for this subject
    gen.noise = runif(1)<=p
    #create folder for this subject to save image
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    #base image: all pixel values around 5
    bl = array(rnorm(30^3, 5, 0.05), c(30, 30, 30))
    for(i in 1:num.vis){
      vis = vis.index[i]
      df = data.frame(id = str_pad(id, 3, pad = '0'), ad = 0, time = times[vis])
      new.df = rbind(new.df, df)
      #add random noise on bl for each visit
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      #generate noisy area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis - 0.1/(g$d3[d3.fil]+1)
      }
      #all the pixel value should exceed 0.1
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', i)
      writeNIfTI(img, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), ad = 0, time = times[vis]+12)
    new.df = rbind(new.df, df)
    
  }
  return(new.df)
}

#create normal to ad
create.ncad = function(start, end, folder, set = 'train',
                       noise, p, shrink, shrink2){
  
  new.df = data.frame()
  for(id in start:end){
    
    #select visits for PET images
    num.vis = sample(3:9, size = 1, prob = c(7:1))
    vis.index = sort(sample(1:9, num.vis))
    #decide from which one to develop ad
    dev.index = sample(2:num.vis, size = 1)
    #if generate noisy area for this subject
    gen.noise = runif(1)<=p
    #create folder for this subject to save image
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    
    #shrinkage of effect size
    s = runif(1, shrink, shrink2)
    bl = array(rnorm(30^3, 5, 0.05), c(30, 30, 30))
    
    for(i in 1:(dev.index-1)){
      
      vis = vis.index[i]
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[vis], ad = 0)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      
      #generate noise area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis 
        - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', i)
      writeNIfTI(img.cur, file.name)
    }
    for(i in dev.index:num.vis){
      vis = vis.index[i]
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[vis], ad = 1)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      #adjust effected area
      effect.area = vis - vis.index[dev.index] + 2
      d1.fil = g$d1 <= min(effect.area, 6)
      d2.fil = g$d2 <= min(1.5*effect.area, 8)
      d1.index = g[d1.fil, c('Var1', 'Var2', 'Var3')]
      d2.index = g[d2.fil, c('Var1', 'Var2', 'Var3')]
      img.cur[as.matrix(d1.index)] = img.cur[as.matrix(d1.index)] - s*0.3*vis - s*0.1/(g$d1[d1.fil]+1)
      img.cur[as.matrix(d2.index)] = img.cur[as.matrix(d2.index)] - s*0.3*vis - s*0.1/(g$d2[d2.fil]+1)
      #generate noise area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', i)
      writeNIfTI(img.cur, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[vis]+12, ad = 1)
    new.df = rbind(new.df, df)
  }
  return(new.df)
}

#create ad
create.ad = function(start, end, folder, set = 'train',
                     noise, p = 0.3, shrink = 0.8, shrink2){
  
  new.df = data.frame()
  for(id in start:end){

    #select visits for PET images
    num.vis = sample(3:9, size = 1, prob = c(7:1))
    vis.index = sort(sample(1:9, num.vis))
    #if generate noisy area for this subject
    gen.noise = runif(1)<=p
    #create folder for this subject to save image
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    #larger s is associated with more reduction in the pixel 
    #shrinkage of effect size
    s = runif(1, shrink, shrink2)

    bl = array(rnorm(30^3, 5, 0.05), c(30, 30, 30))
    for(i in 1:num.vis){
      
      vis = vis.index[i]
      df = data.frame(id = str_pad(id, 3, pad = '0'), ad = 1, time = times[vis])
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      d1.fil = g$d1 <= min(vis, 6)
      d2.fil = g$d2 <= min(1.5*vis, 8)
      d1.index = g[d1.fil, c('Var1', 'Var2', 'Var3')]
      d2.index = g[d2.fil, c('Var1', 'Var2', 'Var3')]
      img.cur[as.matrix(d1.index)] = img.cur[as.matrix(d1.index)] - s*0.3*vis 
      - s*0.1/(g$d1[d1.fil]+1)
      img.cur[as.matrix(d2.index)] = img.cur[as.matrix(d2.index)] - s*0.3*vis 
      - s*0.1/(g$d2[d2.fil]+1)
      #generate noise area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis 
        - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', i)
      writeNIfTI(img, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), ad = 1, time =times[vis]+12)
    new.df = rbind(new.df, df)
  }
  return(new.df)
  
}

#create ad to normal
create.adnc = function(start, end, folder, set = 'train',
                       noise, p, shrink, shrink2){
  
  new.df = data.frame()
  for(id in start:end){
    
    #select visits for PET images
    num.vis = sample(3:9, size = 1, prob = c(7:1))
    vis.index = sort(sample(1:9, num.vis))
    #decide from which one to develop ad
    dev.index = sample(2:num.vis, size = 1)
    #if generate noisy area for this subject
    gen.noise = runif(1)<=p
    #create folder for this subject to save image
    fold.name = paste0(cur_path, '/Data/', folder, '/', set, '/', 
                       str_pad(id, 3, pad = '0'))
    dir.create(fold.name)
    
    #shrinkage of effect size
    s = runif(1, shrink, shrink2)
    bl = array(rnorm(30^3, 5, 0.05), c(30, 30, 30))
    
    for(i in 1:(dev.index-1)){
      
      vis = vis.index[i]
      rev.vis = vis.index[num.vis - i + 1]
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[rev.vis], ad = 0)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      
      #generate noise area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis 
        - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', num.vis - i + 1)
      writeNIfTI(img.cur, file.name)
    }
    for(i in dev.index:num.vis){
      
      vis = vis.index[i]
      rev.vis = vis.index[num.vis - i + 1]
      df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[rev.vis], ad = 1)
      new.df = rbind(new.df, df)
      img.cur = bl + array(rnorm(30^3, 0, noise), c(30, 30, 30))
      #adjust effected area
      effect.area = vis - vis.index[dev.index] + 2
      d1.fil = g$d1 <= min(effect.area, 6)
      d2.fil = g$d2 <= min(1.5*effect.area, 8)
      d1.index = g[d1.fil, c('Var1', 'Var2', 'Var3')]
      d2.index = g[d2.fil, c('Var1', 'Var2', 'Var3')]
      img.cur[as.matrix(d1.index)] = img.cur[as.matrix(d1.index)] - s*0.3*vis - s*0.1/(g$d1[d1.fil]+1)
      img.cur[as.matrix(d2.index)] = img.cur[as.matrix(d2.index)] - s*0.3*vis - s*0.1/(g$d2[d2.fil]+1)
      #generate noise area
      if(gen.noise){
        d3.fil = g$d3 <= min(vis, 5)
        d3.index = g[d3.fil, c('Var1', 'Var2', 'Var3')]
        img.cur[as.matrix(d3.index)] = img.cur[as.matrix(d3.index)] - 0.3*vis - 0.1/(g$d3[d3.fil]+1)
      }
      img.cur = pmax(img.cur, 0.1)
      img = as.nifti(img.cur)
      file.name = paste0(fold.name, '/', num.vis - i + 1)
      writeNIfTI(img.cur, file.name)
    }
    df = data.frame(id = str_pad(id, 3, pad = '0'), time = times[vis]+12, ad = 0)
    new.df = rbind(new.df, df)
  }
  return(new.df)
}


#save the image information for CNN
save.info = function(img.seq, outcome, folder, set = 'train'){

  roi.path = function(folder, set){
    #get the file_names (PIDs)
    cur_path = getwd()
    #should change the folder name according to your need
    folder_name = paste0('/Data/', folder, '/', set)
    image_path = paste0(cur_path, folder_name)
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
  
  image_path = roi.path(folder, set)
  

  #list all the visit by subject
  id.vis = list()
  for(i in outcome$id){
    id.vis[[i]] = img.seq%>%arrange(id, time)%>%filter(id == i)%>%select(id, ad, time)
  }
  
  #save the names by id
  save.by.rid = function(image_names, folder, set){
    
    for(i in names(id.vis)){
      images = image_names[[i]]
      visits = id.vis[[i]]
      print(paste('Code/Info/', folder, '/', set, '/', i, '.csv', sep = ''))
      #should change saving path
      write.csv(data.frame(images = images, visits), 
                paste('Code/Info/', folder, '/', set, '/', i, '.csv', sep = ''))
    }
    
  }
  save.by.rid(image_path, folder, set)
  write.csv(outcome, paste0('Code/Info/', folder,'/', set, '_outcome.csv'))
}

pack.all = function(folder, noise, set = 'train', p, shrink, shrink2 = 1){
  
  nc.df1 = create.nc(1, 100, folder, set, noise, p)
  ncad.df1 = create.ncad(101, 200, folder, set, noise, p, shrink, shrink2)
  ad.df1 = create.ad(201, 300, folder, set, noise, p, shrink, shrink2)
  adnc.df1 = create.adnc(301, 400, folder, set, noise, p, shrink, shrink2)
  
  info1 = rbind(nc.df1, ncad.df1, ad.df1, adnc.df1)
  info1$id = as.character(info1$id)
  outcome1 = info1%>%group_by(id)%>%filter(time == max(time))%>%ungroup(id)
  img.seq1 = info1%>%group_by(id)%>%filter(time < max(time))%>%ungroup(id)
  
  save.info(img.seq1, outcome1, folder, set)
  
}

#function for building graph information
build.graph = function(sim.run, build = 'train'){

  cur.path = getwd()
  folder = paste0(cur.path, '/Code/Info/', sim.run)
  outcome = read.csv(paste0(folder, '/', build, '_outcome.csv'))
  folder = paste0(folder, '/', build)
  rids = dir(folder)
  
  #initiate dataframe for edges and graphs
  all.edges = data.frame()
  all.graphs = data.frame()
  for (rid in rids){
    file = read.csv(paste0(folder, '/', rid))
    #create edge information
    ##combination of row number
    num_nodes = nrow(file)
    src.tar = combn(1:num_nodes, 2)
    times = file$time
    ##weights equal to difference of time
    weights = 1/(times[src.tar[2,]] - times[src.tar[1,]])
    graph_id = substr(rid, 1, 3)
    edges = data.frame(graph_id = graph_id, src = src.tar[1,],
                       dst = src.tar[2,], weight = weights)
    
    #create graph information
    item = filter(outcome, id == as.numeric(graph_id))
    graphs = data.frame(graph_id = graph_id, label = item$ad, num_nodes = num_nodes)
    
    #combine information
    all.graphs = rbind(all.graphs, graphs)
    all.edges = rbind(all.edges, edges)
  }
  
  save.folder = paste0(cur.path, '/Code/Info/SimGraph/', sim.run, '/', build)
  write.csv(all.graphs, paste0(save.folder, '/graphs.csv'), quote = FALSE)
  write.csv(all.edges, paste0(save.folder, '/edges.csv'), quote = FALSE)
  
}

set.seed(100)
#simulation 2
dir.create('Data/graphSim2')
dir.create('Code/Info/graphSim2')
dir.create('Code/Info/SimGraph/graphSim2')
for(i in 1:100){
  num.sim = paste0('graphSim2/', 'sim', i)
  
  #create folder for data
  dir.create(paste0('Data/', num.sim))
  dir.create(paste0('Data/', num.sim, '/train'))
  dir.create(paste0('Data/', num.sim, '/test'))
  #create folder for image info
  dir.create(paste0('Code/Info/', num.sim))
  dir.create(paste0('Code/Info/', num.sim, '/train'))
  dir.create(paste0('Code/Info/', num.sim, '/test'))
  #create folder for graph info
  dir.create(paste0('Code/Info/SimGraph/', num.sim))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/train'))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/test'))
  
  pack.all(num.sim, noise = 0.2, 'train', p = 0.5, 0.05, 0.05)
  pack.all(num.sim, noise = 0.2, 'test', p = 0.5, 0.05, 0.05)
  build.graph(num.sim, 'train')
  build.graph(num.sim, 'test')
  
}


#simulation 3
dir.create('Data/graphSim3')
dir.create('Code/Info/graphSim3')
dir.create('Code/Info/SimGraph/graphSim3')
for(i in 1:100){
  num.sim = paste0('graphSim3/', 'sim', i)
  
  #create folder for data
  dir.create(paste0('Data/', num.sim))
  dir.create(paste0('Data/', num.sim, '/train'))
  dir.create(paste0('Data/', num.sim, '/test'))
  #create folder for image info
  dir.create(paste0('Code/Info/', num.sim))
  dir.create(paste0('Code/Info/', num.sim, '/train'))
  dir.create(paste0('Code/Info/', num.sim, '/test'))
  #create folder for graph info
  dir.create(paste0('Code/Info/SimGraph/', num.sim))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/train'))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/test'))
  
  pack.all(num.sim, noise = 0.2, 'train', p = 0.5, 0.03, 0.03)
  pack.all(num.sim, noise = 0.2, 'test', p = 0.5, 0.03, 0.03)
  build.graph(num.sim, 'train')
  build.graph(num.sim, 'test')
  
}


#simulation 4
dir.create('Data/graphSim4')
dir.create('Code/Info/graphSim4')
dir.create('Code/Info/SimGraph/graphSim4')
for(i in 1:100){
  num.sim = paste0('graphSim4/', 'sim', i)
  
  #create folder for data
  dir.create(paste0('Data/', num.sim))
  dir.create(paste0('Data/', num.sim, '/train'))
  dir.create(paste0('Data/', num.sim, '/test'))
  #create folder for image info
  dir.create(paste0('Code/Info/', num.sim))
  dir.create(paste0('Code/Info/', num.sim, '/train'))
  dir.create(paste0('Code/Info/', num.sim, '/test'))
  #create folder for graph info
  dir.create(paste0('Code/Info/SimGraph/', num.sim))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/train'))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/test'))
  
  pack.all(num.sim, noise = 0.2, 'train', p = 0.5, 0.02, 0.02)
  pack.all(num.sim, noise = 0.2, 'test', p = 0.5, 0.02, 0.02)
  build.graph(num.sim, 'train')
  build.graph(num.sim, 'test')
  
}




#simulation 7
dir.create('Data/graphSim7')
dir.create('Code/Info/graphSim7')
dir.create('Code/Info/SimGraph/graphSim7')
for(i in 1:100){
  num.sim = paste0('graphSim7/', 'sim', i)
  
  #create folder for data
  dir.create(paste0('Data/', num.sim))
  dir.create(paste0('Data/', num.sim, '/train'))
  dir.create(paste0('Data/', num.sim, '/test'))
  #create folder for image info
  dir.create(paste0('Code/Info/', num.sim))
  dir.create(paste0('Code/Info/', num.sim, '/train'))
  dir.create(paste0('Code/Info/', num.sim, '/test'))
  #create folder for graph info
  dir.create(paste0('Code/Info/SimGraph/', num.sim))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/train'))
  dir.create(paste0('Code/Info/SimGraph/', num.sim, '/test'))
  
  pack.all(num.sim, noise = 0.2, 'train', p = 0.5, 0.1, 0.1)
  pack.all(num.sim, noise = 0.2, 'test', p = 0.5, 0.1, 0.1)
  build.graph(num.sim, 'train')
  build.graph(num.sim, 'test')
  
}

