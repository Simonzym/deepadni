library(plot.matrix)
library(oro.nifti)
library(dplyr)
library(stringr)
library(lattice)
setwd('Y:/PhD/Fall 2020/RA')

d = 128
num_p = 50
is = 0:(d-1)
pe = matrix(NA, nrow = num_p, ncol = 128)

for(i in 1:num_p){
  
  div = exp(is[c(TRUE, FALSE)] * -log(10000)/d)
  pe[i, c(TRUE, FALSE)] = sin((i-1) * div)
  pe[i, c(FALSE, TRUE)] = cos((i-1) * div)
}
dis.m = matrix(NA, nrow = num_p, ncol = num_p)
gnn.w = matrix(NA, nrow = num_p, ncol = num_p)
for(p in 1:num_p){
  for(q in 1:num_p){
    dis.m[p, q] = t(pe[p,])%*%pe[q,]
    gnn.w[p, q] = 1/(abs(p-q)+1)
  }
}
tmp = c()
for(i in 1:40){
  tmp = c(tmp, dis.m[i, i+10])
}
plot(tmp, type = 'point')
levelplot(dis.m, border = NA, xlab = 'Time', ylab = 'Time',
     main = 'Dot Product of Positional Embedding')
levelplot(gnn.w, border = NA, xlab = 'Time', ylab = 'Time',
          main = 'Weights between Nodes')

plot(1:30, dis.m[1,])


