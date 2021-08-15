MANDERA= function(messageMatrix,row.index.mal){
  # this function returns the set whose in-group variance is minimum
  # messageMatrix is the gradient matrix 100 x p
  # row.index.mal is a nx1 matrix, where the first column is  the row index of malicious nodes
  # the output is a matrix: 
  mydat.rank=apply(messageMatrix,2,rank)
  var.rank=apply(mydat.rank,1,var)
  calg=kmeans(var.rank,centers = 2)
  group.id=which(tabulate(calg$cluster)<=nrow(mydat.rank)/2)[1]
  poisoned.est = which(calg$cluster==group.id)
  detected=length(intersect(poisoned.est,as.matrix(poisoned)))
  
  total=length(poisoned.est)
  TP= detected
  FP=total-detected
  FN=nrow(row.index.mal)-TP
  TN= nrow(messageMatrix) -length(row.index.mal) -FP
  Precision = TP/(TP+FP)
  Recall= TP/(TP+FN)
  output = data.frame(Accuracy =(TP+TN)/(TP+TN+FP+FN),Precision =TP/(TP+FP),Recall= TP/(TP+FN),F1 =2*Precision*Recall/(Precision+Recall))
  return(output)
}
