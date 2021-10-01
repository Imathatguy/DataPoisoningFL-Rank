# this file compares training performance of 5 types of defence.
defence=c("krum","mandera","bulyan","median","trmean","no_attack") #,"no_defense"
dat.set=c("FASHION","CIFAR")
num.mal=c("05", "10","15","20","25","30")
att=c("GA","ZG","SF","LF")

tbl_colnames=c("Defence", "Attack","Mal.node","Rep","Accuracy","Loss","Epoch","Data")
results <- as_tibble(data.frame(matrix(nrow=0,ncol=length(tbl_colnames)))) # for performance results
colnames(results) <- tbl_colnames

tbl_colnames=c("Defence","Attack","Mal.node","Rep","Time","Epoch","Data")
time.tab <- as_tibble(data.frame(matrix(nrow=0,ncol=length(tbl_colnames)))) # for performance results
colnames(time.tab) <- tbl_colnames

for(def.type in defence){
  for (dat.type in dat.set){
    for(attack.type in att){
      if (attack.type=="LF" & dat.type=="FASHION"){
        att.id=2
      }
      if (attack.type=="GA" & dat.type=="FASHION"){
        att.id=3
      }
      if (attack.type=="ZG"& dat.type=="FASHION"){
        att.id=4
      }
      if (attack.type=="SF"& dat.type=="FASHION"){
        att.id=5
      }
      
      if (attack.type=="LF" & dat.type=="CIFAR"){
        att.id=6
      }
      if (attack.type=="GA" & dat.type=="CIFAR"){
        att.id=7
      }
      if (attack.type=="ZG"& dat.type=="CIFAR"){
        att.id=8
      }
      if (attack.type=="SF"& dat.type=="CIFAR"){
        att.id=9
      }
      if(def.type=="mandera"){def.id=1}
      if(def.type=="krum"){def.id=2}
      if(def.type=="bulyan"){def.id=3}
      if(def.type=="median"){def.id=4}
      if(def.type=="trmean"){def.id=5}
      if(def.type=="no_attack" & dat.type=="FASHION"){def.id=2}
      if(def.type=="no_attack" & dat.type=="CIFAR"){def.id=6}
      if (def.type=="no_attack"){
        for(rep in 0:9){
          dat = read.csv(file=paste0("../results/",def.type,"_results/",def.id,"000",rep,"/",def.id,"000",rep, "_results.csv"),header=FALSE)
          for(num in num.mal){
            rep.result=tibble("Defence"=def.type,  "Attack"=attack.type,"Mal.node"=as.numeric(num),"Rep"=as.numeric(paste0(rep)),"Epoch"=1:25,"Accuracy"=dat[,1],"Loss"=dat[,2],"Data"=dat.type)
            results=rbind(results,rep.result)
          }
        }
      } else if(def.type=="no_defense"){
        for(num in num.mal){
          for(rep in 0:9){
            dat = read.csv(file=paste0("../results/",def.type,"_results/",att.id,num,"0",rep,"/",att.id,num,"0",rep, "_results.csv"),header=FALSE)
            rep.result=tibble("Defence"=def.type,  "Attack"=attack.type,"Mal.node"=as.numeric(num),"Rep"=as.numeric(paste0(rep)),"Epoch"=1:25,"Accuracy"=dat[,1],"Loss"=dat[,2],"Data"=dat.type)
            results=rbind(results,rep.result)
            
          }
        }
      }
      else{
        for(num in num.mal){
          for(rep in 0:9){
            dat = read.csv(file=paste0("../results/",def.type,"_results/",def.id,att.id,num,"0",rep,"/",def.id,att.id,num,"0",rep, "_results.csv"),header=FALSE)
            rep.result=tibble("Defence"=def.type,  "Attack"=attack.type,"Mal.node"=as.numeric(num),"Rep"=as.numeric(paste0(rep)),"Epoch"=1:25,"Accuracy"=dat[,1],"Loss"=dat[,2],"Data"=dat.type)
            results=rbind(results,rep.result)
            
            #dat.time = read.csv(file=paste0("../results/",def.type,"_results/",def.id,att.id,num,"0",rep,"/",def.id,att.id,num,"0",rep, "_results_timing.csv"),header=FALSE)
            #rep.time=tibble("Defence"=def.type,  "Attack"=attack.type,"Mal.node"=as.numeric(num),"Rep"=as.numeric(paste0(rep)),"Time"=(dat.time[2,1]-dat.time[1,1])/25,"Data"=dat.type)
            #time.tab=rbind(time.tab,rep.time)
          }
        }
      }
      
    }
  }
}

save.image("../results/training.performacen.RData")


# plotting
source('plot.prior.setting.R') # load all the settings for plotting



results=results %>% mutate(.,Defence= if_else(Defence=="mandera","MANDERA",Defence)) %>% mutate(.,Defence= if_else(Defence=="krum","Krum",Defence))%>% mutate(.,Defence= if_else(Defence=="bulyan","Bulyan",Defence)) %>% mutate(.,Defence= if_else(Defence=="median","Median",Defence)) %>% mutate(.,Defence= if_else(Defence=="trmean","Trim-mean",Defence)) %>% mutate(.,Defence= if_else(Defence=="no_attack","NO-attack",Defence))
for(dat.type in dat.set){
  a=results%>% filter(.,Data==dat.type) %>% group_by(Attack,Defence,Epoch,Mal.node)%>% summarise(Accuracy= mean(Accuracy),Loss=mean(Loss)) %>% mutate(.,Attack=factor(Attack,levels = att.order)) %>% mutate(.,Defence=factor(Defence,levels = def.order))
  
  p=ggplot(a,aes(x=Epoch,y=Accuracy))
  p=p + geom_line(aes(linetype=Defence,colour = Defence),lwd=0.5)+labs(x="Number of Epoch") + facet_grid(vars(Attack),vars(Mal.node),scale="free_y") + scale_linetype_manual(breaks=def.order, values=c(6:1)) +scale_colour_manual(breaks=def.order, values=c("grey","blue",4,3,7, "red"))
  if(dat.type=="CIFAR"){
    p+ theme(legend.position="top")
  } else{
    p+ theme(legend.position="top")
  }
  
  ggsave(filename = paste0("../results/",dat.type, ".", "Accuracy" ,".pdf"),height = 5,width = 7.5)
  p=ggplot(a,aes(x=Epoch,y=log(Loss)))
  p=p + geom_line(aes(colour = Defence,linetype=Defence),lwd=0.5)+ scale_linetype_manual(breaks=def.order, values=c(6:1))+labs(x="Number of Epoch") + facet_grid(vars(Attack),vars(Mal.node),scale="free_y")  +scale_colour_manual(breaks=def.order, values=c("grey","blue",4,3,7, "red"))
  if(dat.type=="CIFAR"){
    p+ theme(legend.position="top")
  } else{
    p+ theme(legend.position="top")
  }
  ggsave(filename = paste0("../results/",dat.type, ".", "Loss" ,".pdf"),height = 5,width = 7.5)
  
}


