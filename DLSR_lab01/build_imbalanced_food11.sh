#!/bin/bash
source_path='food11re/training' ;
dest_path='food11re/skewed_training' ;
ratio_per_class=(100 30 5 100 60 60 100 30 30 60 100) ;
classes=($(ls $source_path )) ;
for i in ${!classes[@]} ; 
do 
   mkdir -p $dest_path/${classes[i]};
   data_num=$(( $( ls $source_path/${classes[i]} | wc -l ) ));
   echo "$( ls  $source_path/${classes[i]}/* | head -n $(( ${data_num} * $((${ratio_per_class[i]})) / 100 )) )";
   cp $( ls  $source_path/${classes[i]}/* | head -n $(( ${data_num} * $((${ratio_per_class[i]})) / 100 )) ) $dest_path/${classes[i]}/ ;
done

rm -r $source_path ;

