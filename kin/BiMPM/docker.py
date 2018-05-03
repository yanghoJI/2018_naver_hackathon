#nsml: floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.24
#nsml: tensorflow/tensorflow:1.7.0-gpu-py3	cuda error
#nsml: tensorflow/tensorflow:latest-gpu-py3	cuda error
#nsml: chaneyk/tensorflow-cuda9:latest-gpu-py3 notworking
#nsml: chaneyk/tensorflow-cuda9:latest-gpu 	python2 working
#nsml: mediadesignpractices/tensorflow:latest
#nsml: wtan/tf-1.7.0-cuda9:0.0.1	notworking
#nsml: wtan/tf-1.7.0-cuda9-ubuntu:0.0.1	python2 working
#nsml: ioxe/tensorflow-devel-gpu-cuda9-cudnn7:latest 		python2 working
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda9cudnn7-py3_aws.22 	not working
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda8cudnn6-py3_aws.22 	working!!!!!

1.5.1-gpu-py3

model 
nsml submit kimsunmok/kin_phase1/68 8   -04050219


latest-gpu-py3

model info
74 : bimpm	
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda8cudnn6-py3_aws.22  
cudnn LSTM  
tr:True	submit:False 

77 : bimpm	
nsml docker 
no cudnn LSTM  
tr:True	submit:False 
..............Error: Session does not respond
2018/04/05 19:05:42 nsml: Internal server error

79 : kin 	
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda8cudnn6-py3_aws.22 
tr:True	 	submit:True (longtime)

83 : bimpm
code arrange version (SentenceMatchTrainer_nsml_v2.py )	
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda8cudnn6-py3_aws.22  
cudnn LSTM  
tr:False 	submit:False

86 : bimpm
code arrange version (SentenceMatchTrainer_nsml_v2.py )	
#nsml: floydhub/tensorflow:1.5.0-gpu.cuda8cudnn6-py3_aws.22  
cudnn LSTM  
modipy infer input  dic > list
tr:True 	submit:False
..............Error: Session does not respond
2018/04/05 21:05:07 nsml: Internal server error


