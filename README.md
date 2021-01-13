# project_in_ai
A repository containing source code for the course "Project in AI", offered on the Msc programme of engineering in Robotics at SDU.

If you want to try out the code, follow the below steps to get started.


## Content

The hand-in-folder contains 3 folders and 4 files. The 3 folders are:

	/bertles 		The bertles language model, trained on
				the dataset beatles.txt.
				
	/experiments 		All results from the experiments, 
				described in the report.
				
	/melodies		All the melody fragments generated
				randomly using shorturl.at/svwFP.

The 4 files are:
    
   	ai-env.yml		The conda environment containing all
    				dependencies necessary to run the
    				source code.
    	
	beatles.txt		A dataset containing lyrics from all 
    				online available song lyrics by The 
    				Beatles, with removed repititions
    				and non-English words
    	
	harmony.py		The source code for harmony 
    				generation.
    	
	rhyme.py		The source code for song lyric 
    				generation
    

## Set Up


All dependencies are installed in the conda-environment attached in the 
folder. To use the environment, you need to install Anaconda on your PC:
https://docs.anaconda.com/anaconda/install/. 

Then, in the terminal go to the location of the hand-in-folder and type:

	$ 	conda env create -f ai-env.yml 
 
Then, activate the environment:

 	$	conda activate ai-env 
 

## RUN THE CODE 
 
To run the source code for harmony generation, type:

 	$ 	python3 harmony.py  

 Likewise, to run the source code for lyric generation, type:
 
 	$ 	python3 rhyme.py 




 
 

