	INSTALLATION AND RUN OF ProResCONs
	(Copyright 2020 by Xiaofei-Wang, Zhejiang University of Technology, All rights reserved)

1.What is ProResCONs?
	ProResCONs is a protein residue contact prediction method by improving deep convolutional neural network and loss function.
	
2.How to configure the environment?

	Anaconda is recommended to set up Pytorch environments
	Python3.7 with numpy , scipy and numba should be installed
	Pytorch-1.5.1 (https://pytorch.org/get-started/previous-versions/) 
	cuda = 10.2
	
3.How to run ProResCONs?
	1) Input file: The Multiple Sequence Alignment needs to be prepared. As shown in the file given in the example folder,and suffix of the file is a3m.
	
	2) RUN ProResCONs_run python file:
	First, give the path of the file to be tested and the path to save the output file, i.e. input_file(Just give the folder location of the file) 
	and ProResCONs_outfile(The folder location where you want to save the file, because the corresponding prediction file is generated based on the input file) in ProResCONs_run final.	
	Then, activate pytorch environments,find the location of the ProResCONs_run file.
	End, python ProResCONs_run
	
	3) Output_file: after running the ProResCONs_run program, you will get a file.The first and second columns of the file represent the positions of residues, 
	and the third column is the probability value of the contact between residues predicted by the model
	
4.Example
	You can also use the Multiple Sequence Alignment file(e.g., 1a7wA.a3m) in the example folder to test it,and get output file(e.g., 1a7wA).
	
To run this model, you need to prepare input files(i.e., a3m file). Of course, you can change the reading in WMSAdatatool program file to your desired input file format.