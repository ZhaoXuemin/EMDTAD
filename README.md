### EMTAD: an empirical mode decomposition-based method to identify topologically associated domains from chromatin interactions

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

EMTAD introduces the Empirical Modal Decomposition (EMD) algorithm for mode decomposition and feature extraction in the Hi-C interaction matrix, which enables fast and accurate identification of Topological Associated Structural Domains (TADs) in the Hi-C interaction matrix.



Installation instructions below were tested in Linux Ubuntu、MacOS. 


1. Dependent installation packages:

   Python 3.6+
   
   Numpy 1.17.3
   
   EMD-signal 1.2.3
   
   scikit-image 0.17.2
   
   scikit-learn 0.24.2

2. get code from github

   git clone https://github.com/ZhaoXuemin/EMDTAD.git
   
3. Parameters
   1、Dense Hi-C interaction matrix File Path
   2、Path to the output file
   3、Intrinsic Mode Function layers
   4、Window size of Empirical Modal Decomposition
   
4. Run EMDTAD.py

   python EMDTAD.py [input_Matrix_path] [Reuslt_file_path] [imf_deep] [diagonal_deep]

   eg
   ```
   cd EMDTAD

   python EMDTAD.py ./Example/mat.txt ./Example/result.txt 3 30
   ```
5. The result is saved as a column of data where each row is the boundary position of a TAD 
