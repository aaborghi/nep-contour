# Contour integral eigensolvers for nonlinear eigenvalue problems
This code is a simple implementation of contour integral eigensolvers from the following publications
- >M. Brennan, M. Embree, S. Gugercin, **Contour Integral Methods for Nonlinear >Eigenvalue Problems: A Systems Theoretic Approach**, _SIAM Review_, (2023), https://>epubs.siam.org/doi/10.1137/20M1389303
- >W.J. Beyn, **An integral method for solving nonlinear eigenvalue problems**, >_Linear Algebra and Applications_, (436)10, 3839-3863, (2012), https://doi.org/10.>1016/j.laa.2011.03.030 

For benchmarking the algorithms we adopted a 50x50 delay eigenvalue problem (see bigdae.py) and an example from the following publication (see testhospital.py)
- >T. Betcke, N.J. Higham, V. Mehrmann, C. Schroeder, F. Tisseur, **NLEVP: A >Collection of Nonlinear Eigenvalue Problems**,_ACM Transactions in Mathematical >Software_, (39)2, 1-28, (2013), https://dl.acm.org/doi/10.1145/2427023.2427024

## Installation and running
Simply clone this repository with 

`git@github.com:aaborghi/nep-contour.git`

then run test.py to see if everything works as expected. A plot should pop-out showing the resulting eigenvalues matching the real ones. 
For the ones that want to check the contour integral algorithms implementations look under `src/nlevpfunc.py`.