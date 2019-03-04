# PlanarMonocularSLAM
Clone the repository and add the unzipped
[dataset](https://drive.google.com/open?id=1t0GJB9xKKzIJ7ibvVF2MYithinhRRJku)
in the directory of the
repository itself (after this step you should have all the data in
`PlanarMonocularSLAM/dataset`). Compile and run the program with:
~~~~
make
./bin/main
~~~~
Note that the current `Makefile` is using as path for Eigen
`/usr/local/include/eigen3`, change it accordingly to the right path on
your system.

To generate the plots of the final result, run the script:
~~~~
matlab/plot_trajectories.m
~~~~
A slightly more detailed description of the project is available [here](https://github.com/micco00x/ProbabilisticRoboticsReport/blob/master/report.pdf).
