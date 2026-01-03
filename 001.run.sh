
filePar="./001.par"
code="./001.forward.cpp"
proc="./001.out"

rm -f $proc
c++ -O3 $code -o $proc -lpthread -fopenmp -lblas -llapack
#Run and input Par:
time $proc $filePar 10