#include <iostream>
#include <fstream>
#include <chrono>

#include "solver.h"
#include "timer.h"

int main()
{
    duplexsolver::Parameters parameters;
    duplexsolver::SolParams solparams;
    //solparams.nkernel = 10
    parameters.R = 5;
    parameters.vol_fraction = 0.4; //0.1
    parameters.L = 100;
    solparams.maxwindow = 1000;
    duplexsolver::Solver solver(parameters, solparams);
    solver.prepare_linear_system();
    int nsteps = 100000;
//
    timer::Timer t;
    std::ofstream file;
    std::ofstream filestep;
    t.reset();
    file.open("../data/result_realistic");
    for(int i = 0; i < nsteps; i++){
        //solver.step();
        file << solver.step().transpose() << std::endl;
    }
    file.close();
    std::cout << t.elapsed() << std::endl;
    std::cout << solver.get_precision() << std::endl;
    std::cout << solver.get_rhs() << std::endl;
}
