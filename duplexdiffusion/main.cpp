#include <iostream>
#include <fstream>
#include <chrono>

#include "solver.h"
#include "timer.h"

//Main solver
int main()
{
    duplexsolver::Parameters parameters;
    duplexsolver::SolParams solparams;
    //solparams.nkernel = 10
    parameters.R = 5;
    parameters.vol_fraction = 0.4; //0.4
    parameters.L = 100;
    solparams.maxwindow = 5000;//1000
    solparams.timestep /= 5;
    int nsteps = 100000;
    nsteps *= 5;
    duplexsolver::Solver solver(parameters, solparams);
    solver.prepare_linear_system();
    timer::Timer t;
    std::ofstream file;
    std::ofstream filestep;
    t.reset();
    file.open("../data/result_realistic_detailed");
    for(int i = 0; i < nsteps; i++){
        //solver.step();
        file << solver.step().transpose() << std::endl;
    }
    file.close();
    std::cout << t.elapsed() << std::endl;
    std::cout << solver.get_precision() << std::endl;
    std::cout << solver.get_rhs() << std::endl;
}
