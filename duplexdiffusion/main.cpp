#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "solver.h"
#include "timer.h"

//Main solver
void fixed_timestep_solve(duplexsolver::Parameters& parameters,
                          duplexsolver::SolParams& solparams,
                          double tmax,
                          std::string savename) {
    duplexsolver::Solver solver(parameters, solparams);
    solver.prepare_linear_system();
    std::ofstream file;
    file.open(savename);
    while(solver.last_timestep() < tmax){
        //solver.step();
        auto current_val = solver.step(solparams.timestep);
        file << solver.last_timestep() << " " << current_val.transpose() << std::endl;
    }
    file.close();
}

int main()
{
    //Just leave the defaults for now
    duplexsolver::Parameters parameters;
    duplexsolver::SolParams solparams;
    parameters.cylinder = true;
    parameters.area_fraction = 0.0;
    solparams.maxwindow = 2;
    std::string savename = "../data/result_clean_test";
    double tfinal = 25000.0;
    fixed_timestep_solve(parameters, solparams, tfinal, savename);
}
