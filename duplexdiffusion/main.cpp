#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "solver.h"
#include "timer.h"
#include "controller.h"

//Main solver
void fixed_timestep_solve(duplexsolver::Parameters& parameters,
                          duplexsolver::SolParams& solparams,
                          double tmax,
                          std::string savename) {
    duplexsolver::Solver solver(parameters, solparams);
    std::ofstream file;
    file.open(savename);
    while(solver.last_timestep() < tmax){
        //solver.step();
        auto current_val = solver.step();
        file << solver.last_timestep() << " " << current_val.transpose() << std::endl;
    }
    file.close();
}

void variable_timestep_solve(duplexsolver::Parameters& parameters,
                             duplexsolver::SolParams& solparams,
                             control::Controller& controller,
                             double tmax,
                             std::string savename) {
    duplexsolver::Solver solver(parameters, solparams);
    std::ofstream file;
    file.open(savename);
    while(solver.last_timestep() < tmax){
        //solver.step();
        double dt = controller.suggest_step();
        auto current_val = solver.step(dt);
        double error = solver.get_last_step_error();
        bool accepted = controller.evaluate_step(error);
        if(accepted){
            file << solver.last_timestep() << " " << current_val.transpose() << std::endl;
        } else {
            solver.cancel_last_step();
        }
    }
    file.close();
}

int main()
{
    //Just leave the defaults for now
    duplexsolver::Parameters parameters;
    duplexsolver::SolParams solparams;
    parameters.cylinder = true;
    parameters.vol_fraction = 0.0;
    solparams.maxwindow = 2;
    //
    std::string savename = "../data/result_large_convolution_clean";
    double tfinal = 25000.0;

    //control::Controller controller(0.025, 25.00);
    //variable_timestep_solve(parameters, solparams, controller, tfinal, savename);

    fixed_timestep_solve(parameters, solparams, tfinal, savename);
}
