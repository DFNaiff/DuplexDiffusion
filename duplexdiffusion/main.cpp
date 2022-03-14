#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>

#include "json.hpp"

#include "solver.h"
#include "timer.h"
#include "controller.h"

//Main solver
void fixed_timestep_solve(duplexsolver::PhysicalParams& physparams,
                          duplexsolver::SolverParams& solparams,
                          duplexsolver::BoundaryConditions& bcconditions,
                          double tmax,
                          std::string savename) {
    duplexsolver::Solver solver(physparams, solparams, bcconditions);
    std::ofstream file;
    file.open(savename);
    //Write space line
    file << 0.0 << " "; //Dummy
    for(int i = 0; i < solparams.nspace; i++){
        double x = double(i)/(solparams.nspace-1)*physparams.L;
        if(i == solparams.nspace-1){
            file << x << std::endl;
        } else {
            file << x << " ";
        }
    }
    //Solve and write
    while(solver.last_timestep() < tmax){
        //solver.step();
        auto current_val = solver.step();
        file << solver.last_timestep() << " " << current_val.transpose() << std::endl;
    }
    file.close();
}

void variable_timestep_solve(duplexsolver::PhysicalParams& physparams,
                             duplexsolver::SolverParams& solparams,
                             duplexsolver::BoundaryConditions& bcconditions,
                             control::Controller& controller,
                             double tmax,
                             std::string savename) {
    duplexsolver::Solver solver(physparams, solparams, bcconditions);
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

void json_reader(std::string filename,
                 duplexsolver::PhysicalParams& physparams,
                 duplexsolver::SolverParams& solparams,
                 duplexsolver::BoundaryConditions& bcconditions){
    std::ifstream inputfile(filename);
    std::stringstream inputbuffer;
    std::string inputstring;

    inputbuffer << inputfile.rdbuf();
    inputstring = inputbuffer.str();
    nlohmann::json jsondata = nlohmann::json::parse(inputstring, nullptr, true, true);

    physparams.cylinder = jsondata.at("physical").at("cylinder");
    physparams.D = jsondata.at("physical").at("matrixD");
    physparams.alpha = jsondata.at("physical").at("precipitateD");
    physparams.K = jsondata.at("physical").at("cratio");
    physparams.vol_fraction = jsondata.at("physical").at("vol_fraction");
    physparams.area_fraction = jsondata.at("physical").at("area_fraction");
    physparams.L = jsondata.at("physical").at("length");

    bcconditions.left_bc_type = jsondata.at("bcconditions").at("left_bc_type");
    bcconditions.left_bc_value = jsondata.at("bcconditions").at("left_bc_value");
    bcconditions.right_bc_type = jsondata.at("bcconditions").at("right_bc_type");
    bcconditions.right_bc_value = jsondata.at("bcconditions").at("right_bc_value");

    try{
        solparams.nspace = jsondata.at("solver").at("nspace");
    } catch(nlohmann::detail::out_of_range e){}
    try{
        solparams.timestep = jsondata.at("solver").at("dt");
    } catch(nlohmann::detail::out_of_range e){}
    try{
        solparams.kernel_limit = jsondata.at("solver").at("kernel_limit");
    } catch(nlohmann::detail::out_of_range e){}
    try{
        solparams.maxkernel = jsondata.at("solver").at("maxkernel");
    } catch(nlohmann::detail::out_of_range e){}
    try{
        solparams.decay_limit = jsondata.at("solver").at("decay_limit");
    } catch(nlohmann::detail::out_of_range e){}
    try{
        solparams.maxwindow = jsondata.at("solver").at("maxwindow");
    } catch(nlohmann::detail::out_of_range e){}
}


int main(int argc, char **argv)
{

//    //Just leave the defaults for now
    //for(int i = 0; i < argc; i++){
    //    std::cout << argv[i] << std::endl;
    //}
    duplexsolver::PhysicalParams physparams;
    duplexsolver::SolverParams solparams;
    duplexsolver::BoundaryConditions bcconditions;

    double tfinal;
    std::string tfinalstr;
    std::string inputstring;
    std::string savestring;
    if(argc >= 2){
        tfinalstr = argv[1];
        tfinal = std::stod(tfinalstr);
    } else {
        tfinal = 25.0; //Default
    }
    if(argc >= 3){
        savestring = argv[2];
    } else {
        savestring = "../data/testcli/result_json_toy.txt"; //Default (for debug)
    }
    if(argc >= 4){
        inputstring = argv[3];
    } else {
        inputstring = "../data/testcli/params.json";
    }

    json_reader(inputstring, physparams, solparams, bcconditions);
    //std::string savename = "../data/testcli/result_json_toy";
    //double tfinal = 25.0;
    fixed_timestep_solve(physparams, solparams, bcconditions, tfinal, savestring);

//
//    std::string savename = "../data/result_toy";
//    double tfinal = 25.0;
//
//    //control::Controller controller(0.025, 25.00);
//    //variable_timestep_solve(physparams, solparams, controller, tfinal, savename);
//
//    fixed_timestep_solve(physparams, solparams, bcconditions, tfinal, savename);
}
