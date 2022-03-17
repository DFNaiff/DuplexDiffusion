#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <cmath>

#include "json.hpp"

#include "solver.h"
#include "timer.h"
#include "controller.h"

//Code snippet taken from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf,
//in the answer given by iFreilicht
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

//Main solver
void fixed_timestep_solve(duplexsolver::PhysicalParams& physparams,
                          duplexsolver::SolverParams& solparams,
                          duplexsolver::BoundaryConditions& bcconditions,
                          double tmax,
                          std::string savename) {
    duplexsolver::Solver solver(physparams, solparams, bcconditions);
    std::ofstream file_bulk;
    std::ofstream file_precip;
    std::vector<double> xspace = solver.get_xspace();
    file_bulk.open(savename + "_b");
    file_precip.open(savename + "_p");
    //Write space line
    std::vector<std::ofstream*> files{&file_bulk, &file_precip};
    for(std::ofstream* file : files){
        *file << 0.0 << " "; //Dummy
        for(int i = 0; i < solparams.nspace; i++){
            double x = xspace[i];
            if(i == solparams.nspace-1){
                *file << x << std::endl;
            } else {
                *file << x << " ";
            }
        }
    }
    //Solve and write
    double percentage = 0.0;
    int precision = 1;
    double precisionmultiplier = std::pow(10, precision);
    while(solver.last_timestep() < tmax){
        //solver.step();
        auto bulk_val = solver.step();
        auto precip_val = solver.get_precipitate_concentration();
        file_bulk << solver.last_timestep() << " " << bulk_val.transpose() << std::endl;
        file_precip << solver.last_timestep() << " " << precip_val.transpose() << std::endl;

        double currper = std::floor(100*precisionmultiplier*solver.last_timestep()/tmax)/precisionmultiplier;
        if(currper >= percentage){
            percentage = currper;
            std::cout << "Percentage of calculation done:" << percentage << "%\r";
        }
    }
    file_bulk.close();
    file_precip.close();
}

//Due to integration accuracy issues
//support for variable timestep is limited for now, and API won't be updated
void variable_timestep_solve(duplexsolver::PhysicalParams& physparams,
                             duplexsolver::SolverParams& solparams,
                             duplexsolver::BoundaryConditions& bcconditions,
                             control::Controller& controller,
                             double tmax,
                             std::string savename) {
    duplexsolver::Solver solver(physparams, solparams, bcconditions);
    std::ofstream file;
    file.open(savename);
    std::vector<double> xspace = solver.get_xspace();
    file.open(savename);
    //Write space line
    file << 0.0 << " "; //Dummy
    for(int i = 0; i < solparams.nspace; i++){
        double x = xspace[i];
        if(i == solparams.nspace-1){
            file << x << std::endl;
        } else {
            file << x << " ";
        }
    }
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

    try{ //Has default value
        physparams.bulk_geom = jsondata.at("physical").at("bulk_geometry");
    } catch(nlohmann::detail::out_of_range e){}
    try{ //Has default value
        physparams.precip_geom = jsondata.at("physical").at("precipitate_geometry");
    } catch(nlohmann::detail::out_of_range e){}
    physparams.D = jsondata.at("physical").at("matrixD"); //Required
    physparams.alpha = jsondata.at("physical").at("precipitateD"); //Required
    physparams.K = jsondata.at("physical").at("cratio"); //Required
    if(physparams.precip_geom == 0){
        physparams.vol_fraction = jsondata.at("physical").at("vol_fraction");
    } else if (physparams.precip_geom == 1){
        physparams.area_fraction = jsondata.at("physical").at("area_fraction");
    }
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
        savestring = "../data/testcli/result_json_toy"; //Default (for debug)
    }
    if(argc >= 4){
        inputstring = argv[3];
    } else {
        inputstring = "../params/params.json";
    }

    std::string instring = "Reading parameters from " + inputstring + "\n" +
                           string_format("Running model until t=%f\n", tfinal);
    std::cout << instring;
    timer::Timer timerobj;
    timerobj.reset();

    json_reader(inputstring, physparams, solparams, bcconditions);
    fixed_timestep_solve(physparams, solparams, bcconditions, tfinal, savestring);

    double elapsed = timerobj.elapsed();
    std::string outstring = string_format("Done! The elapsed time was %f\n", elapsed) +
                            "Result file can be found in " + savestring + "_b" + " (for bulk) \n" +
                            "and in " + savestring + "_p" + "(for precipitate) \n";
    std::cout << outstring;
}
