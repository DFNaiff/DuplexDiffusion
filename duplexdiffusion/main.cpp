#include <iostream>
#include <fstream>
#include <chrono>

#include "solver.h"

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using clock_type = std::chrono::steady_clock;
	using second_type = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<clock_type> m_beg { clock_type::now() };

public:
	void reset()
	{
		m_beg = clock_type::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<second_type>(clock_type::now() - m_beg).count();
	}
};


int main()
{
    duplexsolver::Parameters parameters;
    duplexsolver::SolParams solparams;
    //solparams.nkernel = 10
    parameters.vol_fraction = 0.1; //0.1
    solparams.maxwindow = 1000;
    duplexsolver::Solver solver(parameters, solparams);
    solver.prepare_linear_system();
    int nsteps = 100;
//
    Timer t;
    std::ofstream file;
    std::ofstream filestep;
    t.reset();
    file.open("../notebooks/result_little");
    filestep.open("../notebooks/result_little_t");
    for(int i = 0; i < nsteps; i++){
        //solver.step();
        file << solver.step().transpose() << std::endl;
        filestep << solparams.timestep << "\n" << std::endl;
    }
    file.close();
    filestep.close();
    std::cout << t.elapsed() << std::endl;
    std::cout << solver.get_precision() << std::endl;
    std::cout << solver.get_rhs() << std::endl;
}
