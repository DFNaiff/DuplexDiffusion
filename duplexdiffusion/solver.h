#ifndef DUPLEX_SOLVER_H
#define DUPLEX_SOLVER_H

#include <cmath>
#include <vector>
#include <exception>
#include <tuple>
#include <deque>
#include <iostream>
#include <string>
#include <exception>

#include <boost/math/special_functions/bessel.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


namespace duplexsolver
{

/**The physical parameters of the system */
struct PhysicalParams{
    int bulk_geom = 0; //0 for plate, 1 for cylinder, 2 for sphere
    int precip_geom = 0; //0 for spherical, 1 for cylindrical
    double D = 60.0; // um^2/s
    double alpha = 1.4*1e-4; // um^2/s
    double R = 5.0; // um
    double K = 32.51/0.033; //unitless
    double vol_fraction = 0.4; //unitless
    double area_fraction = 0.4; //unitless
    double L = 100; // #um
    double xinit = 0.0; // #um
    double Gamma(){
        if(precip_geom == 0){
            return vol_fraction/(4.0/3*M_PI*R*R*R);} //1/um^3
        else if (precip_geom == 1){
            return area_fraction/(4.0*M_PI*R*R);
        } else {
            throw std::domain_error("Not a valid precipitate geometry");
        }//1/um^2
    }
    double beta(){
        if(precip_geom == 0){
            return 8*M_PI*R*Gamma()*alpha*K;
        } else if (precip_geom == 1){
            return 4*M_PI*Gamma()*alpha*K;
        } else {
            throw std::domain_error("Not a valid precipitate geometry");
        }
    }
};

struct BoundaryConditions{
    int left_bc_type = 1; //1 for Dirichlet, 2 for Neunmann
    float left_bc_value = 1.0;
    int right_bc_type = 1;
    float right_bc_value = 0.0;
};

/**The solver parameters */
struct SolverParams{
    int nspace = 21;
    double kernel_limit = 0.01;
    int maxkernel = 1000;
    double timestep = 0.25;
    double decay_limit = 0.01;
    int maxwindow = 1000;
};

/**
 * Main solver class
 */
class Solver{
    public:
        /**
         * Initializer.
         * @param parameters - The physical physparams of the problem
         * @param solparams - The solver parameters
         */
        Solver(PhysicalParams physparams, SolverParams solparams, BoundaryConditions bcconditions);
        /**Getter for the memory of cvalues */
        const auto& get_memory() {return m_memory;}
        /**Getter for the precision matrix */
        const auto& get_timesteps() {return m_timesteps;}
        /**Get last time step */
        double last_timestep() {return m_timesteps.back();}
        /**Get the x-linspace */
        std::vector<double> get_xspace();
        /**Makes a fixed time step*/
        Eigen::VectorXd& step();
        /**Makes a variable time step*/
        Eigen::VectorXd& step(double dt);
        /**Cancels the last step*/
        void cancel_last_step();
        /**Get last step error*/
        double get_last_step_error();
    private:
        PhysicalParams m_physparams;
        SolverParams m_solparams;
        BoundaryConditions m_bcconditions;
        std::vector<double> m_omegavals;
        Eigen::VectorXd m_rhs;
        std::deque<Eigen::VectorXd> m_memory;
        Eigen::SparseMatrix<double> m_sparse_precision;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> m_sparse_decomposition;
        bool m_sparse_initialized;
        std::deque<double> m_timesteps;
        /**Get the precomputed kernel values. @returns these values */
        std::vector<double>& omegavalues();
        /**Make the finite difference matrix. @returns this matrix */
        Eigen::SparseMatrix<double>& make_finite_difference_matrix(Eigen::SparseMatrix<double>& matrix, bool initialized=false);
        /**Adds the time terms to the diagonal of the finite difference matrix. @returns this matrix */
        Eigen::SparseMatrix<double>& add_step_to_diag(Eigen::SparseMatrix<double>& matrix);
        /**Adds the time terms to the diagonal of the finite difference matrix. @returns this matrix */
        Eigen::SparseMatrix<double>& add_step_to_diag(Eigen::SparseMatrix<double>& matrix, double dt);
        /**Add the Dirichlet conditions to this matrix. @returns this matrix */
        Eigen::SparseMatrix<double>& add_matrix_bc(Eigen::SparseMatrix<double>& matrix);
        /**Makes the precision matrix. @returns that matrix */
        Eigen::SparseMatrix<double>& make_precision_matrix();
        /**Makes the precision matrix. @returns that matrix */
        Eigen::SparseMatrix<double>& make_precision_matrix(double dt);
        /**Makes the LHS of the system */
        Eigen::VectorXd& make_equation_lhs();
        /**Makes the LHS of the system */
        Eigen::VectorXd& make_equation_lhs(double dt);
        /**Add Dirichlet (1, 0) condition */
        Eigen::VectorXd& add_lhs_bc(Eigen::VectorXd& vec);
        /**Makes the initial 0 condition (and 1 at boundary) */
        Eigen::VectorXd& make_initial_condition();
        /**Prepares the linear system. @returns the tuple with the precision matrix and the LHS */
        void prepare_linear_system(double dt);
        /**Makes a time step. Returns the result in this step*/
        void prepare_linear_system();
        /**Makes a time step. Returns the result in this step*/
        double omegakernel(double t);

};

}
#endif
