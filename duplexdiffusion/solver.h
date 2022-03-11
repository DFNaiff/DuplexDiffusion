#ifndef DUPLEX_SOLVER_H
#define DUPLEX_SOLVER_H

#include <cmath>
#include <vector>
#include <exception>
#include <tuple>
#include <deque>
#include <iostream>

#include <boost/math/special_functions/bessel.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


namespace duplexsolver
{

/**The physical parameters of the system */
struct Parameters{
    bool cylinder = false;
    double D = 60.0; // um^2/s
    double alpha = 1.4*1e-4; // um^2/s
    double R = 5.0; // um
    double K = 32.51/0.033; //unitless
    double vol_fraction = 0.4; //unitless
    double area_fraction = 0.4; //unitless
    double L = 100; // #um
    double cinit = 1.0; // #mol/um^3
    double Gamma(){
        if(!cylinder){return vol_fraction/(4.0/3*M_PI*R*R*R);} //1/um^3
        else{return area_fraction/(4.0*M_PI*R*R);} //1/um^2
    }
    double beta(){
        if(!cylinder){return 8*M_PI*R*Gamma()*alpha*K;}
        else{return 4*M_PI*Gamma()*alpha*K;}
    }
};

/**The solver parameters */
struct SolParams{
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
         * @param parameters - The physical parameters of the problem
         * @param solparams - The solver parameters
         */
        Solver(Parameters parameters, SolParams solparams);
        /**Getter for the memory of cvalues */
        const auto& get_memory() {return m_memory;}
        /**Getter for the precision matrix */
        const auto& get_timesteps() {return m_timesteps;}
        /**Get last time step */
        double last_timestep() {return m_timesteps.back();}
        /**Makes a fixed time step*/
        Eigen::VectorXd& step();
        /**Makes a variable time step*/
        Eigen::VectorXd& step(double dt);
        /**Cancels the last step*/
        void cancel_last_step();
        /**Get last step error*/
        double get_last_step_error();
    private:
        Parameters m_parameters;
        SolParams m_solparams;
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
        double omegakernel(double t){
            double res = 0;
            double base = 1.0;
            for(int k = 1; k <= m_solparams.maxkernel; k++){
                double coef{};
                if(!m_parameters.cylinder){
                    coef = m_parameters.alpha*std::pow(M_PI*k/m_parameters.R, 2);
                } else {
                    coef = m_parameters.alpha*std::pow(boost::math::cyl_bessel_j_zero(0.0, k)/m_parameters.R, 2);
                }
                double increment = std::exp(-coef*t);
                if(k == 1){
                    base = increment;
                } else {
                    if(increment/base < m_solparams.decay_limit){
                        break;
                    }
                }
                res += increment;
            }
            return res;
        }

};

}
#endif
