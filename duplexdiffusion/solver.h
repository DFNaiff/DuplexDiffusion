#ifndef DUPLEX_SOLVER_H
#define DUPLEX_SOLVER_H

#include <cmath>
#include <vector>
#include <exception>
#include <tuple>
#include <deque>

#include <boost/math/special_functions/bessel.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


namespace duplexsolver
{

/**The physical parameters of the system */
struct Parameters{
    bool cylinder = false;
    double D = 6.0*1e-11*1e12; // um^2/s
    double alpha = 1.4*1e-16*1e12; // um^2/s
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
    int nkernel = 10;
    double timestep = 0.25;
    double decay_limit = 0.1;
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
        const auto& get_precision() {return m_precision;}
        /**Getter for the rhs of the system */
        const auto& get_rhs() {return m_rhs;}
        /**Getter for timesteps */
        const auto& get_timesteps() {return m_timesteps;}
        /**Get last time step */
        double last_timestep() {return m_timesteps.back();}
        /**Get the precomputed kernel values. @returns these values */
        std::vector<double>& omegavalues();
        /**Make the finite difference matrix. @returns this matrix */
        Eigen::MatrixXd& make_finite_difference_matrix();
        /**Adds the time terms to the diagonal of the finite difference matrix. @returns this matrix */
        Eigen::MatrixXd& add_step_to_diag(Eigen::MatrixXd& matrix);
        /**Add the Dirichlet conditions to this matrix. @returns this matrix */
        Eigen::MatrixXd& add_matrix_bc(Eigen::MatrixXd& matrix);
        /**Makes the precision matrix. @returns that matrix */
        Eigen::MatrixXd& make_precision_matrix();
        /**Makes the sparse matrix from the precision matrix. @returns that sparse matrix */
        Eigen::SparseMatrix<double>& make_sparse_precision_from_dense();
        /**Makes the sparse matrix from dense matrix. @returns that sparse matrix */
        Eigen::SparseMatrix<double>& make_sparse_precision_from_dense(Eigen::MatrixXd& dense);
        /**Makes the LHS of the system */
        Eigen::VectorXd& make_equation_lhs();
        /**Makes the LHS of the system */
        Eigen::VectorXd& make_equation_lhs(double dt);
        /**Add Dirichlet (1, 0) condition */
        Eigen::VectorXd& add_lhs_bc(Eigen::VectorXd& vec);
        /**Makes the initial 0 condition (and 1 at boundary) */
        Eigen::VectorXd& make_initial_condition();
        /**Prepares the linear system. @returns the tuple with the precision matrix and the LHS */
        std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> prepare_linear_system(double dt);
        /**Makes a time step. Returns the result in this step*/
        std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> prepare_linear_system();
        /**Makes a time step. Returns the result in this step*/
        Eigen::VectorXd& step();
        /**Makes a variable time step*/
        Eigen::VectorXd& step(double dt);
    private:
        Parameters m_parameters;
        SolParams m_solparams;
        std::vector<double> m_omegavals;
        Eigen::MatrixXd m_precision;
        Eigen::VectorXd m_rhs;
        std::deque<Eigen::VectorXd> m_memory;
        Eigen::PartialPivLU<Eigen::MatrixXd> m_decomposition;
        Eigen::SparseMatrix<double> m_sparse_precision;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> m_sparse_decomposition;
        std::deque<double> m_timesteps;
        double omegakernel(double t){
            double res;
            for(int k = 1; k <= m_solparams.nkernel; k++){
                double coef{};
                if(!m_parameters.cylinder){
                    coef = m_parameters.alpha*std::pow(M_PI*k/m_parameters.R, 2);
                } else {
                    coef = m_parameters.alpha*boost::math::cyl_bessel_j_zero(0.0, k);
                }
                res += std::exp(-coef*t);
            }
            return res;
        }
};

}
#endif
