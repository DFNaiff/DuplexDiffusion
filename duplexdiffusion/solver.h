#ifndef DUPLEX_SOLVER_H
#define DUPLEX_SOLVER_H

#include <cmath>
#include <vector>
#include <exception>
#include <tuple>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>


namespace duplexsolver
{

/**The physical parameters of the system */
struct Parameters{
    double D = 6.0*1e-11*1e12; // um^2/s
    double alpha = 1.4*1e-16*1e12; // um^2/s
    double R = 1.0; // um
    double K = 32.51/0.033; //unitless
    double vol_fraction = 0.1; //unitless
    double L = 100; // #um
    double cinit = 1.0; // #mol/um^3
    double Gamma(){
        return vol_fraction/(4.0/3*M_PI*R*R*R); //1/um^3
    }
    double beta(){
        return 8*M_PI*R*Gamma()*alpha*K;
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
        auto get_memory() {return m_memory;}
        /**Getter for the precision matrix */
        auto get_precision() {return m_precision;}
        /**Getter for the rhs of the system */
        auto get_rhs() {return m_rhs;}
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
        /**Add Dirichlet (1, 0) condition */
        Eigen::VectorXd& add_lhs_bc(Eigen::VectorXd& vec);
        /**Makes the initial 0 condition (and 1 at boundary) */
        Eigen::VectorXd& make_initial_condition();
        /**Prepares the linear system. @returns the tuple with the precision matrix and the LHS */
        std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> prepare_linear_system();
        /**Makes a time step. Returns the result in this step*/
        Eigen::VectorXd& step();
    private:
        Parameters m_parameters;
        SolParams m_solparams;
        std::vector<double> m_omegavals;
        Eigen::MatrixXd m_precision;
        Eigen::VectorXd m_rhs;
        std::vector<Eigen::VectorXd> m_memory;
        Eigen::PartialPivLU<Eigen::MatrixXd> m_decomposition;
        Eigen::SparseMatrix<double> m_sparse_precision;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> m_sparse_decomposition;
        double omegakernel(double t){
            double res;
            for(int k = 1; k <= m_solparams.nkernel; k++){
                double coef = m_parameters.alpha*std::pow(M_PI*k/m_parameters.R, 2);
                res += std::exp(-coef*t);
            }
            return res;
        }
};

}
#endif
