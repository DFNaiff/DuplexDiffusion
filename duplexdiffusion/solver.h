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

struct Parameters{
    double D = 6.0*1e-11*1e12; // um^2/s
    double alpha = 1.4*1e-16*1e12; // um^2/s
    double R = 1.0; // um
    double K = 32.51/0.033; //unitless
    double vol_fraction = 0.1; //unitless
    double L = 100; // #um
    double cinit = 1.0; // #mol/um^3
    double Gamma(){
        return vol_fraction/(4.0/3*M_PI*R*R); //1/um^3
    }
    double beta(){
        return 8*M_PI*R*Gamma()*alpha*K;
    }
};

struct SolParams{
    int nspace = 21;
    int nkernel = 10;
    double timestep = 0.25;
    double decay_limit = 0.1;
    int maxwindow = 100;
};

class Solver{
    //Declarations;

    public:
        Solver(Parameters parameters, SolParams solparams);
        auto get_memory() {return m_memory;}
        auto get_precision() {return m_precision;}
        auto get_rhs() {return m_rhs;}
        std::vector<double>& omegavalues();
        Eigen::MatrixXd& make_finite_difference_matrix();
        Eigen::MatrixXd& add_step_to_diag(Eigen::MatrixXd& matrix);
        Eigen::MatrixXd& add_matrix_bc(Eigen::MatrixXd& matrix);
        Eigen::MatrixXd& make_precision_matrix();
        Eigen::SparseMatrix<double>& make_sparse_precision_from_dense();
        Eigen::SparseMatrix<double>& make_sparse_precision_from_dense(Eigen::MatrixXd& dense);
        Eigen::VectorXd& make_equation_lhs();
        Eigen::VectorXd& add_lhs_bc(Eigen::VectorXd& vec);
        Eigen::VectorXd& make_initial_condition();
        std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> prepare_linear_system();
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
