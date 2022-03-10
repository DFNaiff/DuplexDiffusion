#include <cmath>
#include <vector>
#include <exception>
#include <tuple>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseQR>

# define M_PI 3.14159265358979323846

#include "solver.h"

namespace duplexsolver{


Solver::Solver(Parameters parameters, SolParams solparams)
            :m_parameters{parameters},
             m_solparams{solparams},
             m_omegavals{},
             m_precision(solparams.nspace, solparams.nspace),
             m_rhs(solparams.nspace),
             m_memory{},
             m_decomposition{},
             m_sparse_precision(solparams.nspace, solparams.nspace) {
                omegavalues();
                make_initial_condition();
                make_precision_matrix();
                make_sparse_precision_from_dense();
             };

std::vector<double>& Solver::omegavalues(){
    m_omegavals.clear();
    int j = 0;
    double t = m_solparams.timestep*(j + 0.5);
    float omegabase = omegakernel(t);
    float omega = omegabase;
    m_omegavals.push_back(omega);
    while(omega/omegabase > m_solparams.decay_limit){
        j += 1;
        t = m_solparams.timestep*(j + 0.5);
        omega = omegakernel(t);
        m_omegavals.push_back(omega);
        if(j >= m_solparams.maxwindow-1){
            break;
        }
    }
    return m_omegavals;
}

Eigen::MatrixXd& Solver::make_finite_difference_matrix(){
    //n points -> n-1 intervals -> h = parameters.L/(n-1)
    int n = m_solparams.nspace;
    double h = m_parameters.L/(n-1);
    double coef = m_parameters.D/(h*h);
    Eigen::MatrixXd res(n, n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i == j){
                res(i, j) = 2*coef;
            } else if ((i == j - 1) || (i == j + 1)){
                res(i, j) = -coef;
            } else {
                res(i, j) = 0.0;
            }
        }
    }
    m_precision.topLeftCorner(n, n) = res.topLeftCorner(n, n);
    return m_precision;
}

Eigen::MatrixXd& Solver::add_step_to_diag(Eigen::MatrixXd& matrix){
    double constant = 1.0/m_solparams.timestep + m_parameters.beta()*m_omegavals[0];
    int n = m_solparams.nspace;
    for(int i = 0; i < n; i++){
        matrix(i, i) += constant;
    }
    return matrix;
}

Eigen::MatrixXd& Solver::add_matrix_bc(Eigen::MatrixXd& matrix){
    //Dirichlet conditions
    matrix(0, 0) = 1.0;
    for(int j = 1; j < matrix.rows(); j++){
        matrix(0, j) = 0.0;
    }
    matrix(matrix.rows() - 1, matrix.rows() - 1) = 1.0;
    for(int j = 0; j < matrix.rows() - 1; j++){
        matrix(matrix.rows() - 1, j) = 0.0;
    }
    return matrix;
}

Eigen::MatrixXd& Solver::make_precision_matrix(){
    auto& precision_matrix = make_finite_difference_matrix();
    precision_matrix = add_step_to_diag(precision_matrix);
    precision_matrix = add_matrix_bc(precision_matrix);
    m_decomposition.compute(precision_matrix);
    return precision_matrix;
}

Eigen::SparseMatrix<double>& Solver::make_sparse_precision_from_dense(){
    return make_sparse_precision_from_dense(m_precision);
}

Eigen::SparseMatrix<double>& Solver::make_sparse_precision_from_dense(Eigen::MatrixXd& dense){
    //m_sparse_precision.reserve(Eigen::VectorXi(dense.cols(), 3));
    for(int i = 0; i < dense.cols(); i++){
        if(i > 0){
            m_sparse_precision.insert(i, i-1) = dense(i, i-1);
        }
        if(i < dense.cols() - 1){
            m_sparse_precision.insert(i, i+1) = dense(i, i+1);
        }
        m_sparse_precision.insert(i, i) = dense(i, i);
    }
    m_sparse_precision.makeCompressed();
    m_sparse_decomposition.compute(m_sparse_precision);
    return m_sparse_precision;
}

Eigen::VectorXd& Solver::make_equation_lhs(){
    if(m_memory.size() == 0){
        throw std::length_error("Empty list");
    }
    double beta = m_parameters.beta();
    double cn_constant = 1.0/m_solparams.timestep + beta*m_omegavals[0];
    Eigen::VectorXd res = cn_constant*m_memory[m_memory.size() - 1];
    for(int i = m_memory.size() - 2; i>= 0; i--){
        int omega_i = m_memory.size() - 1 - i; //1, 2, ...
        if(omega_i >= m_omegavals.size()){
            break;
        }
        res -= beta*(m_memory[i+1] - m_memory[i])*m_omegavals[omega_i];
    }
    m_rhs.col(0) = res.col(0);
    return m_rhs;
}

Eigen::VectorXd& Solver::add_lhs_bc(Eigen::VectorXd& vec){
    //Dirichlet conditions : c(0) = c_init, c(L) = 0;
    vec(0) = m_parameters.cinit;
    vec(vec.rows()-1) = 0;
    return vec;
}

Eigen::VectorXd& Solver::make_initial_condition(){
    m_memory.clear();
    Eigen::VectorXd cvec(m_solparams.nspace);
    cvec(0) = m_parameters.cinit;
    for(int i = 1; i < m_solparams.nspace; i++){
        cvec(i) = 0.0;
    }
    m_memory.push_back(cvec);
    return m_memory[0];
}

std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> Solver::prepare_linear_system(){
    auto& lhs = make_equation_lhs();
    add_lhs_bc(lhs);
    std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> res(m_precision, lhs);
    return res;
}

Eigen::VectorXd& Solver::step(){
    prepare_linear_system();
    Eigen::VectorXd cnew = m_sparse_decomposition.solve(m_rhs);
    m_memory.push_back(cnew);
    return m_memory[m_memory.size()-1];
}

}
