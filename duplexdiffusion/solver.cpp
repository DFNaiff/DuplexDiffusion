#include <cmath>
#include <vector>
#include <exception>
#include <tuple>
#include <queue>

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
             m_sparse_precision(solparams.nspace, solparams.nspace),
             m_timesteps{} {
                omegavalues(); //Precompute the values of omega
                make_initial_condition(); //Creates the initial condition
                make_precision_matrix(); //Creates the precision matrix
                make_sparse_precision_from_dense(); //Makes the sparse precision matrix
             };

std::vector<double>& Solver::omegavalues(){
    m_omegavals.clear();
    int j = 0;
    double t = m_solparams.timestep*(j + 0.5);
    float omegabase = omegakernel(t);
    float omega = omegabase;
    m_omegavals.push_back(omega);
    while(omega/omegabase > m_solparams.decay_limit){ //The main condition of significanse
        j += 1;
        t = m_solparams.timestep*(j + 0.5);
        omega = omegakernel(t);
        m_omegavals.push_back(omega);
        if(j >= m_solparams.maxwindow-1){ //Windows has exceeded side
            break;
        }
    }
    return m_omegavals;
}

Eigen::MatrixXd& Solver::make_finite_difference_matrix(){
    //n points -> n-1 intervals -> h = parameters.L/(n-1)
    int n = m_solparams.nspace;
    double h = m_parameters.L/(n-1); //The grid spacing
    double coef = m_parameters.D/(h*h); //Finite element coefficient
    Eigen::MatrixXd res(n, n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i == j){
                res(i, j) = 2*coef; //Diagonal
            } else if ((i == j - 1) || (i == j + 1)){
                res(i, j) = -coef; //Off-diagonal
            } else {
                res(i, j) = 0.0; //Zero at the rest
            }
        }
    }
    m_precision.topLeftCorner(n, n) = res.topLeftCorner(n, n); //Copies to precision matrix
    return m_precision;
}

Eigen::MatrixXd& Solver::add_step_to_diag(Eigen::MatrixXd& matrix){ //Adds to diagonal of precision matrix the LHS terms
    double constant = 1.0/m_solparams.timestep + m_parameters.beta()*m_omegavals[0]; //The LHS term
    int n = m_solparams.nspace;
    for(int i = 0; i < n; i++){ //Adds to the diagonal
        matrix(i, i) += constant;
    }
    return matrix;
}

Eigen::MatrixXd& Solver::add_matrix_bc(Eigen::MatrixXd& matrix){
    //Dirichlet conditions
    //Put dirichlet condition on first line, [1, 0,...,0]
    matrix(0, 0) = 1.0;
    for(int j = 1; j < matrix.rows(); j++){
        matrix(0, j) = 0.0;
    }
    matrix(matrix.rows() - 1, matrix.rows() - 1) = 1.0;
    //Put dirichlet condition on final line, [0, ..., 0, 1]
    for(int j = 0; j < matrix.rows() - 1; j++){
        matrix(matrix.rows() - 1, j) = 0.0;
    }
    return matrix;
}

Eigen::MatrixXd& Solver::make_precision_matrix(){
    auto& precision_matrix = make_finite_difference_matrix();
    precision_matrix = add_step_to_diag(precision_matrix);
    precision_matrix = add_matrix_bc(precision_matrix);
    m_decomposition.compute(precision_matrix); //Computes the decomposition. Legacy, using sparse now
    return precision_matrix;
}

Eigen::SparseMatrix<double>& Solver::make_sparse_precision_from_dense(){
    return make_sparse_precision_from_dense(m_precision);
}

Eigen::SparseMatrix<double>& Solver::make_sparse_precision_from_dense(Eigen::MatrixXd& dense){
    //m_sparse_precision.reserve(Eigen::VectorXi(dense.cols(), 3));
    //Just go through the diagonal and off-diagonal and add to sparse
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
    m_sparse_decomposition.compute(m_sparse_precision); //Computes the LU decomposition
    return m_sparse_precision;
}

Eigen::VectorXd& Solver::make_equation_lhs(){
    if(m_memory.size() == 0){
        throw std::length_error("Empty list");
    }
    double beta = m_parameters.beta();
    double cn_constant = 1.0/m_solparams.timestep + beta*m_omegavals[0]; //This is the term c_{n}
    Eigen::VectorXd res = cn_constant*m_memory[m_memory.size() - 1]; //This is the term containing only c_{n}
    for(int i = m_memory.size() - 2; i>= 0; i--){ //n-1, n, ..., 0
        //Access memory right from left, and precomputed kernel values from left to right
        int omega_i = m_memory.size() - 1 - i; //1, 2, ...
        if(omega_i >= m_omegavals.size()){ //Exceeded the precomputed window
            break;
        }
        res -= beta*(m_memory[i+1] - m_memory[i])*m_omegavals[omega_i]; //The summation terms
    }
    m_rhs.col(0) = res.col(0); //Copy
    return m_rhs;
}

Eigen::VectorXd& Solver::make_equation_lhs(double dt){
    if(m_memory.size() == 0){
        throw std::length_error("Empty list");
    }
    double beta = m_parameters.beta();
    double omega_base = omegakernel(dt/2); //(t_{i} + dt - (t_{i} + d_t + t_i)/2) = dt/2
    double cn_constant = 1.0/m_solparams.timestep + beta*omega_base; //This is the term c_{n}
    Eigen::VectorXd res = cn_constant*m_memory[m_memory.size() - 1]; //This is the term containing only c_{n}
    for(int i = m_memory.size() - 2; i>= 0; i--){ //n-1, n, ..., 0
        //Calculates the corresponding omega value
        double omegat = m_timesteps.back() + dt - (m_timesteps[i] + m_timesteps[i+1])/2;
        double omega = omegakernel(omegat);
        if(omega/omega_base < m_solparams.decay_limit){
            break;
        }
        res -= beta*(m_memory[i+1] - m_memory[i])*omega; //The summation terms
    }
    m_rhs.col(0) = res.col(0); //Copy
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
    m_timesteps.clear();
    Eigen::VectorXd cvec(m_solparams.nspace);
    //Puts the initial condition zero everywhere except at the boundary
    cvec(0) = m_parameters.cinit;
    for(int i = 1; i < m_solparams.nspace; i++){
        cvec(i) = 0.0;
    }
    m_memory.push_back(cvec);
    m_timesteps.push_back(0.0);
    return m_memory[0];
}

std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> Solver::prepare_linear_system(double dt){
    auto& lhs = make_equation_lhs();
    add_lhs_bc(lhs);
    std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> res(m_precision, lhs); //Makes the return tuple
    return res;
}

std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> Solver::prepare_linear_system(){
    auto& lhs = make_equation_lhs();
    add_lhs_bc(lhs);
    std::tuple<Eigen::MatrixXd&, Eigen::VectorXd&> res(m_precision, lhs); //Makes the return tuple
    return res;
}

Eigen::VectorXd& Solver::step(){
    prepare_linear_system(); //Prepare the LHS of the system
    Eigen::VectorXd cnew = m_sparse_decomposition.solve(m_rhs); //Solves the system
    m_memory.push_back(cnew); //Add to memory
    //if(m_memory.size() > m_solparams.maxwindow){m_memory.pop_front();};
    m_timesteps.push_back(m_timesteps.back() + m_solparams.timestep);
    //if(m_timesteps.size() > m_solparams.maxwindow){m_memory.pop_front();};
    return m_memory[m_memory.size()-1];
}

Eigen::VectorXd& Solver::step(double dt){
    prepare_linear_system(dt); //Prepare the LHS of the system
    Eigen::VectorXd cnew = m_sparse_decomposition.solve(m_rhs); //Solves the system
    m_memory.push_back(cnew); //Add to memory
    if(m_memory.size() > m_solparams.maxwindow){m_memory.pop_front();};
    m_timesteps.push_back(m_timesteps.back() + dt);
    if(m_timesteps.size() > m_solparams.maxwindow){m_memory.pop_front();};
    return m_memory[m_memory.size()-1];
}

}
