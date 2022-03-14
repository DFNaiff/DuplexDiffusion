#include <cmath>
#include <vector>
#include <exception>
#include <tuple>
#include <queue>
#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/SparseQR>

# define M_PI 3.14159265358979323846

#include "solver.h"

using std::abs;


namespace duplexsolver{
    void insertion_or_coeffRef(Eigen::SparseMatrix<double>& matrix, int i, int j, double coef, bool initialized){
        if(initialized){
            matrix.coeffRef(i, j) = coef;
        } else {
            matrix.insert(i, j) = coef;
        }
    }

    Solver::Solver(PhysicalParams physparams, SolverParams solparams, BoundaryConditions bcconditions)
                :m_physparams{physparams},
                 m_solparams{solparams},
                 m_bcconditions{bcconditions},
                 m_omegavals{},
                 m_rhs(solparams.nspace),
                 m_memory{},
                 m_sparse_precision(solparams.nspace, solparams.nspace),
                 m_sparse_decomposition{},
                 m_sparse_initialized{false},
                 m_timesteps{} {
                    omegavalues(); //Precompute the values of omega
                    make_initial_condition(); //Creates the initial condition
                    make_precision_matrix(); //Creates the precision matrix
                 };

    std::vector<double> Solver::get_xspace(){
        std::vector<double> res;
        for(int i = 0; i < m_solparams.nspace; i++){
            res.push_back(double(i)/(m_solparams.nspace-1)*m_physparams.L + m_physparams.xinit);
        }
        return res;
    }

    double Solver::omegakernel(double t){
        double res = 0;
        double base = 1.0;
        for(int k = 1; k <= m_solparams.maxkernel; k++){
            double coef{};
            if(m_physparams.precip_geom == 0){ //Spherical case
                coef = m_physparams.alpha*std::pow(M_PI*k/m_physparams.R, 2);
            } else if (m_physparams.precip_geom == 1){ //Cylindrical case
                coef = m_physparams.alpha*std::pow(boost::math::cyl_bessel_j_zero(0.0, k)/m_physparams.R, 2);
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

    std::vector<double>& Solver::omegavalues(){
        m_omegavals.clear();
        int j = 0;
        double t = m_solparams.timestep*(j + 0.5);
        double omegabase = omegakernel(t);
        double omega = omegabase;
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

    Eigen::SparseMatrix<double>& Solver::make_finite_difference_matrix(Eigen::SparseMatrix<double>& matrix, bool initialized){
        //n points -> n-1 intervals -> h = physparams.L/(n-1)
        int n = m_solparams.nspace;
        double h = m_physparams.L/(n-1); //The grid spacing
        double coeff1_base = m_physparams.D/h;
        double coeff2 = m_physparams.D/(h*h); //Finite element coefficient
        double drift_const = m_physparams.bulk_geom; //Nomeclature coincides with drift_const
        std::vector<double> rspace = get_xspace(); //1/x. We will replace it inplace
        for(unsigned i = 0; i < rspace.size(); i++){
            double x = rspace[i];
            if(x == 0.0){
                rspace[i] = 0.0; //This should be suppressed by a boundary condition anyway
            } else {
                rspace[i] = 1/x;
            }
        }

        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                if(j == i){ //x_i
                    double item = 2*coeff2;
                    insertion_or_coeffRef(matrix, i, j, item, initialized);
                } else if (j == i - 1) { //x_{i-1}
                    double item = -(coeff2 - drift_const*rspace[i]*coeff1_base);
                    insertion_or_coeffRef(matrix, i, j, item, initialized);
                } else if (j == i + 1){ //x_{i+1}
                    double item = -(coeff2 + drift_const*rspace[i]*coeff1_base);
                    insertion_or_coeffRef(matrix, i, j, item, initialized);
                } else { //No coefficient
                    double item = 0.0;
                    insertion_or_coeffRef(matrix, i, j, item, initialized);
                }
            }
        }
        return matrix;
    }

    Eigen::SparseMatrix<double>& Solver::add_step_to_diag(Eigen::SparseMatrix<double>& matrix){ //Adds to diagonal of precision matrix the LHS terms
        double constant = 1.0/m_solparams.timestep + m_physparams.beta()*m_omegavals[0]; //The LHS term
        int n = m_solparams.nspace;
        for(int i = 0; i < n; i++){ //Adds to the diagonal
            matrix.coeffRef(i, i) += constant;
        }
        return matrix;
    }

    Eigen::SparseMatrix<double>& Solver::add_step_to_diag(Eigen::SparseMatrix<double>& matrix, double dt){ //Adds to diagonal of precision matrix the LHS terms
        double constant = 1.0/dt + m_physparams.beta()*omegakernel(dt/2); //The LHS term
        int n = m_solparams.nspace;
        for(int i = 0; i < n; i++){ //Adds to the diagonal
            matrix.coeffRef(i, i) += constant;
        }
        return matrix;
        //
    }

    Eigen::SparseMatrix<double>& Solver::add_matrix_bc(Eigen::SparseMatrix<double>& matrix){
        double h = m_physparams.L/(m_solparams.nspace-1); //The grid spacing
        if(m_bcconditions.left_bc_type == 1){ //x_0 = left_bc_value
            matrix.coeffRef(0, 0) = 1.0;
            matrix.coeffRef(0, 1) = 0.0;
        } else if (m_bcconditions.left_bc_type == 2){ //(x_1 - x_0)/h = -right_bc_value -> x0/h - x1/h = right_bc_value (we consider flux inlet condition here)
            matrix.coeffRef(0, 0) = 1.0/h;
            matrix.coeffRef(0, 1) = 1.0/h;
        }
        if(m_bcconditions.right_bc_type == 1){ //
            matrix.coeffRef(matrix.rows() - 1, matrix.rows() - 1) = 1.0;
            matrix.coeffRef(matrix.rows() - 1, matrix.rows() - 2) = 0.0;
        } else if (m_bcconditions.right_bc_type == 2){ //(x_{n-1} - x_{n-2})/h = right_bc_value
            matrix.coeffRef(matrix.rows() - 1, matrix.rows() - 1) = 1.0/h;
            matrix.coeffRef(matrix.rows() - 1, matrix.rows() - 2) = -1.0/h;
        }
        return matrix;
    }

    Eigen::SparseMatrix<double>& Solver::make_precision_matrix(){
        make_finite_difference_matrix(m_sparse_precision, m_sparse_initialized);
        add_step_to_diag(m_sparse_precision);
        add_matrix_bc(m_sparse_precision);
        m_sparse_decomposition.compute(m_sparse_precision); //Computes the decomposition
        if(!m_sparse_initialized){
            m_sparse_initialized = true;
        }
        return m_sparse_precision;
    }

    Eigen::SparseMatrix<double>& Solver::make_precision_matrix(double dt){
        make_finite_difference_matrix(m_sparse_precision, m_sparse_initialized);
        add_step_to_diag(m_sparse_precision, dt);
        add_matrix_bc(m_sparse_precision);
        m_sparse_decomposition.compute(m_sparse_precision); //Computes the decomposition
        if(!m_sparse_initialized){
            m_sparse_initialized = true;
        }
        return m_sparse_precision;
    }

    Eigen::VectorXd& Solver::make_equation_lhs(){
        if(m_memory.size() == 0){
            throw std::length_error("Empty list");
        }
        double beta = m_physparams.beta();
        double cn_constant = 1.0/m_solparams.timestep + beta*m_omegavals[0]; //This is the term c_{n}
        Eigen::VectorXd res = cn_constant*m_memory[m_memory.size() - 1]; //This is the term containing only c_{n}
        for(int i = m_memory.size() - 2; i>= 0; i--){ //n-1, n, ..., 0
            //Access memory right from left, and precomputed kernel values from left to right
            int omega_i = m_memory.size() - 1 - i; //1, 2, ...
            int maxwindow = m_omegavals.size();
            if(omega_i >= maxwindow){ //Exceeded the precomputed window
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
        double beta = m_physparams.beta();
        double omega_base = omegakernel(dt/2); //(t_{i} + dt - (t_{i} + d_t + t_i)/2) = dt/2
        double cn_constant = 1.0/dt + beta*omega_base; //This is the term c_{n}
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
        vec(0) = m_bcconditions.left_bc_value;
        vec(vec.rows()-1) = m_bcconditions.right_bc_value;
        return vec;
    }

    Eigen::VectorXd& Solver::make_initial_condition(){
        m_memory.clear();
        m_timesteps.clear();
        Eigen::VectorXd cvec(m_solparams.nspace);
        //Puts the initial condition zero (boundary values will be set automatically later)
        //cvec(0) = m_physparams.cinit;
        for(int i = 0; i < m_solparams.nspace; i++){
            cvec(i) = 0.0;
        }
        m_memory.push_back(cvec);
        m_timesteps.push_back(0.0);
        return m_memory[0];
    }

    void Solver::prepare_linear_system(double dt){
        make_precision_matrix(dt);
        auto& lhs = make_equation_lhs(dt);
        add_lhs_bc(lhs);
        return;
    }

    void Solver::prepare_linear_system(){
        auto& lhs = make_equation_lhs();
        add_lhs_bc(lhs);
        return;
    }

    Eigen::VectorXd& Solver::step(){
        prepare_linear_system(); //Prepare the LHS of the system
        Eigen::VectorXd cnew = m_sparse_decomposition.solve(m_rhs); //Solves the system
        m_memory.push_back(cnew); //Add to memory
        if(int(m_memory.size()) > m_solparams.maxwindow){
            m_memory.pop_front();
        }
        m_timesteps.push_back(m_timesteps.back() + m_solparams.timestep);
        if(int(m_timesteps.size()) > m_solparams.maxwindow){
            m_timesteps.pop_front();
        };
        return m_memory[m_memory.size()-1];
    }

    Eigen::VectorXd& Solver::step(double dt){
        prepare_linear_system(dt); //Prepare the LHS of the system
        Eigen::VectorXd cnew = m_sparse_decomposition.solve(m_rhs); //Solves the system
        m_memory.push_back(cnew); //Add to memory
        if(int(m_memory.size()) > m_solparams.maxwindow){
            m_memory.pop_front();
        }
        m_timesteps.push_back(m_timesteps.back() + dt);
        if(int(m_timesteps.size()) > m_solparams.maxwindow){
            m_timesteps.pop_front();
        };
        return m_memory[m_memory.size()-1];
    }

    void Solver::cancel_last_step(){
        m_memory.pop_back();
        m_timesteps.pop_back();
    }

    double Solver::get_last_step_error(){
        if(m_memory.size() < 2){
            throw std::length_error("One or zero member in memory");
        }
        Eigen::VectorXd& cnew = m_memory[m_memory.size()-1];
        Eigen::VectorXd& cprev = m_memory[m_memory.size()-2];
        double res = (cnew - cprev).cwiseAbs().maxCoeff()/(cprev.cwiseAbs().maxCoeff());
        return res;
    }
}
