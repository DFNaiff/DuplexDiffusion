#ifndef DUPLEX_CONTROLLER_H
#define DUPLEX_CONTROLLER_H

#include <deque>
#include <exception>
#include <cmath>
#include <iostream>

//https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.582.4300&rep=rep1&type=pdf

namespace control{
    class Controller{
        public:
            Controller(double dtmin, double dtmax,
                       double errmax=0.01, double varmin=0.5, double varmax=2.0)
                :m_dtmin{dtmin}, m_dtmax{dtmax}, m_errmax{errmax}, m_varmin{varmin}, m_varmax{varmax},
                 m_dtholder{0.0}, m_dtlast{dtmax/10}, m_errors{}
                {}
            /**Get the next time step of system */
            double suggest_step(){
                double res{};
                if(m_dtholder > 0.0){ //There is suggestion in memory. Use it.
                    res = m_dtholder;
                    m_dtholder = 0.0;
                } else { //N
                    if(m_errors.size() < 3){
                        res = m_dtlast;
                    } else if (m_errors.size() == 3){
                        double e0 = m_errors[2];
                        double e1 = m_errors[1];
                        double e2 = m_errors[0];
                        res = std::pow(e1/e0, m_kp)*std::pow(m_errmax/e0, m_ki)*std::pow(e1*e1/(e0*e2), m_kd)*m_dtlast;
                        if(res < m_dtmin){
                            res = m_dtmin;
                        }
                        if(res > m_dtmax){
                            res = m_dtmax;
                        }
                        if(res < (m_varmin*m_dtlast)){
                            res = m_varmin*m_dtlast;
                        }
                        if(res > (m_varmax*m_dtlast)){
                            res = m_varmax*m_dtlast;
                        }
                    } else if (m_errors.size() > 3){
                        throw std::length_error("Oversized memory (some bug)");
                    }
                }
                m_dtlast = res;
                return res;
            }
            /**Evaluate whether to accept step or cancel it */
            bool evaluate_step(double err){
                if((err <= m_errmax) || (m_dtlast <= m_dtmin)){ //Accept error and add it to memory
                    m_errors.push_back(err);
                    if(m_errors.size() > 3){ //Only accepts at most three in memory
                        m_errors.pop_front();
                    }
                    m_dtholder = 0.0; //Empties the holder
                    return true;
                } else { //Reject the error and suggest next timestep
                    m_dtholder = m_errmax/err*m_dtlast;
                    return false;
                }
            }
        private:
            double m_dtmin;
            double m_dtmax;
            double m_errmax;
            double m_varmin;
            double m_varmax;

            double m_dtholder; //0.0 if there is no dt to be suggested already
            double m_dtlast; //Will be initialized to m_dtmin
            std::deque<double> m_errors; //Hold the latest three errors
            double m_kp = 0.075;
            double m_ki = 0.175;
            double m_kd = 0.01;

    };
}
#endif
