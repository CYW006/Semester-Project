
#ifndef CASADI_QPSOLVER_CPLEX_EXPORT_H
#define CASADI_QPSOLVER_CPLEX_EXPORT_H

#ifdef CASADI_QPSOLVER_CPLEX_STATIC_DEFINE
#  define CASADI_QPSOLVER_CPLEX_EXPORT
#  define CASADI_QPSOLVER_CPLEX_NO_EXPORT
#else
#  ifndef CASADI_QPSOLVER_CPLEX_EXPORT
#    ifdef casadi_qpsolver_cplex_EXPORTS
        /* We are building this library */
#      define CASADI_QPSOLVER_CPLEX_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_QPSOLVER_CPLEX_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_QPSOLVER_CPLEX_NO_EXPORT
#    define CASADI_QPSOLVER_CPLEX_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_QPSOLVER_CPLEX_DEPRECATED
#  define CASADI_QPSOLVER_CPLEX_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_QPSOLVER_CPLEX_DEPRECATED_EXPORT CASADI_QPSOLVER_CPLEX_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_QPSOLVER_CPLEX_DEPRECATED_NO_EXPORT CASADI_QPSOLVER_CPLEX_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_QPSOLVER_CPLEX_NO_DEPRECATED
#endif

#endif
