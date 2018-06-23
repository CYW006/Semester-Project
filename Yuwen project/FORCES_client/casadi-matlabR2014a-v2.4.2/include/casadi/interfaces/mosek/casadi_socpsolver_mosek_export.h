
#ifndef CASADI_SOCPSOLVER_MOSEK_EXPORT_H
#define CASADI_SOCPSOLVER_MOSEK_EXPORT_H

#ifdef CASADI_SOCPSOLVER_MOSEK_STATIC_DEFINE
#  define CASADI_SOCPSOLVER_MOSEK_EXPORT
#  define CASADI_SOCPSOLVER_MOSEK_NO_EXPORT
#else
#  ifndef CASADI_SOCPSOLVER_MOSEK_EXPORT
#    ifdef casadi_socpsolver_mosek_EXPORTS
        /* We are building this library */
#      define CASADI_SOCPSOLVER_MOSEK_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_SOCPSOLVER_MOSEK_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_SOCPSOLVER_MOSEK_NO_EXPORT
#    define CASADI_SOCPSOLVER_MOSEK_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_SOCPSOLVER_MOSEK_DEPRECATED
#  define CASADI_SOCPSOLVER_MOSEK_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_SOCPSOLVER_MOSEK_DEPRECATED_EXPORT CASADI_SOCPSOLVER_MOSEK_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_SOCPSOLVER_MOSEK_DEPRECATED_NO_EXPORT CASADI_SOCPSOLVER_MOSEK_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_SOCPSOLVER_MOSEK_NO_DEPRECATED
#endif

#endif
