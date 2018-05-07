
#ifndef CASADI_LPSOLVER_QP_EXPORT_H
#define CASADI_LPSOLVER_QP_EXPORT_H

#ifdef CASADI_LPSOLVER_QP_STATIC_DEFINE
#  define CASADI_LPSOLVER_QP_EXPORT
#  define CASADI_LPSOLVER_QP_NO_EXPORT
#else
#  ifndef CASADI_LPSOLVER_QP_EXPORT
#    ifdef casadi_lpsolver_qp_EXPORTS
        /* We are building this library */
#      define CASADI_LPSOLVER_QP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_LPSOLVER_QP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_LPSOLVER_QP_NO_EXPORT
#    define CASADI_LPSOLVER_QP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_LPSOLVER_QP_DEPRECATED
#  define CASADI_LPSOLVER_QP_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_LPSOLVER_QP_DEPRECATED_EXPORT CASADI_LPSOLVER_QP_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_LPSOLVER_QP_DEPRECATED_NO_EXPORT CASADI_LPSOLVER_QP_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_LPSOLVER_QP_NO_DEPRECATED
#endif

#endif
