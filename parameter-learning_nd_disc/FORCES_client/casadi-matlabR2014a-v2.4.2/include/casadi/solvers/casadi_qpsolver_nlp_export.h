
#ifndef CASADI_QPSOLVER_NLP_EXPORT_H
#define CASADI_QPSOLVER_NLP_EXPORT_H

#ifdef CASADI_QPSOLVER_NLP_STATIC_DEFINE
#  define CASADI_QPSOLVER_NLP_EXPORT
#  define CASADI_QPSOLVER_NLP_NO_EXPORT
#else
#  ifndef CASADI_QPSOLVER_NLP_EXPORT
#    ifdef casadi_qpsolver_nlp_EXPORTS
        /* We are building this library */
#      define CASADI_QPSOLVER_NLP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_QPSOLVER_NLP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_QPSOLVER_NLP_NO_EXPORT
#    define CASADI_QPSOLVER_NLP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_QPSOLVER_NLP_DEPRECATED
#  define CASADI_QPSOLVER_NLP_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_QPSOLVER_NLP_DEPRECATED_EXPORT CASADI_QPSOLVER_NLP_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_QPSOLVER_NLP_DEPRECATED_NO_EXPORT CASADI_QPSOLVER_NLP_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_QPSOLVER_NLP_NO_DEPRECATED
#endif

#endif
