
#ifndef CASADI_CLESOLVER_SIMPLE_EXPORT_H
#define CASADI_CLESOLVER_SIMPLE_EXPORT_H

#ifdef CASADI_CLESOLVER_SIMPLE_STATIC_DEFINE
#  define CASADI_CLESOLVER_SIMPLE_EXPORT
#  define CASADI_CLESOLVER_SIMPLE_NO_EXPORT
#else
#  ifndef CASADI_CLESOLVER_SIMPLE_EXPORT
#    ifdef casadi_clesolver_simple_EXPORTS
        /* We are building this library */
#      define CASADI_CLESOLVER_SIMPLE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_CLESOLVER_SIMPLE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_CLESOLVER_SIMPLE_NO_EXPORT
#    define CASADI_CLESOLVER_SIMPLE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_CLESOLVER_SIMPLE_DEPRECATED
#  define CASADI_CLESOLVER_SIMPLE_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_CLESOLVER_SIMPLE_DEPRECATED_EXPORT CASADI_CLESOLVER_SIMPLE_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_CLESOLVER_SIMPLE_DEPRECATED_NO_EXPORT CASADI_CLESOLVER_SIMPLE_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_CLESOLVER_SIMPLE_NO_DEPRECATED
#endif

#endif
