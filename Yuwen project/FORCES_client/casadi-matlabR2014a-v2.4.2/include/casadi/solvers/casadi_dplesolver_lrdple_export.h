
#ifndef CASADI_DPLESOLVER_LRDPLE_EXPORT_H
#define CASADI_DPLESOLVER_LRDPLE_EXPORT_H

#ifdef CASADI_DPLESOLVER_LRDPLE_STATIC_DEFINE
#  define CASADI_DPLESOLVER_LRDPLE_EXPORT
#  define CASADI_DPLESOLVER_LRDPLE_NO_EXPORT
#else
#  ifndef CASADI_DPLESOLVER_LRDPLE_EXPORT
#    ifdef casadi_dplesolver_lrdple_EXPORTS
        /* We are building this library */
#      define CASADI_DPLESOLVER_LRDPLE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_DPLESOLVER_LRDPLE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_DPLESOLVER_LRDPLE_NO_EXPORT
#    define CASADI_DPLESOLVER_LRDPLE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_DPLESOLVER_LRDPLE_DEPRECATED
#  define CASADI_DPLESOLVER_LRDPLE_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_DPLESOLVER_LRDPLE_DEPRECATED_EXPORT CASADI_DPLESOLVER_LRDPLE_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_DPLESOLVER_LRDPLE_DEPRECATED_NO_EXPORT CASADI_DPLESOLVER_LRDPLE_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_DPLESOLVER_LRDPLE_NO_DEPRECATED
#endif

#endif
