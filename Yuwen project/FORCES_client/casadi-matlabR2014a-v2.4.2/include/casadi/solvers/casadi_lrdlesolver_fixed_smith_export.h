
#ifndef CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT_H
#define CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT_H

#ifdef CASADI_LRDLESOLVER_FIXED_SMITH_STATIC_DEFINE
#  define CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT
#  define CASADI_LRDLESOLVER_FIXED_SMITH_NO_EXPORT
#else
#  ifndef CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT
#    ifdef casadi_lrdlesolver_fixed_smith_EXPORTS
        /* We are building this library */
#      define CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_LRDLESOLVER_FIXED_SMITH_NO_EXPORT
#    define CASADI_LRDLESOLVER_FIXED_SMITH_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_LRDLESOLVER_FIXED_SMITH_DEPRECATED
#  define CASADI_LRDLESOLVER_FIXED_SMITH_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_LRDLESOLVER_FIXED_SMITH_DEPRECATED_EXPORT CASADI_LRDLESOLVER_FIXED_SMITH_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_LRDLESOLVER_FIXED_SMITH_DEPRECATED_NO_EXPORT CASADI_LRDLESOLVER_FIXED_SMITH_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_LRDLESOLVER_FIXED_SMITH_NO_DEPRECATED
#endif

#endif
