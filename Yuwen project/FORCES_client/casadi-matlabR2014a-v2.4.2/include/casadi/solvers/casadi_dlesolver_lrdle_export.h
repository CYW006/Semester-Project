
#ifndef CASADI_DLESOLVER_LRDLE_EXPORT_H
#define CASADI_DLESOLVER_LRDLE_EXPORT_H

#ifdef CASADI_DLESOLVER_LRDLE_STATIC_DEFINE
#  define CASADI_DLESOLVER_LRDLE_EXPORT
#  define CASADI_DLESOLVER_LRDLE_NO_EXPORT
#else
#  ifndef CASADI_DLESOLVER_LRDLE_EXPORT
#    ifdef casadi_dlesolver_lrdle_EXPORTS
        /* We are building this library */
#      define CASADI_DLESOLVER_LRDLE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_DLESOLVER_LRDLE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_DLESOLVER_LRDLE_NO_EXPORT
#    define CASADI_DLESOLVER_LRDLE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_DLESOLVER_LRDLE_DEPRECATED
#  define CASADI_DLESOLVER_LRDLE_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_DLESOLVER_LRDLE_DEPRECATED_EXPORT CASADI_DLESOLVER_LRDLE_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_DLESOLVER_LRDLE_DEPRECATED_NO_EXPORT CASADI_DLESOLVER_LRDLE_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_DLESOLVER_LRDLE_NO_DEPRECATED
#endif

#endif
