
#ifndef CASADI_DPLESOLVER_LIFTING_EXPORT_H
#define CASADI_DPLESOLVER_LIFTING_EXPORT_H

#ifdef CASADI_DPLESOLVER_LIFTING_STATIC_DEFINE
#  define CASADI_DPLESOLVER_LIFTING_EXPORT
#  define CASADI_DPLESOLVER_LIFTING_NO_EXPORT
#else
#  ifndef CASADI_DPLESOLVER_LIFTING_EXPORT
#    ifdef casadi_dplesolver_lifting_EXPORTS
        /* We are building this library */
#      define CASADI_DPLESOLVER_LIFTING_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_DPLESOLVER_LIFTING_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_DPLESOLVER_LIFTING_NO_EXPORT
#    define CASADI_DPLESOLVER_LIFTING_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_DPLESOLVER_LIFTING_DEPRECATED
#  define CASADI_DPLESOLVER_LIFTING_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_DPLESOLVER_LIFTING_DEPRECATED_EXPORT CASADI_DPLESOLVER_LIFTING_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_DPLESOLVER_LIFTING_DEPRECATED_NO_EXPORT CASADI_DPLESOLVER_LIFTING_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_DPLESOLVER_LIFTING_NO_DEPRECATED
#endif

#endif
