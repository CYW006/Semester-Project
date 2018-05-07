
#ifndef CASADI_LRDPLESOLVER_LIFTING_EXPORT_H
#define CASADI_LRDPLESOLVER_LIFTING_EXPORT_H

#ifdef CASADI_LRDPLESOLVER_LIFTING_STATIC_DEFINE
#  define CASADI_LRDPLESOLVER_LIFTING_EXPORT
#  define CASADI_LRDPLESOLVER_LIFTING_NO_EXPORT
#else
#  ifndef CASADI_LRDPLESOLVER_LIFTING_EXPORT
#    ifdef casadi_lrdplesolver_lifting_EXPORTS
        /* We are building this library */
#      define CASADI_LRDPLESOLVER_LIFTING_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_LRDPLESOLVER_LIFTING_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_LRDPLESOLVER_LIFTING_NO_EXPORT
#    define CASADI_LRDPLESOLVER_LIFTING_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_LRDPLESOLVER_LIFTING_DEPRECATED
#  define CASADI_LRDPLESOLVER_LIFTING_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_LRDPLESOLVER_LIFTING_DEPRECATED_EXPORT CASADI_LRDPLESOLVER_LIFTING_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_LRDPLESOLVER_LIFTING_DEPRECATED_NO_EXPORT CASADI_LRDPLESOLVER_LIFTING_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_LRDPLESOLVER_LIFTING_NO_DEPRECATED
#endif

#endif
