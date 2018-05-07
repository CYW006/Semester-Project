
#ifndef CASADI_LRDLESOLVER_DLE_EXPORT_H
#define CASADI_LRDLESOLVER_DLE_EXPORT_H

#ifdef CASADI_LRDLESOLVER_DLE_STATIC_DEFINE
#  define CASADI_LRDLESOLVER_DLE_EXPORT
#  define CASADI_LRDLESOLVER_DLE_NO_EXPORT
#else
#  ifndef CASADI_LRDLESOLVER_DLE_EXPORT
#    ifdef casadi_lrdlesolver_dle_EXPORTS
        /* We are building this library */
#      define CASADI_LRDLESOLVER_DLE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_LRDLESOLVER_DLE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_LRDLESOLVER_DLE_NO_EXPORT
#    define CASADI_LRDLESOLVER_DLE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_LRDLESOLVER_DLE_DEPRECATED
#  define CASADI_LRDLESOLVER_DLE_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_LRDLESOLVER_DLE_DEPRECATED_EXPORT CASADI_LRDLESOLVER_DLE_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_LRDLESOLVER_DLE_DEPRECATED_NO_EXPORT CASADI_LRDLESOLVER_DLE_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_LRDLESOLVER_DLE_NO_DEPRECATED
#endif

#endif
