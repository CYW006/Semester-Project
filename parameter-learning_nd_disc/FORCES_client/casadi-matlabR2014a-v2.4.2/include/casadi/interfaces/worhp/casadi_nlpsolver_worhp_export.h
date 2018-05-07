
#ifndef CASADI_NLPSOLVER_WORHP_EXPORT_H
#define CASADI_NLPSOLVER_WORHP_EXPORT_H

#ifdef CASADI_NLPSOLVER_WORHP_STATIC_DEFINE
#  define CASADI_NLPSOLVER_WORHP_EXPORT
#  define CASADI_NLPSOLVER_WORHP_NO_EXPORT
#else
#  ifndef CASADI_NLPSOLVER_WORHP_EXPORT
#    ifdef casadi_nlpsolver_worhp_EXPORTS
        /* We are building this library */
#      define CASADI_NLPSOLVER_WORHP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_NLPSOLVER_WORHP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_NLPSOLVER_WORHP_NO_EXPORT
#    define CASADI_NLPSOLVER_WORHP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_NLPSOLVER_WORHP_DEPRECATED
#  define CASADI_NLPSOLVER_WORHP_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_NLPSOLVER_WORHP_DEPRECATED_EXPORT CASADI_NLPSOLVER_WORHP_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_NLPSOLVER_WORHP_DEPRECATED_NO_EXPORT CASADI_NLPSOLVER_WORHP_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_NLPSOLVER_WORHP_NO_DEPRECATED
#endif

#endif
