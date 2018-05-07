
#ifndef CASADI_SDPSOLVER_DSDP_EXPORT_H
#define CASADI_SDPSOLVER_DSDP_EXPORT_H

#ifdef CASADI_SDPSOLVER_DSDP_STATIC_DEFINE
#  define CASADI_SDPSOLVER_DSDP_EXPORT
#  define CASADI_SDPSOLVER_DSDP_NO_EXPORT
#else
#  ifndef CASADI_SDPSOLVER_DSDP_EXPORT
#    ifdef casadi_sdpsolver_dsdp_EXPORTS
        /* We are building this library */
#      define CASADI_SDPSOLVER_DSDP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_SDPSOLVER_DSDP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_SDPSOLVER_DSDP_NO_EXPORT
#    define CASADI_SDPSOLVER_DSDP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_SDPSOLVER_DSDP_DEPRECATED
#  define CASADI_SDPSOLVER_DSDP_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_SDPSOLVER_DSDP_DEPRECATED_EXPORT CASADI_SDPSOLVER_DSDP_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_SDPSOLVER_DSDP_DEPRECATED_NO_EXPORT CASADI_SDPSOLVER_DSDP_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_SDPSOLVER_DSDP_NO_DEPRECATED
#endif

#endif
