
#ifndef CASADI_SDQPSOLVER_SDP_EXPORT_H
#define CASADI_SDQPSOLVER_SDP_EXPORT_H

#ifdef CASADI_SDQPSOLVER_SDP_STATIC_DEFINE
#  define CASADI_SDQPSOLVER_SDP_EXPORT
#  define CASADI_SDQPSOLVER_SDP_NO_EXPORT
#else
#  ifndef CASADI_SDQPSOLVER_SDP_EXPORT
#    ifdef casadi_sdqpsolver_sdp_EXPORTS
        /* We are building this library */
#      define CASADI_SDQPSOLVER_SDP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CASADI_SDQPSOLVER_SDP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CASADI_SDQPSOLVER_SDP_NO_EXPORT
#    define CASADI_SDQPSOLVER_SDP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CASADI_SDQPSOLVER_SDP_DEPRECATED
#  define CASADI_SDQPSOLVER_SDP_DEPRECATED __attribute__ ((__deprecated__))
#  define CASADI_SDQPSOLVER_SDP_DEPRECATED_EXPORT CASADI_SDQPSOLVER_SDP_EXPORT __attribute__ ((__deprecated__))
#  define CASADI_SDQPSOLVER_SDP_DEPRECATED_NO_EXPORT CASADI_SDQPSOLVER_SDP_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CASADI_SDQPSOLVER_SDP_NO_DEPRECATED
#endif

#endif
