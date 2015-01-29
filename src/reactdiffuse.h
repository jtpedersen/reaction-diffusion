#ifndef __REACT_DIFFUSE_H__
#define __REACT_DIFFUSE_H__

#include <stdlib.h>

#define DU 1.0
#define DV 0.5

typedef struct {
    size_t width;
    size_t height;
    double f;
    double k;
    double du;
    double dv;

    double *U;
    double *V;
    double *swapU;
    double *swapV;
} reaction_diffusion_system ;

void reaction_diffusion_system_init(reaction_diffusion_system *system,
                                    size_t width,
                                    size_t height,
                                    double f,
                                    double k,
                                    double du,
                                    double dv);

void reaction_diffusion_system_free(reaction_diffusion_system *system);

void reaction_diffusion_system_update(reaction_diffusion_system *system, double dt);

double reaction_diffusion_system_get(reaction_diffusion_system *system, double *m, size_t x, size_t y);

void reaction_diffusion_system_set(reaction_diffusion_system *system, double *m, size_t x, size_t y, double v);

double reaction_diffusion_system_get_no_bounds_check(reaction_diffusion_system *system, double *m, size_t x, size_t y);

#endif

