#include "reactdiffuse.h"
#include <math.h>

void reaction_diffusion_system_init(reaction_diffusion_system *s,
                                    size_t width,
                                    size_t height,
                                    double f,
                                    double k,
                                    double du,
                                    double dv) {

    s->width = width;
    s->height = height;
    s->f = f;
    s->k = k;
    s->du = du;
    s->dv = dv;
    s->U = (double*)malloc(width * height * sizeof(double));
    s->V = (double*)malloc(width * height * sizeof(double));
    s->swapU = (double*)malloc(width * height * sizeof(double));
    s->swapV = (double*)malloc(width * height * sizeof(double));

    // initialize
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            reaction_diffusion_system_set(s, s->U, x, y, 1);
            reaction_diffusion_system_set(s, s->V, x, y, 0);
        }
    }
}

void reaction_diffusion_system_free(reaction_diffusion_system *s) {
    free(s->U);
    free(s->V);
    free(s->swapU);
    free(s->swapV);
}

void calculate_laplacian(reaction_diffusion_system *s, double *from, double *to) {
    double v;

//#pragma omp parallel 
    for (int y = 1; y < s->height-1; y++) {
	for (int x = 1; x < s->width-1; x++) {
            v = .05 * reaction_diffusion_system_get_no_bounds_check(s, from, x-1, y-1) +
                .2  * reaction_diffusion_system_get_no_bounds_check(s, from, x-1, y  ) +
                .05 * reaction_diffusion_system_get_no_bounds_check(s, from, x-1, y+1) +
                .2  * reaction_diffusion_system_get_no_bounds_check(s, from, x  , y-1) +
                -1. * reaction_diffusion_system_get_no_bounds_check(s, from, x  , y  ) +
                .2  * reaction_diffusion_system_get_no_bounds_check(s, from, x  , y+1) +
                .05 * reaction_diffusion_system_get_no_bounds_check(s, from, x+1, y-1) +
                .2  * reaction_diffusion_system_get_no_bounds_check(s, from, x+1, y  ) +
                .05 * reaction_diffusion_system_get_no_bounds_check(s, from, x+1, y+1);
            reaction_diffusion_system_set(s, to, x, y, v);
        }
    }
}


void reaction_diffusion_system_update(reaction_diffusion_system *s, double dt) {
    // write laplacians calculation to swaps
    calculate_laplacian(s, s->U, s->swapU);
    calculate_laplacian(s, s->V, s->swapV);

    /* allocate temporary buffers */
    double *tmp_u = malloc(sizeof(double) * (s->width * s->height) );
    double *tmp_v = malloc(sizeof(double) * (s->width * s->height) );

    // calculate new concentrations, and write the new value to swaps
//#pragma omp parallel slows down due to access conflicts/locs - i guesss...
    for (int y = 1; y < s->height-1; y++) {
	for (int x = 1; x < s->width-1; x++) {
            double u = reaction_diffusion_system_get_no_bounds_check(s, s->U, x, y);
            double v = reaction_diffusion_system_get_no_bounds_check(s, s->V, x, y);
            double deltaU = s->du*reaction_diffusion_system_get_no_bounds_check(s, s->swapU, x, y)
                            - (u * v * v)
                            + s->f*(1. - u);
            double deltaV = s->dv*reaction_diffusion_system_get_no_bounds_check(s, s->swapV, x, y)
                            + (u * v * v)
                            - (s->k + s->f)*v;

            reaction_diffusion_system_set(s, tmp_u, x, y, u + deltaU*dt);
            reaction_diffusion_system_set(s, tmp_v, x, y, v + deltaV*dt);
        }
    }

    free(s->U);
    free(s->V);
    s->U = tmp_u;
    s->V = tmp_v;
}

double reaction_diffusion_system_get(reaction_diffusion_system *s, double *m, size_t x, size_t y) {
    x = (x + s->width)  % s->width;
    y = (y + s->height) % s->height;
    return m[y*s->width + x];
}

void reaction_diffusion_system_set(reaction_diffusion_system *s, double *m, size_t x, size_t y, double v) {
    x = (x + s->width)  % s->width;
    y = (y + s->height) % s->height;
    m[y*s->width + x] = fmin(1, fmax(-1, v));
}

double reaction_diffusion_system_get_no_bounds_check(reaction_diffusion_system *s, double *m, size_t x, size_t y) {
    return m[y*s->width + x];
}
