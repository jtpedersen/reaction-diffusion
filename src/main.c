#include <SDL.h>
#include <stdio.h>
#include "reactdiffuse.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800

#define MATRIX_W 200
#define MATRIX_H 200

#define RATIO_X (WINDOW_WIDTH / MATRIX_W)
#define RATIO_Y (WINDOW_HEIGHT / MATRIX_H)

reaction_diffusion_system rds;

Uint32 SDL_MapHSV (SDL_PixelFormat *fmt, float h, float s, float v) {
    float r, g, b;
    int hi = (int)(h/60.f) %6;
    float f = h/60.f - (int)h/60;
    float p = v * (1.f - s);
    float q = v * (1.f - f*s);
    float t = v * (1.f - (1.f - f) * s);

    switch (hi) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    case 5: r = v, g = p, b = q; break;
    }

    return SDL_MapRGB(fmt, r*255, g*255, b*255);
}

void draw(SDL_Surface *surface) {
    SDL_Rect rect;
    rect.w = RATIO_X;
    rect.h = RATIO_Y;

    for (int y = 0; y < MATRIX_H; y++) {
	for (int x = 0; x < MATRIX_W; x++) {
            double u = reaction_diffusion_system_get(&rds, rds.U, x, y);
            double v = reaction_diffusion_system_get(&rds, rds.V, x, y);

            rect.x = RATIO_X * x;
            rect.y = RATIO_Y * y;

            SDL_FillRect(surface,
                         &rect,
                         SDL_MapHSV(surface->format, (int)(u * 1000) % 360, v, .9*v + .1*u));
        }
    }
}

int main(int argc, char **argv) {
    /* SDL SEtup */
    if ( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
    atexit(SDL_Quit);

    SDL_Surface *surface;

    surface = SDL_SetVideoMode(WINDOW_WIDTH, WINDOW_HEIGHT, 32, SDL_HWSURFACE);
    if ( surface == NULL ) {
        fprintf(stderr, "Could not setup screen to resolution %dx%d : %s\n", 
                WINDOW_WIDTH, WINDOW_HEIGHT, SDL_GetError());
        exit(1);
    }

    // default
    reaction_diffusion_system_init(&rds, MATRIX_W, MATRIX_H, .055, .062, 1.0, 0.5);

    // mitosis
    // reaction_diffusion_system_init(&rds, MATRIX_W, MATRIX_H, .0367, .0649, 1.0, 0.5);

    // coral
    // reaction_diffusion_system_init(&rds, MATRIX_W, MATRIX_H, .0545, .062, 1.0, 0.5);


    SDL_Event event;
    int leftPressed = 0;
    int rightPressed = 0;
    int mustExit = 0;

    float fps = 10.0;
    Uint32 start_time = SDL_GetTicks();

    
    while (!mustExit) {

        while (SDL_PollEvent(&event)) {
	    switch (event.type) {
	    case SDL_QUIT:
		mustExit = 1;
		break;
	    case SDL_MOUSEBUTTONDOWN:
	    case SDL_MOUSEBUTTONUP:
		switch (event.button.button) {
		case SDL_BUTTON_LEFT:
		    leftPressed = event.button.state == SDL_PRESSED;
		    break;
		case SDL_BUTTON_RIGHT:
		    rightPressed = event.button.state == SDL_PRESSED;
		    break;
		}
		break;
	    case SDL_MOUSEMOTION:
		if (leftPressed) {
		    reaction_diffusion_system_set(&rds, rds.V,
						  (int)(event.motion.x / RATIO_X),
						  (int)(event.motion.y / RATIO_Y),
						  1);
		}
	    }
	}

        draw(surface);
        SDL_Flip(surface);

        reaction_diffusion_system_update(&rds, 1.);
	// calc FPS
        Uint32 end_time = SDL_GetTicks();
        Uint32 dt = end_time - start_time;
        fps = fps * .9 + .1 * (1000.0 / dt); /* averaging */
        start_time = end_time;

	// display
        char buf[512];
        sprintf(buf, "Reaction Diffusion (%u %.2f fps)", dt, fps);
	printf("%s\n", buf);
        //SDL_SetWindowTitle(surface, buf);

    }

    reaction_diffusion_system_free(&rds);

    return 0;
}

