#ifndef klx_g_lib_h
#include "kcrt.h"

void KLC_cocoa_sample_func(KLC_var impl);
KLC_var KLC_cocoa_int_to_var(int i);
void KLC_cocoa_release(int i);
KLC_var KLC_cocoa_drawing_context();
void KLC_cocoa_fill_rect(double x, double y, double w, double h);
void KLC_cocoa_set_color(KLC_int r, KLC_int g, KLC_int b);

#endif/*klx_g_lib_h*/
