#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

typedef struct KLCNhacksZBguiZBApi KLCNhacksZBguiZBApi;
typedef struct KLCNhacksZBguiZBGraphicsContext KLCNhacksZBguiZBGraphicsContext;
typedef struct KLCNhacksZBguiZBFont KLCNhacksZBguiZBFont;
struct KLCNhacksZBguiZBOptions;
struct KLCNhacksZBguiZBColor;

extern KLC_typeinfo KLC_typehacksZBguiZBApi;
extern KLC_typeinfo KLC_typehacksZBguiZBGraphicsContext;
extern KLC_typeinfo KLC_typehacksZBguiZBFont;

void KLC_deletehacksZBguiZBApi(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBGraphicsContext(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBFont(KLC_header*, KLC_header**);

KLCNTry* KLCNhacksZBguiZBtryApiZEinit();

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message);
void KLCNhacksZBguiZBApiZFstart(KLCNhacksZBguiZBApi* api, struct KLCNhacksZBguiZBOptions*);

double KLCNhacksZBguiZBGraphicsContextZFGETwidth(KLCNhacksZBguiZBGraphicsContext* gc);
double KLCNhacksZBguiZBGraphicsContextZFGETheight(KLCNhacksZBguiZBGraphicsContext* gc);
KLCNList* KLCNhacksZBguiZBGraphicsContextZFfillTextSize(
  KLCNhacksZBguiZBGraphicsContext* gc,
  KLCNString* text);
void KLCNhacksZBguiZBGraphicsContextZFsetFillColor(
  KLCNhacksZBguiZBGraphicsContext* gc,
  struct KLCNhacksZBguiZBColor* color);
void KLCNhacksZBguiZBGraphicsContextZFsetFont(
  KLCNhacksZBguiZBGraphicsContext* gc,
  KLCNhacksZBguiZBFont* font);
void KLCNhacksZBguiZBGraphicsContextZFsetTextForegroundColor(
  KLCNhacksZBguiZBGraphicsContext* gc,
  struct KLCNhacksZBguiZBColor* color);
void KLCNhacksZBguiZBGraphicsContextZFsetTextBackgroundColor(
  KLCNhacksZBguiZBGraphicsContext* gc,
  struct KLCNhacksZBguiZBColor* color);
void KLCNhacksZBguiZBGraphicsContextZFfillRect(
  KLCNhacksZBguiZBGraphicsContext* gc,
  double x, double y, double width, double height);
void KLCNhacksZBguiZBGraphicsContextZFfillText(
  KLCNhacksZBguiZBGraphicsContext* gc,
  double x, double y, KLCNString* text);

KLCNString* KLCNhacksZBguiZBFontZFGETname(KLCNhacksZBguiZBFont* font);
KLC_int KLCNhacksZBguiZBFontZFGETsize(KLCNhacksZBguiZBFont* font);

KLCNTry* KLCNhacksZBguiZBgetFontZDtry(KLCNString* name, KLC_int size);

/* BEGIN Function declarations for extracting data from Options */
KLC_int KLCNhacksZBguiZBOptionsZFGETx(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETy(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETwidth(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETheight(struct KLCNhacksZBguiZBOptions* opts);
KLC_var KLCNhacksZBguiZBOptionsZFGETdrawCallback(struct KLCNhacksZBguiZBOptions* opts);
/* END */

/* BEGIN Function declarations for extracting data from Color */
double KLCNhacksZBguiZBColorZFGETr(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETg(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETb(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETa(struct KLCNhacksZBguiZBColor* color);
/* END */

#endif/*klc_os_fs_h*/
