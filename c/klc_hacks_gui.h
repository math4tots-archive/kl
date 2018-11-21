#ifndef klc_os_fs_h
#define klc_os_fs_h

#include "klc_prelude.h"

typedef struct KLCNhacksZBguiZBApi KLCNhacksZBguiZBApi;
typedef struct KLCNhacksZBguiZBGraphicsContext KLCNhacksZBguiZBGraphicsContext;
typedef struct KLCNhacksZBguiZBFont KLCNhacksZBguiZBFont;
typedef struct KLCNhacksZBguiZBKeyEvent KLCNhacksZBguiZBKeyEvent;
struct KLCNhacksZBguiZBOptions;
struct KLCNhacksZBguiZBColor;

extern KLC_typeinfo KLC_typehacksZBguiZBApi;
extern KLC_typeinfo KLC_typehacksZBguiZBGraphicsContext;
extern KLC_typeinfo KLC_typehacksZBguiZBFont;
extern KLC_typeinfo KLC_typehacksZBguiZBKeyEvent;

void KLC_deletehacksZBguiZBApi(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBGraphicsContext(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBFont(KLC_header*, KLC_header**);
void KLC_deletehacksZBguiZBKeyEvent(KLC_header*, KLC_header**);

KLCNTry* KLCNhacksZBguiZBtryApiZEinit();

void KLCNhacksZBguiZBApiZFstart(KLCNhacksZBguiZBApi* api, struct KLCNhacksZBguiZBOptions*);
KLCNhacksZBguiZBGraphicsContext* KLCNhacksZBguiZBApiZFZAgetGc(KLCNhacksZBguiZBApi* api);
void KLCNhacksZBguiZBApiZFZAbeginDraw(KLCNhacksZBguiZBApi* api);
void KLCNhacksZBguiZBApiZFZAendDraw(KLCNhacksZBguiZBApi* api);

double KLCNhacksZBguiZBGraphicsContextZFGETwidth(KLCNhacksZBguiZBGraphicsContext* gc);
double KLCNhacksZBguiZBGraphicsContextZFGETheight(KLCNhacksZBguiZBGraphicsContext* gc);
KLCNList* KLCNhacksZBguiZBGraphicsContextZFfillTextSizeAsList(
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

KLCNString* KLCNhacksZBguiZBKeyEventZFGETchars(KLCNhacksZBguiZBKeyEvent* e);
KLC_bool KLCNhacksZBguiZBKeyEventZFGetItem(KLCNhacksZBguiZBKeyEvent* e, KLCNString* mod);

KLCNString* KLCNhacksZBguiZBFontZFGETname(KLCNhacksZBguiZBFont* font);
KLC_int KLCNhacksZBguiZBFontZFGETsize(KLCNhacksZBguiZBFont* font);

KLCNTry* KLCNhacksZBguiZBgetFontZDtry(KLCNString* name, KLC_int size);

/* BEGIN Function declarations for extracting data from Options */
KLC_int KLCNhacksZBguiZBOptionsZFGETx(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETy(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETwidth(struct KLCNhacksZBguiZBOptions* opts);
KLC_int KLCNhacksZBguiZBOptionsZFGETheight(struct KLCNhacksZBguiZBOptions* opts);
KLC_var KLCNhacksZBguiZBOptionsZFGETstartCallback(struct KLCNhacksZBguiZBOptions* opts);
KLC_var KLCNhacksZBguiZBOptionsZFGETkeyCallback(struct KLCNhacksZBguiZBOptions* opts);
KLC_var KLCNhacksZBguiZBOptionsZFGETmodifierKeyCallback(struct KLCNhacksZBguiZBOptions* opts);
/* END */

/* BEGIN Function declarations for extracting data from Color */
double KLCNhacksZBguiZBColorZFGETr(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETg(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETb(struct KLCNhacksZBguiZBColor* color);
double KLCNhacksZBguiZBColorZFGETa(struct KLCNhacksZBguiZBColor* color);
/* END */

#endif/*klc_os_fs_h*/
