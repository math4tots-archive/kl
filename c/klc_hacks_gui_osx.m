#include "klc_hacks_gui.h"
#import "Cocoa/Cocoa.h"

typedef struct KLCNhacksZBguiZBOptions Options;
typedef struct KLCNhacksZBguiZBGraphicsContext GC;
typedef struct KLCNhacksZBguiZBColor Color;
typedef struct KLCNhacksZBguiZBFont Font;
typedef struct KLCNhacksZBguiZBKeyEvent KeyEvent;

/* NOTE: ARC is assumed/required here */

/* NOTE: You should always create an autoreleasepool */
/* when making calls to any objective-C libraries. */

/* TODO: Once ARC support for object pointers in structs is */
/* more widely available, remove all the '__unsafe_unretained' */
/* and stop using retainForever */

@class KLCOBJCView;
@class KLCOBJCAppDelegate;

struct KLCNhacksZBguiZBApi {
  KLC_header header;
  Options* opts;
};

struct KLCNhacksZBguiZBGraphicsContext {
  KLC_header header;

  /* This is safe to do because every onDraw cycle, we */
  /* check that graphics contexts are always released at the end */
  /* of the call. So they should never outlive the view that */
  /* spawned them */
  __unsafe_unretained KLCOBJCView* view;

  Font* font;
  Color* textForegroundColor;
  Color* textBackgroundColor;
};

struct KLCNhacksZBguiZBFont {
  KLC_header header;

  /* held with retainForever */
  __unsafe_unretained NSFont* font;
};

struct KLCNhacksZBguiZBKeyEvent {
  KLC_header header;
  KLCNString* chars;
  NSEventModifierFlags modifierFlags;
};

@interface KLCOBJCView: NSView {
  __weak KLCOBJCAppDelegate* appDelegate;
}
@end

@interface KLCOBJCAppDelegate: NSObject <NSApplicationDelegate> {
  NSWindow* window;
@public
  Options* opts;
}
@end

static NSMutableArray<NSObject*>* retainArray = nil;

static NSMutableArray<NSObject*>* getRetainSet() {
  if (retainArray == nil) {
    retainArray = [[NSMutableArray<NSObject*> alloc] init];
  }
  return retainArray;
}

static void retainForever(NSObject* obj) {
  [getRetainSet() addObject: obj];
}

static KeyEvent* makeKeyEvent(const char* utf8, NSEventModifierFlags flags) {
  KeyEvent* e = (KeyEvent*) malloc(sizeof(KeyEvent));
  KLC_init_header((KLC_header*) e, &KLC_typehacksZBguiZBKeyEvent);
  e->chars = KLC_mkstr(utf8);
  e->modifierFlags = flags;
  return e;
}

static Font* mkfont(NSFont* font) {
  Font* ret = (Font*) malloc(sizeof(Font));
  KLC_init_header((KLC_header*) ret, &KLC_typehacksZBguiZBFont);
  ret->font = font;
  retainForever(font);
  return ret;
}

static GC* makeGC(KLCOBJCView* view) {
  GC* gc = (GC*) malloc(sizeof(GC));
  KLC_init_header((KLC_header*) gc, &KLC_typehacksZBguiZBGraphicsContext);
  gc->view = view;
  gc->font = NULL;
  gc->textForegroundColor = NULL;
  gc->textBackgroundColor = NULL;
  return gc;
}

static NSColor* toNSColor(Color* color) {
  return [NSColor
    colorWithRed: KLCNhacksZBguiZBColorZFGETr(color)
    green: KLCNhacksZBguiZBColorZFGETg(color)
    blue: KLCNhacksZBguiZBColorZFGETb(color)
    alpha: KLCNhacksZBguiZBColorZFGETa(color)];
}

static NSDictionary* getDrawTextAttributes(KLCNhacksZBguiZBGraphicsContext* gc) {
  NSMutableDictionary* attrs = [[NSMutableDictionary alloc] init];
  [attrs setValue: gc->font->font forKey: NSFontAttributeName];
  if (gc->textForegroundColor) {
    [attrs
      setValue: toNSColor(gc->textForegroundColor)
      forKey: NSForegroundColorAttributeName];
  }
  if (gc->textBackgroundColor) {
    [attrs
      setValue: toNSColor(gc->textBackgroundColor)
      forKey: NSBackgroundColorAttributeName];
  }
  return attrs;
}

@implementation KLCOBJCView: NSView
- (id)initWithAppDelegate:(KLCOBJCAppDelegate*) appdel {
  if (self = [super init]) {
    appDelegate = appdel;
  }
  return self;
}
- (BOOL)acceptsFirstResponder {
  return YES;
}
- (void)drawRect:(NSRect)dirtyRect {
  @autoreleasepool {
    /* NOTE: There's an implicit assumption here that bounds.origin is always (0, 0) */
    NSRect bounds = self.bounds;
    GC* gc = makeGC(self);
    KLC_var drawCallback = KLCNhacksZBguiZBOptionsZFGETdrawCallback(appDelegate->opts);
    if (KLC_truthy(drawCallback)) {
      KLC_var gcvar = KLC_object_to_var((KLC_header*) gc);
      KLC_release_var(KLC_var_call(drawCallback, 1, &gcvar));
    }
    if (gc->header.refcnt) {
      KLC_errorf("Retaining GraphicsContext outside onDraw callback is illegal");
    }
    KLC_release_var(drawCallback);
    KLC_release((KLC_header*) gc);
  }
}
- (void)keyDown:(NSEvent*)event {
  @autoreleasepool {
    KLC_var keyCallback =
      KLCNhacksZBguiZBOptionsZFGETkeyCallback(appDelegate->opts);
    if (KLC_truthy(keyCallback)) {
      KeyEvent* ke = makeKeyEvent(
        [[event characters] UTF8String], [event modifierFlags]);
      KLC_var evar = KLC_object_to_var((KLC_header*) ke);
      KLC_release_var(KLC_var_call(keyCallback, 1, &evar));
      KLC_release((KLC_header*) ke);
    } else {
      [super keyDown: event];
    }
    KLC_release_var(keyCallback);
  }
}
- (void)flagsChanged:(NSEvent*)event {
  @autoreleasepool {
    KLC_var modifierKeyCallback =
      KLCNhacksZBguiZBOptionsZFGETmodifierKeyCallback(appDelegate->opts);
    if (KLC_truthy(modifierKeyCallback)) {
      KeyEvent* ke = makeKeyEvent("", [event modifierFlags]);
      KLC_var evar = KLC_object_to_var((KLC_header*) ke);
      KLC_release_var(KLC_var_call(modifierKeyCallback, 1, &evar));
      KLC_release((KLC_header*) ke);
    } else {
      [super flagsChanged: event];
    }
    KLC_release_var(modifierKeyCallback);
  }
}
@end


@implementation KLCOBJCAppDelegate: NSObject
- (id)initWithContentRect:(NSRect)windowRect options:(Options*)xopts {
  if (self = [super init]) {
    window = [[NSWindow alloc]
      initWithContentRect: windowRect
      styleMask:
        NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
      backing: NSBackingStoreBuffered
      defer: NO];
    window.contentView = [[KLCOBJCView alloc] initWithAppDelegate: self];
    opts = xopts;
    KLC_retain((KLC_header*) opts);
  }
  return self;
}
- (void)dealloc {
  KLC_release((KLC_header*) opts);
}
- (void)applicationWillFinishLaunching:(NSNotification *)notification {
  window.title = NSProcessInfo.processInfo.processName;
  [window cascadeTopLeftFromPoint: NSMakePoint(20,20)];
  [window makeKeyAndOrderFront: self];
}
@end

void KLC_deletehacksZBguiZBApi(KLC_header* api, KLC_header** dq) {
}

void KLC_deletehacksZBguiZBGraphicsContext(KLC_header* robj, KLC_header** dq) {
  KLCNhacksZBguiZBGraphicsContext* gc = (KLCNhacksZBguiZBGraphicsContext*) robj;
  KLC_partial_release((KLC_header*) gc->font, dq);
}

void KLC_deletehacksZBguiZBFont(KLC_header* robj, KLC_header** dq) {
  @autoreleasepool {
    KLCNhacksZBguiZBFont* font = (KLCNhacksZBguiZBFont*) robj;
    /* nsrelease(font->font); */
  }
}

void KLC_deletehacksZBguiZBKeyEvent(KLC_header* robj, KLC_header** dq) {
  KeyEvent* e = (KeyEvent*) robj;
  KLC_partial_release((KLC_header*) e->chars, dq);
}

KLCNTry* KLCNhacksZBguiZBtryApiZEinit() {
  KLCNTry* ret;
  KLCNhacksZBguiZBApi* api =
    (KLCNhacksZBguiZBApi*) malloc(sizeof(KLCNhacksZBguiZBApi));
  KLC_init_header(&api->header, &KLC_typehacksZBguiZBApi);
  api->opts = NULL;
  ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) api));
  KLC_release((KLC_header*) api);
  return ret;
}

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message) {
  @autoreleasepool {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:
      [NSString stringWithUTF8String:message->utf8]];
    [alert runModal];
  }
}

void KLCNhacksZBguiZBApiZFstart(KLCNhacksZBguiZBApi* api, Options* opts) {
  @autoreleasepool {
    NSApplication* app = NSApplication.sharedApplication;
    KLCOBJCAppDelegate* appDelegate = [[KLCOBJCAppDelegate alloc]
      initWithContentRect: NSMakeRect(
        KLCNhacksZBguiZBOptionsZFGETx(opts),
        KLCNhacksZBguiZBOptionsZFGETy(opts),
        KLCNhacksZBguiZBOptionsZFGETwidth(opts),
        KLCNhacksZBguiZBOptionsZFGETheight(opts))
      options: opts];
    NSMenuItem* item;

    app.ActivationPolicy = NSApplicationActivationPolicyRegular;
    item = NSMenuItem.new;
    NSApp.mainMenu = NSMenu.new;
    item.submenu = NSMenu.new;
    [app.mainMenu addItem: item];
    [item.submenu addItem: [[NSMenuItem alloc]
      initWithTitle: [@"Quit "
          stringByAppendingString: NSProcessInfo.processInfo.processName]
      action:@selector(terminate:) keyEquivalent:@"q"]];
    app.delegate = appDelegate;
    [NSApp run];
  }
}

double KLCNhacksZBguiZBGraphicsContextZFGETwidth(KLCNhacksZBguiZBGraphicsContext* gc) {
  return gc->view.bounds.size.width;
}

double KLCNhacksZBguiZBGraphicsContextZFGETheight(KLCNhacksZBguiZBGraphicsContext* gc) {
  return gc->view.bounds.size.height;
}

KLCNList* KLCNhacksZBguiZBGraphicsContextZFfillTextSizeAsList(
    KLCNhacksZBguiZBGraphicsContext* gc,
    KLCNString* text) {
  NSString* nstext = [NSString stringWithUTF8String: text->utf8];
  NSSize size = [nstext sizeWithAttributes: getDrawTextAttributes(gc)];
  KLCNList* ret = KLC_mklist(2);
  KLCNListZFpush(ret, KLC_double_to_var(size.width));
  KLCNListZFpush(ret, KLC_double_to_var(size.height));
  return ret;
}

void KLCNhacksZBguiZBGraphicsContextZFsetFillColor(
    KLCNhacksZBguiZBGraphicsContext* gc, Color* color) {
  [toNSColor(color) setFill];
}

void KLCNhacksZBguiZBGraphicsContextZFsetFont(
    KLCNhacksZBguiZBGraphicsContext* gc, Font* font) {
  KLC_retain((KLC_header*) font);
  KLC_release((KLC_header*) gc->font);
  gc->font = font;
}

void KLCNhacksZBguiZBGraphicsContextZFsetTextForegroundColor(
    KLCNhacksZBguiZBGraphicsContext* gc, Color* color) {
  KLC_retain((KLC_header*) color);
  KLC_release((KLC_header*) gc->textForegroundColor);
  gc->textForegroundColor = color;
}

void KLCNhacksZBguiZBGraphicsContextZFsetTextBackgroundColor(
    KLCNhacksZBguiZBGraphicsContext* gc, Color* color) {
  KLC_retain((KLC_header*) color);
  KLC_release((KLC_header*) gc->textBackgroundColor);
  gc->textBackgroundColor = color;
}

void KLCNhacksZBguiZBGraphicsContextZFfillRect(
    KLCNhacksZBguiZBGraphicsContext* gc,
    double x, double y, double width, double height) {
  @autoreleasepool {
    NSRectFill(NSMakeRect(x, y, width, height));
  }
}

void KLCNhacksZBguiZBGraphicsContextZFfillText(
    KLCNhacksZBguiZBGraphicsContext* gc,
    double x, double y, KLCNString* text) {
  @autoreleasepool {
    NSString* nstext = [NSString stringWithUTF8String: text->utf8];
    [nstext
      drawAtPoint: NSMakePoint(x, y)
      withAttributes:getDrawTextAttributes(gc)];
  }
}

KLCNString* KLCNhacksZBguiZBKeyEventZFGETchars(KLCNhacksZBguiZBKeyEvent* e) {
  KLC_retain((KLC_header*) e->chars);
  return e->chars;
}

KLC_bool KLCNhacksZBguiZBKeyEventZFGetItem(
    KLCNhacksZBguiZBKeyEvent* e, KLCNString* mod) {
  NSEventModifierFlags flag = 0;
  if (strcmp(mod->utf8, "shift") == 0) {
    flag = NSEventModifierFlagShift;
  } else if (strcmp(mod->utf8, "control") == 0) {
    flag = NSEventModifierFlagControl;
  } else if (strcmp(mod->utf8, "alt") == 0) {
    flag = NSEventModifierFlagOption;
  } else if (strcmp(mod->utf8, "command") == 0) {
    flag = NSEventModifierFlagCommand;
  }
  if (flag == 0) {
    KLC_errorf("Unrecognized modifier name");
  }
  return (e->modifierFlags & flag) ? 1 : 0;
}

KLC_bool KLCNhacksZBguiZBKeyEventZFGETshift(KLCNhacksZBguiZBKeyEvent* e) {
  return (e->modifierFlags & NSEventModifierFlagShift) ? 1 : 0;
}

KLC_bool KLCNhacksZBguiZBKeyEventZFGETcontrol(KLCNhacksZBguiZBKeyEvent* e) {
  return (e->modifierFlags & NSEventModifierFlagControl) ? 1 : 0;
}

KLC_bool KLCNhacksZBguiZBKeyEventZFGETalt(KLCNhacksZBguiZBKeyEvent* e) {
  return (e->modifierFlags & NSEventModifierFlagOption) ? 1 : 0;
}

KLC_bool KLCNhacksZBguiZBKeyEventZFGETcommand(KLCNhacksZBguiZBKeyEvent* e) {
  return (e->modifierFlags & NSEventModifierFlagCommand) ? 1 : 0;
}

KLCNString* KLCNhacksZBguiZBFontZFGETname(KLCNhacksZBguiZBFont* font) {
  return KLC_mkstr(font->font.fontName.UTF8String);
}

KLC_int KLCNhacksZBguiZBFontZFGETsize(KLCNhacksZBguiZBFont* font) {
  return (KLC_int) font->font.pointSize;
}

KLCNTry* KLCNhacksZBguiZBgetFontZDtry(KLCNString* name, KLC_int size) {
  @autoreleasepool {
    NSString* nsname = [NSString stringWithUTF8String: name->utf8];
    NSFont* nsfont = [NSFont fontWithName: nsname size: size];
    if (nsfont) {
      Font* font = mkfont(nsfont);
      KLCNTry* ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) font));
      KLC_release((KLC_header*) font);
      return ret;
    }
    return KLC_failm("Could not find font with given name");
  }
}
