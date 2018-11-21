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
/* and stop using nsretain/nsrelease */

@class KLCOBJCView;
@class KLCOBJCAppDelegate;

struct KLCNhacksZBguiZBApi {
  KLC_header header;
  Options* opts;
  __unsafe_unretained KLCOBJCAppDelegate* appDelegate;
};

struct KLCNhacksZBguiZBGraphicsContext {
  KLC_header header;
  __unsafe_unretained NSImage* image;
  Font* font;
  Color* textForegroundColor;
  Color* textBackgroundColor;
};

struct KLCNhacksZBguiZBFont {
  KLC_header header;
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
  NSImage* layer;
  Options* opts;
}
@end

/* TODO: Use a hashmap for better performance semantics at some point */
static NSMutableArray<NSMutableArray*>* retainMap = nil;

static NSMutableArray<NSMutableArray*>* getRetainMap() {
  if (retainMap == nil) {
    retainMap = [[NSMutableArray alloc] init];
  }
  return retainMap;
}

static long getRefcnt(id obj) {
  for (NSArray* pair in getRetainMap()) {
    if ([pair objectAtIndex: 0] == obj) {
      return [[pair objectAtIndex: 1] longValue];
    }
  }
  return 0;
}

static void setRefcnt(id obj, long refcnt) {
  NSMutableArray<NSMutableArray*>* rm = getRetainMap();
  long len = (long) [rm count];
  long i;
  for (i = 0; i < len; i++) {
    NSMutableArray* pair = [rm objectAtIndex:i];
    if ([pair objectAtIndex: 0] == obj) {
      if (refcnt >= 0) {
        [pair
          replaceObjectAtIndex:1
          withObject:[NSNumber numberWithLong:refcnt]];
      } else {
        [rm removeObjectAtIndex:i];
      }
      return;
    }
  }
  if (refcnt >= 0) {
    NSMutableArray* pair = [[NSMutableArray alloc] init];
    [pair addObject:obj];
    [pair addObject:[NSNumber numberWithLong:refcnt]];
    [rm addObject:pair];
  }
}

static void nsretain(id obj) {
  setRefcnt(obj, getRefcnt(obj) + 1);
}

static void nsrelease(id obj) {
  setRefcnt(obj, getRefcnt(obj) - 1);
}

static NSMutableArray<NSObject*>* retainArray = nil;

static NSMutableArray<NSObject*>* getRetainSet() {
  if (retainArray == nil) {
    retainArray = [[NSMutableArray<NSObject*> alloc] init];
  }
  return retainArray;
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
  nsretain(font);
  return ret;
}

static GC* makeGC(NSImage* image) {
  GC* gc = (GC*) malloc(sizeof(GC));
  KLC_init_header((KLC_header*) gc, &KLC_typehacksZBguiZBGraphicsContext);
  gc->image = image;
  gc->font = NULL;
  gc->textForegroundColor = NULL;
  gc->textBackgroundColor = NULL;
  nsretain(image);
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
    [appDelegate->layer
      drawAtPoint:dirtyRect.origin
      fromRect:dirtyRect
      operation:NSCompositingOperationSourceOver
      fraction:1.0];
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
      [self setNeedsDisplayInRect:self.bounds];
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
    NSWindowStyleMask styleMask =
      NSWindowStyleMaskTitled |
      NSWindowStyleMaskClosable |
      NSWindowStyleMaskMiniaturizable;
    if (KLCNhacksZBguiZBOptionsZFGETresizable(xopts)) {
      styleMask |= NSWindowStyleMaskResizable;
    }
    window = [[NSWindow alloc]
      initWithContentRect: windowRect
      styleMask: styleMask
      backing: NSBackingStoreBuffered
      defer: NO];
    window.contentView = [[KLCOBJCView alloc] initWithAppDelegate: self];
    layer = [[NSImage alloc] initWithSize:windowRect.size];
    opts = xopts;
    KLC_retain((KLC_header*) opts);
  }
  return self;
}
- (void)prepareLayerAndRunStartCallback {
  KLC_var startCallback = KLCNhacksZBguiZBOptionsZFGETstartCallback(opts);
  if (KLC_truthy(startCallback)) {
    KLC_release_var(KLC_var_call(startCallback, 0, NULL));
  }
  KLC_release_var(startCallback);
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

void KLC_deletehacksZBguiZBApi(KLC_header* rapi, KLC_header** dq) {
  KLCNhacksZBguiZBApi* api = (KLCNhacksZBguiZBApi*) rapi;
  nsrelease(api->appDelegate);
}

void KLC_deletehacksZBguiZBGraphicsContext(KLC_header* robj, KLC_header** dq) {
  KLCNhacksZBguiZBGraphicsContext* gc = (KLCNhacksZBguiZBGraphicsContext*) robj;
  nsrelease(gc->image);
  KLC_partial_release((KLC_header*) gc->font, dq);
}

void KLC_deletehacksZBguiZBFont(KLC_header* robj, KLC_header** dq) {
  @autoreleasepool {
    KLCNhacksZBguiZBFont* font = (KLCNhacksZBguiZBFont*) robj;
    nsrelease(font->font);
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
  api->appDelegate = nil;
  ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) api));
  KLC_release((KLC_header*) api);
  return ret;
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
    api->appDelegate = appDelegate;
    nsretain(api->appDelegate);

    [appDelegate prepareLayerAndRunStartCallback];
    [NSApp run];
  }
}

struct KLCNhacksZBguiZBSize* KLCNhacksZBguiZBApiZFgetSize(KLCNhacksZBguiZBApi* api) {
  if (!api->appDelegate) {
    KLC_errorf("Api.start must be called before Api.getSize()");
  }
  @autoreleasepool {
    NSSize size = api->appDelegate->layer.size;
    return KLCNhacksZBguiZBSizeZEnew(size.width, size.height);
  }
}

KLCNhacksZBguiZBGraphicsContext* KLCNhacksZBguiZBApiZFZAgetGc(KLCNhacksZBguiZBApi* api) {
  if (!api->appDelegate) {
    KLC_errorf("Api.start must be called before Api._getGc()");
  }
  return makeGC(api->appDelegate->layer);
}

void KLCNhacksZBguiZBApiZFZAbeginDraw(KLCNhacksZBguiZBApi* api) {
  if (!api->appDelegate) {
    KLC_errorf("Api.start must be called before Api._beginDraw()");
  }
  [api->appDelegate->layer lockFocus];
}

void KLCNhacksZBguiZBApiZFZAendDraw(KLCNhacksZBguiZBApi* api) {
  if (!api->appDelegate) {
    KLC_errorf("Api.start must be called before Api._endDraw()");
  }
  [api->appDelegate->layer unlockFocus];
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
